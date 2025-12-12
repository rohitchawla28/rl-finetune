# src/eval_utils.py
from typing import List, Tuple, Optional
import numpy as np
import torch
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration, PreTrainedModel, PreTrainedTokenizerBase

# cache metrics once
_ROUGE = evaluate.load("rouge")
_BLEU  = evaluate.load("bleu")


def get_dataset_keys(dataset_name: str) -> tuple:
    """Get text_key and ref_key for a dataset."""
    if dataset_name == "cnn_dailymail":
        return "article", "highlights"
    elif dataset_name == "xsum":
        return "document", "summary"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

@torch.no_grad()
def generate_summaries_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset,
    *,
    text_key: str = "article",
    ref_key: str  = "highlights",
    device: Optional[str] = None,
    batch_size: int = 16,
    max_input_len: int = 512,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    do_sample: bool = False,
) -> Tuple[List[str], List[str]]:
    """Fast batched generation for CNN/DM-style datasets."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds, refs = [], []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        enc = tokenizer(
            batch[text_key],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        ).to(device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        refs.extend(batch[ref_key])
    return preds, refs


def _repetition_rate(text: str) -> float:
    toks = text.lower().split()
    return 0.0 if len(toks) <= 1 else 1.0 - (len(set(toks)) / len(toks))


def metrics_table(preds: List[str], refs: List[str], save_path: Optional[str] = None) -> dict:
    """
    ROUGE (+ BLEU, compression, repetition) in one dict.
    
    Args:
        preds: List of prediction strings
        refs: List of reference strings
        save_path: Optional path to save results as JSON
    
    Returns:
        Dictionary of metrics
    """
    r = _ROUGE.compute(predictions=preds, references=refs)
    b = _BLEU.compute(predictions=preds, references=[[x] for x in refs])
    lens_pred = np.array([len(p.split()) for p in preds])
    lens_ref  = np.array([len(r.split()) for r in refs])
    compression = float(np.mean(lens_pred / np.maximum(1, lens_ref)))
    repetition  = float(np.mean([_repetition_rate(p) for p in preds]))
    r.update({"bleu": b["bleu"], "compression": compression, "repetition": repetition})
    
    if save_path:
        import json
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(r, f, indent=2)
    
    return r


@torch.no_grad()
def eval_model(
    model_id_or_path: str,
    eval_ds,
    *,
    text_key: str = "article",
    ref_key: str  = "highlights",
    batch_size: int = 16,
    num_beams: int = 4,
    max_input_len: int = 512,
    max_new_tokens: int = 128,
    tokenizer_class=None,
    model_class=None,
    save_path: Optional[str] = None,
) -> dict:
    """
    Convenience: load -> generate_batched -> metrics_table.
    By default uses T5Tokenizer/T5ForConditionalGeneration so local PPO/SFT checkpoints load cleanly.
    
    Args:
        model_id_or_path: Model path or HuggingFace model ID
        eval_ds: Evaluation dataset
        text_key: Key for input text (default "article" for CNN/DM, "document" for XSum)
        ref_key: Key for reference (default "highlights" for CNN/DM, "summary" for XSum)
        save_path: Optional path to save results JSON
    """
    Tok = tokenizer_class or T5Tokenizer
    Mdl = model_class or T5ForConditionalGeneration

    tok = Tok.from_pretrained(model_id_or_path)
    mdl = Mdl.from_pretrained(model_id_or_path)

    preds, refs = generate_summaries_batched(
        mdl, tok, eval_ds,
        text_key=text_key, ref_key=ref_key,
        batch_size=batch_size,
        max_input_len=max_input_len,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return metrics_table(preds, refs, save_path=save_path)