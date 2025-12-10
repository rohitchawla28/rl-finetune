# src/eval_utils.py
from typing import List, Tuple, Optional
import numpy as np
import torch
import evaluate
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# cache metrics once
_ROUGE = evaluate.load("rouge")
_BLEU  = evaluate.load("bleu")

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


def metrics_table(preds: List[str], refs: List[str]) -> dict:
    """ROUGE (+ BLEU, compression, repetition) in one dict."""
    r = _ROUGE.compute(predictions=preds, references=refs)
    b = _BLEU.compute(predictions=preds, references=[[x] for x in refs])
    lens_pred = np.array([len(p.split()) for p in preds])
    lens_ref  = np.array([len(r.split()) for r in refs])
    compression = float(np.mean(lens_pred / np.maximum(1, lens_ref)))
    repetition  = float(np.mean([_repetition_rate(p) for p in preds]))
    r.update({"bleu": b["bleu"], "compression": compression, "repetition": repetition})
    return r


@torch.no_grad()
def eval_model(
    model_id_or_path: str,
    eval_ds,
    tokenizer_class=PreTrainedTokenizerBase,
    model_class=PreTrainedModel,
    *,
    text_key: str = "article",
    ref_key: str  = "highlights",
    batch_size: int = 16,
    num_beams: int = 4,
    max_input_len: int = 512,
    max_new_tokens: int = 128,
) -> dict:
    """Convenience: load -> generate_batched -> metrics_table."""
    from transformers import T5Tokenizer, T5ForConditionalGeneration  # default
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
    return metrics_table(preds, refs)