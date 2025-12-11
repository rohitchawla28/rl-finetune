# src/eval_utils.py
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import evaluate
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Cache metrics once
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
    """
    Fast batched generation for CNN/DM- or XSum-style datasets.
    Returns (predictions, references).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds, refs = [], []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
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
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            use_cache=True,
        )
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        refs.extend(batch[ref_key])
    return preds, refs


def _repetition_rate(text: str) -> float:
    toks = text.lower().split()
    return 0.0 if len(toks) <= 1 else 1.0 - (len(set(toks)) / len(toks))


def metrics_table(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """
    ROUGE (1/2/L) + BLEU + compression + repetition in one dict.
    """
    rouge_scores = _ROUGE.compute(predictions=preds, references=refs)
    bleu_scores  = _BLEU.compute(predictions=preds, references=[[x] for x in refs])

    lens_pred = np.array([len(p.split()) for p in preds], dtype=float)
    lens_ref  = np.array([len(ref.split()) for ref in refs], dtype=float)
    compression = float(np.mean(lens_pred / np.maximum(1.0, lens_ref)))
    repetition  = float(np.mean([_repetition_rate(p) for p in preds]))

    rouge_scores.update({
        "bleu": bleu_scores["bleu"],
        "compression": compression,
        "repetition": repetition,
    })
    return rouge_scores


@torch.no_grad()
def eval_model(
    model_id_or_path: str,
    eval_ds,
    tokenizer_class=None,
    model_class=None,
    *,
    text_key: str = "article",
    ref_key: str  = "highlights",
    batch_size: int = 16,
    num_beams: int = 4,
    max_input_len: int = 512,
    max_new_tokens: int = 128,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Convenience: load -> generate_batched -> metrics_table.
    Defaults to Auto classes; override tokenizer_class/model_class if needed.
    """
    Tok = tokenizer_class or AutoTokenizer
    Mdl = model_class or AutoModelForSeq2SeqLM

    tok = Tok.from_pretrained(model_id_or_path)
    mdl = Mdl.from_pretrained(model_id_or_path)

    preds, refs = generate_summaries_batched(
        mdl,
        tok,
        eval_ds,
        text_key=text_key,
        ref_key=ref_key,
        device=device,
        batch_size=batch_size,
        max_input_len=max_input_len,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )
    return metrics_table(preds, refs)