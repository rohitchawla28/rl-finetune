# src/eval/metrics.py
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerBase
import evaluate, torch

_rouge = evaluate.load("rouge")

def generate_summaries(
    model: T5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    max_input_len: int = 512,
    max_target_len: int = 128,
    device: Optional[str] = None,
):
    preds, refs = [], []
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    prefix = "summarize: "
    for ex in dataset:
        inputs = tokenizer(prefix + ex["article"], return_tensors="pt", truncation=True, max_length=max_input_len).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=max_target_len, num_beams=4)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
        refs.append(ex["highlights"])
    return preds, refs

def compute_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    return _rouge.compute(predictions=preds, references=refs, use_stemmer=True)