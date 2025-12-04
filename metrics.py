# metrics.py
from typing import List, Dict
from datasets import Dataset
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerBase
import evaluate
import torch

# Load ROUGE metric once
rouge = evaluate.load("rouge")


def generate_summaries(
    model: T5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    max_input_len: int = 512,
    max_target_len: int = 128,
    device: str = "cuda",
) -> (List[str], List[str]):
    """
    Generate summaries for a CNN/DailyMail-style dataset with 'article' and 'highlights' fields.
    Returns (predictions, references).
    """
    preds, refs = [], []
    model.to(device)
    model.eval()

    for ex in dataset:
        article = ex["article"]
        ref = ex["highlights"]

        inputs = tokenizer(
            article,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_target_len,
                num_beams=4,
            )

        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(pred)
        refs.append(ref)

    return preds, refs


def compute_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores given lists of predictions and references.
    """
    return rouge.compute(predictions=preds, references=refs)