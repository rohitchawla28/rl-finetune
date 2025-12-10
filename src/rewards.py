# src/rewards.py
from typing import List
import torch
import evaluate

_rouge = evaluate.load("rouge")

def rouge_l_rewards(preds: List[str], refs: List[str], length_penalty: float = 0.0) -> List[torch.Tensor]:
    scores = _rouge.compute(
        predictions=preds,
        references=refs,
        rouge_types=["rougeL"],
        use_aggregator=False,
    )["rougeL"]
    if length_penalty and length_penalty > 0:
        return [torch.tensor(s - length_penalty * len(p.split()), dtype=torch.float32)
                for s, p in zip(scores, preds)]
    return [torch.tensor(s, dtype=torch.float32) for s in scores]