# src/rewards.py
"""
Reward functions for RL fine-tuning of summarization models.

Supports multiple reward designs as described in the paper:
- ROUGE-based rewards (ROUGE-1, ROUGE-2, ROUGE-L)
- Brevity rewards (encourage concise summaries)
- Redundancy penalties (discourage repetition)
- Composite rewards (combinations of above)
"""
from typing import List, Optional
import torch
import evaluate

_rouge = evaluate.load("rouge")
_bleu = evaluate.load("bleu")

def rouge_l_rewards(preds: List[str], refs: List[str], length_penalty: float = 0.0) -> List[torch.Tensor]:
    """
    Compute ROUGE-L rewards for predictions.
    Handles empty predictions by returning 0.0 reward.
    """
    # Filter out empty predictions and handle them
    valid_preds = []
    valid_refs = []
    empty_indices = []
    
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        if pred and pred.strip():
            valid_preds.append(pred)
            valid_refs.append(ref)
        else:
            empty_indices.append(i)
    
    # Compute scores for valid predictions
    if valid_preds:
        scores = _rouge.compute(
            predictions=valid_preds,
            references=valid_refs,
            rouge_types=["rougeL"],
            use_aggregator=False,
        )["rougeL"]
    else:
        scores = []
    
    # Build full reward list (0.0 for empty predictions)
    full_scores = []
    valid_idx = 0
    for i in range(len(preds)):
        if i in empty_indices:
            full_scores.append(0.0)
        else:
            score = scores[valid_idx] if valid_idx < len(scores) else 0.0
            if length_penalty and length_penalty > 0:
                score = score - length_penalty * len(valid_preds[valid_idx].split())
            full_scores.append(max(0.0, score))  # Ensure non-negative
            valid_idx += 1
    
    return [torch.tensor(s, dtype=torch.float32) for s in full_scores]


def brevity_reward(preds: List[str], target_length: int = 50, penalty_factor: float = 0.01) -> List[float]:
    """
    Reward for concise summaries. Penalizes summaries that are too long.
    
    Args:
        preds: List of prediction strings
        target_length: Target summary length in words
        penalty_factor: How much to penalize per word over target
    
    Returns:
        List of reward scores (higher for shorter summaries)
    """
    rewards = []
    for pred in preds:
        if not pred or not pred.strip():
            rewards.append(0.0)
            continue
        
        length = len(pred.split())
        if length <= target_length:
            # Reward for being concise
            reward = 1.0 - (length / target_length) * 0.2  # Slight penalty even if under
        else:
            # Penalize excess length
            excess = length - target_length
            reward = max(0.0, 1.0 - (excess / target_length) * penalty_factor * 100)
        rewards.append(reward)
    return rewards


def redundancy_penalty(preds: List[str]) -> List[float]:
    """
    Penalty for repetitive text. Higher penalty for more repetition.
    
    Returns:
        List of penalty scores (negative values = penalties)
    """
    penalties = []
    for pred in preds:
        if not pred or not pred.strip():
            penalties.append(0.0)
            continue
        
        words = pred.lower().split()
        if len(words) <= 1:
            penalties.append(0.0)
        else:
            unique_ratio = len(set(words)) / len(words)
            # Penalty increases as unique_ratio decreases (more repetition)
            penalty = (1.0 - unique_ratio) * 0.5  # Scale to reasonable range
            penalties.append(-penalty)  # Negative = penalty
    return penalties


def composite_reward(
    preds: List[str],
    refs: List[str],
    rouge_l_weight: float = 1.0,
    rouge1_weight: float = 0.0,
    rouge2_weight: float = 0.0,
    bleu_weight: float = 0.0,
    brevity_weight: float = 0.0,
    redundancy_penalty_weight: float = 0.0,
    target_length: int = 50,
    length_penalty: float = 0.0,
) -> List[torch.Tensor]:
    """
    Composite reward combining multiple metrics.
    
    This is the core reward function for your experiments!
    Allows testing different reward designs as described in your paper.
    
    Args:
        preds: List of prediction strings
        refs: List of reference strings
        rouge_l_weight: Weight for ROUGE-L score
        rouge1_weight: Weight for ROUGE-1 score
        rouge2_weight: Weight for ROUGE-2 score
        bleu_weight: Weight for BLEU score
        brevity_weight: Weight for brevity reward
        redundancy_penalty_weight: Weight for redundancy penalty (usually negative)
        target_length: Target summary length for brevity
        length_penalty: Additional length penalty for ROUGE-L
    
    Returns:
        List of combined reward tensors
    """
    rewards = []
    
    # ROUGE-L
    if rouge_l_weight > 0:
        rouge_l_scores = rouge_l_rewards(preds, refs, length_penalty=length_penalty)
        rewards.append([r.item() * rouge_l_weight for r in rouge_l_scores])
    
    # ROUGE-1 and ROUGE-2
    if rouge1_weight > 0 or rouge2_weight > 0:
        # Filter empty predictions
        valid_preds = []
        valid_refs = []
        for p, r in zip(preds, refs):
            if p and p.strip():
                valid_preds.append(p)
                valid_refs.append(r)
        
        if valid_preds:
            rouge_scores = _rouge.compute(
                predictions=valid_preds,
                references=valid_refs,
                rouge_types=["rouge1", "rouge2"],
                use_aggregator=False,
            )
            
            # Map back to full list (0.0 for empty)
            rouge1_full = []
            rouge2_full = []
            valid_idx = 0
            for pred in preds:
                if pred and pred.strip():
                    rouge1_full.append(rouge_scores["rouge1"][valid_idx] if valid_idx < len(rouge_scores["rouge1"]) else 0.0)
                    rouge2_full.append(rouge_scores["rouge2"][valid_idx] if valid_idx < len(rouge_scores["rouge2"]) else 0.0)
                    valid_idx += 1
                else:
                    rouge1_full.append(0.0)
                    rouge2_full.append(0.0)
            
            if rouge1_weight > 0:
                rewards.append([s * rouge1_weight for s in rouge1_full])
            if rouge2_weight > 0:
                rewards.append([s * rouge2_weight for s in rouge2_full])
        else:
            if rouge1_weight > 0:
                rewards.append([0.0] * len(preds))
            if rouge2_weight > 0:
                rewards.append([0.0] * len(preds))
    
    # BLEU (compute per-example)
    if bleu_weight > 0:
        bleu_full = []
        for pred, ref in zip(preds, refs):
            if pred and pred.strip():
                try:
                    bleu_score = _bleu.compute(
                        predictions=[pred],
                        references=[[ref]],
                    ).get("bleu", 0.0)
                    bleu_full.append(bleu_score)
                except:
                    bleu_full.append(0.0)
            else:
                bleu_full.append(0.0)
        rewards.append([s * bleu_weight for s in bleu_full])
    
    # Brevity
    if brevity_weight > 0:
        brevity_scores = brevity_reward(preds, target_length=target_length)
        rewards.append([s * brevity_weight for s in brevity_scores])
    
    # Redundancy penalty
    if redundancy_penalty_weight > 0:
        redundancy_scores = redundancy_penalty(preds)
        rewards.append([s * redundancy_penalty_weight for s in redundancy_scores])
    
    # Combine all rewards
    if not rewards:
        return [torch.tensor(0.0, dtype=torch.float32) for _ in preds]
    
    combined = []
    for i in range(len(preds)):
        total = sum(r[i] for r in rewards)
        combined.append(torch.tensor(total, dtype=torch.float32))
    
    return combined