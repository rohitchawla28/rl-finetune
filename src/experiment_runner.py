"""
experiment_runner.py

Helper functions for running and organizing RL fine-tuning experiments.
Supports multiple reward functions, datasets, and hyperparameter configurations.
"""

from typing import Dict, List, Optional, Callable, Any
import os
import json
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.rewards import (
    rouge_l_rewards,
    composite_reward,
    brevity_reward,
    redundancy_penalty,
)
from src.ppo import run_ppo
from src.eval_utils import eval_model, generate_summaries_batched, metrics_table


def create_reward_fn(reward_config: Dict[str, Any]) -> Callable:
    """
    Create a reward function from configuration.
    
    Args:
        reward_config: Dictionary with reward function configuration
            - type: "rouge_l", "composite", etc.
            - Additional parameters based on type
    
    Returns:
        Reward function that takes (preds, refs) and returns List[torch.Tensor]
    """
    reward_type = reward_config.get("type", "rouge_l")
    
    if reward_type == "rouge_l":
        length_penalty = reward_config.get("length_penalty", 0.0)
        return lambda preds, refs, **kw: rouge_l_rewards(
            preds, refs, length_penalty=length_penalty
        )
    
    elif reward_type == "composite":
        components = reward_config.get("components", {})
        return lambda preds, refs, **kw: composite_reward(
            preds, refs,
            rouge_l_weight=components.get("rouge_l", 0.0),
            rouge1_weight=components.get("rouge1", 0.0),
            rouge2_weight=components.get("rouge2", 0.0),
            bleu_weight=components.get("bleu", 0.0),
            brevity_weight=components.get("brevity", 0.0),
            redundancy_penalty_weight=components.get("redundancy_penalty", 0.0),
            target_length=reward_config.get("target_length", 50),
            length_penalty=reward_config.get("length_penalty", 0.0),
        )
    
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def get_dataset_keys(dataset_name: str) -> tuple:
    """Get text_key and ref_key for a dataset."""
    if dataset_name == "cnn_dailymail":
        return "article", "highlights"
    elif dataset_name == "xsum":
        return "document", "summary"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_eval_dataset(dataset_name: str, n_eval: int = 1000):
    """Load evaluation dataset."""
    if dataset_name == "cnn_dailymail":
        return load_dataset("cnn_dailymail", "3.0.0")["validation"].select(range(n_eval))
    elif dataset_name == "xsum":
        return load_dataset("xsum")["validation"].select(range(n_eval))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_ppo_experiment(
    exp_config: Dict[str, Any],
    reward_functions: Dict[str, Dict],
    base_model_path: str,
    sft_model_path: Optional[str],
    eval_dataset,
    dataset_name: str,
    results_dir: str = "./results",
    verbosity: str = "summary",
) -> Dict[str, Any]:
    """
    Run a single PPO experiment.
    
    Args:
        exp_config: Experiment configuration
        reward_functions: Dictionary of reward function configs
        base_model_path: Path to base model
        sft_model_path: Path to SFT model (if starting from SFT)
        eval_dataset: Evaluation dataset
        dataset_name: Name of dataset (for field keys)
        results_dir: Directory to save results
        verbosity: Verbosity level
    
    Returns:
        Dictionary with experiment results
    """
    exp_name = exp_config["name"]
    print(f"\n{'='*70}")
    print(f"Running PPO Experiment: {exp_name}")
    print(f"{'='*70}")
    
    # Get starting model
    if exp_config["from"] == "base":
        start_path = base_model_path
    elif exp_config["from"] == "sft":
        if sft_model_path is None:
            raise ValueError("SFT model path required but not provided")
        start_path = sft_model_path
    else:
        raise ValueError(f"Unknown starting point: {exp_config['from']}")
    
    # Get reward function
    reward_name = exp_config["reward"]
    if reward_name not in reward_functions:
        raise ValueError(f"Unknown reward function: {reward_name}")
    
    reward_fn = create_reward_fn(reward_functions[reward_name])
    
    # Create output directory
    out_dir = os.path.join(results_dir, "checkpoints", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Run PPO
    try:
        ppo_dir = run_ppo(
            model_name=start_path,
            out_dir=out_dir,
            n_train=exp_config.get("n_train", 5000),
            batch_size=exp_config.get("batch_size", 64),
            epochs=exp_config.get("epochs", 1),
            lr=exp_config.get("lr", 1e-6),
            ppo_epochs=exp_config.get("ppo_epochs", 2),
            target_kl=exp_config.get("target_kl", 0.1),
            reward_fn=reward_fn,
            min_len=exp_config.get("min_len", 200),
            max_len=exp_config.get("max_len", 1200),
            max_new_tokens=exp_config.get("max_new_tokens", 64),
            min_new_tokens=exp_config.get("min_new_tokens", 8),
            temperature=exp_config.get("temperature", 0.7),
            top_p=exp_config.get("top_p", 0.8),
            top_k=exp_config.get("top_k", 50),
            whiten_rewards=exp_config.get("whiten_rewards", True),
            reward_clamp=exp_config.get("reward_clamp", 1.0),
            verbosity=verbosity,
            log_csv=os.path.join(out_dir, "train_log.csv"),
        )
        
        # Evaluate
        text_key, ref_key = get_dataset_keys(dataset_name)
        results_path = os.path.join(results_dir, f"results_{exp_name}.json")
        scores = eval_model(
            ppo_dir,
            eval_dataset,
            text_key=text_key,
            ref_key=ref_key,
            save_path=results_path,
        )
        
        print(f"\n[Results for {exp_name}]")
        print(f"  ROUGE-1: {scores['rouge1']:.4f}")
        print(f"  ROUGE-2: {scores['rouge2']:.4f}")
        print(f"  ROUGE-L: {scores['rougeL']:.4f}")
        print(f"  BLEU: {scores['bleu']:.4f}")
        print(f"  Compression: {scores['compression']:.4f}")
        print(f"  Repetition: {scores['repetition']:.4f}")
        
        return {
            "experiment": exp_name,
            "checkpoint_dir": ppo_dir,
            "results_path": results_path,
            "scores": scores,
            "success": True,
        }
    
    except Exception as e:
        print(f"\n[ERROR] Experiment {exp_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "experiment": exp_name,
            "success": False,
            "error": str(e),
        }


def compare_all_results(results_dir: str = "./results", output_file: Optional[str] = None):
    """
    Compare all experiment results and create a summary table.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Optional path to save comparison CSV
    """
    import pandas as pd
    import glob
    
    result_files = glob.glob(os.path.join(results_dir, "results_*.json"))
    
    if not result_files:
        print("No result files found!")
        return None
    
    all_results = {}
    for file in result_files:
        exp_name = os.path.basename(file).replace("results_", "").replace(".json", "")
        with open(file, 'r') as f:
            scores = json.load(f)
            all_results[exp_name] = scores
    
    df = pd.DataFrame(all_results).T
    
    # Select key metrics
    key_metrics = ["rouge1", "rouge2", "rougeL", "bleu", "compression", "repetition"]
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPARISON")
    print("="*70)
    print(df[available_metrics].round(4))
    
    if output_file:
        df[available_metrics].to_csv(output_file)
        print(f"\nSaved comparison to {output_file}")
    
    return df

