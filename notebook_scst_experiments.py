"""
SCST Experiments for Notebook
=============================
Copy this code into your notebook cells to run SCST experiments.

SCST (Self-Critical Sequence Training) uses a greedy baseline and sampled
generations to compute advantages for policy gradient updates.
"""

# ============================================================================
# CELL 1: Setup and Imports
# ============================================================================

import torch
import importlib
from src.experiment_runner import (
    run_scst_experiment,
    load_eval_dataset,
    get_dataset_keys,
    compare_all_results
)
from src.eval_utils import eval_model

# Reload modules to pick up any code changes
importlib.reload(__import__('src.scst', fromlist=['']))
importlib.reload(__import__('src.experiment_runner', fromlist=['']))
importlib.reload(__import__('src.eval_utils', fromlist=['']))

print("✅ All modules reloaded and imported!")

# ============================================================================
# CELL 2: SCST Experiment Configuration
# ============================================================================

# Define SCST experiments
# Note: SCST uses ROUGE-L as reward (built-in, not configurable like PPO)
scst_experiments = [
    {
        "name": "scst_base_rouge_l",
        "from": "base",  # Start from base model
        "n_train": 2000,
        "batch_size": 8,
        "epochs": 1,
        "lr": 1e-6,
        "warmup_ratio": 0.1,
        "greedy_beams": 4,
        "greedy_max_new": 64,
        "sample_max_new": 64,
        "sample_min_new": 8,
        "top_k": 30,
        "top_p": 0.8,
        "temperature": 0.7,
        "no_repeat_ngram_size": 4,
        "max_input_len": 512,
        "max_target_len": 128,
        "advantage_normalize": True,
        "reward_clip": 0.5,
        "min_len": 200,
        "max_len": 1200,
        "seed": 42,
    },
    {
        "name": "scst_sft_rouge_l",
        "from": "sft",  # Start from SFT model
        "n_train": 2000,
        "batch_size": 8,
        "epochs": 1,
        "lr": 1e-6,
        "warmup_ratio": 0.1,
        "greedy_beams": 4,
        "greedy_max_new": 64,
        "sample_max_new": 64,
        "sample_min_new": 8,
        "top_k": 30,
        "top_p": 0.8,
        "temperature": 0.7,
        "no_repeat_ngram_size": 4,
        "max_input_len": 512,
        "max_target_len": 128,
        "advantage_normalize": True,
        "reward_clip": 0.5,
        "min_len": 200,
        "max_len": 1200,
        "seed": 42,
    },
    # Add more experiments with different hyperparameters
    {
        "name": "scst_sft_rouge_l_lr5e7",
        "from": "sft",
        "n_train": 2000,
        "batch_size": 8,
        "epochs": 1,
        "lr": 5e-7,  # Lower learning rate
        "warmup_ratio": 0.1,
        "greedy_beams": 4,
        "greedy_max_new": 64,
        "sample_max_new": 64,
        "sample_min_new": 8,
        "top_k": 30,
        "top_p": 0.8,
        "temperature": 0.7,
        "no_repeat_ngram_size": 4,
        "max_input_len": 512,
        "max_target_len": 128,
        "advantage_normalize": True,
        "reward_clip": 0.5,
        "min_len": 200,
        "max_len": 1200,
        "seed": 42,
    },
    {
        "name": "scst_sft_rouge_l_no_clip",
        "from": "sft",
        "n_train": 2000,
        "batch_size": 8,
        "epochs": 1,
        "lr": 1e-6,
        "warmup_ratio": 0.1,
        "greedy_beams": 4,
        "greedy_max_new": 64,
        "sample_max_new": 64,
        "sample_min_new": 8,
        "top_k": 30,
        "top_p": 0.8,
        "temperature": 0.7,
        "no_repeat_ngram_size": 4,
        "max_input_len": 512,
        "max_target_len": 128,
        "advantage_normalize": True,
        "reward_clip": None,  # No reward clipping
        "min_len": 200,
        "max_len": 1200,
        "seed": 42,
    },
]

print(f"✅ Configured {len(scst_experiments)} SCST experiments")

# ============================================================================
# CELL 3: Load Evaluation Dataset
# ============================================================================

# Load evaluation dataset (same as used for PPO)
eval_raw = load_eval_dataset(CFG["dataset"], n_eval=1000)
text_key, ref_key = get_dataset_keys(CFG["dataset"])

print(f"✅ Loaded evaluation dataset: {len(eval_raw)} examples")
print(f"   Dataset: {CFG['dataset']}")
print(f"   Text key: {text_key}, Reference key: {ref_key}")

# ============================================================================
# CELL 4: Run All SCST Experiments
# ============================================================================

scst_results = {}

for exp in scst_experiments:
    result = run_scst_experiment(
        exp_config=exp,
        base_model_path=CFG["model_name"],
        sft_model_path=best_dir,  # Use best SFT model
        eval_dataset=eval_raw,
        dataset_name=CFG["dataset"],
        results_dir=CFG["results_dir"],
        verbosity="summary",  # Use "steps" for more detailed output
    )
    scst_results[exp["name"]] = result

print("\n" + "="*70)
print("ALL SCST EXPERIMENTS COMPLETED")
print("="*70)

# ============================================================================
# CELL 5: Compare SCST Results
# ============================================================================

print("\n" + "="*70)
print("SCST EXPERIMENTS SUMMARY")
print("="*70)

for exp_name, result in scst_results.items():
    if result.get("success", False):
        scores = result["scores"]
        print(f"\n{exp_name}:")
        print(f"  ROUGE-1: {scores['rouge1']:.4f}")
        print(f"  ROUGE-2: {scores['rouge2']:.4f}")
        print(f"  ROUGE-L: {scores['rougeL']:.4f}")
        print(f"  BLEU: {scores['bleu']:.4f}")
        print(f"  Compression: {scores['compression']:.4f}")
        print(f"  Repetition: {scores['repetition']:.4f}")
    else:
        print(f"\n{exp_name}: FAILED")
        print(f"  Error: {result.get('error', 'Unknown error')}")

# ============================================================================
# CELL 6: Compare All Results (SFT, PPO, SCST)
# ============================================================================

# Compare all results including SFT, PPO, and SCST
print("\n" + "="*70)
print("COMPREHENSIVE RESULTS COMPARISON")
print("="*70)

# Load base model results
base_scores = eval_model(CFG["model_name"], eval_raw, text_key=text_key, ref_key=ref_key)
print("\nBase Model:")
print(f"  ROUGE-L: {base_scores['rougeL']:.4f}")

# SFT results (from your earlier training)
print("\nSFT (Best):")
print(f"  ROUGE-L: 0.1820")  # From your SFT results

# SCST results
print("\nSCST Experiments:")
for exp_name, result in scst_results.items():
    if result.get("success", False):
        print(f"  {exp_name}: ROUGE-L = {result['scores']['rougeL']:.4f}")

# If you have PPO results, add them here too
# print("\nPPO Experiments:")
# for exp_name, result in ppo_results.items():
#     if result.get("success", False):
#         print(f"  {exp_name}: ROUGE-L = {result['scores']['rougeL']:.4f}")

# ============================================================================
# CELL 7: Generate Comparison Table (Optional)
# ============================================================================

# Create a comparison table using pandas
try:
    import pandas as pd
    
    # Collect all results
    comparison_data = {}
    
    # Base model
    comparison_data["base"] = {
        "rouge1": base_scores["rouge1"],
        "rouge2": base_scores["rouge2"],
        "rougeL": base_scores["rougeL"],
        "bleu": base_scores["bleu"],
    }
    
    # SFT
    # You can load SFT results if saved, or use known values
    comparison_data["sft_best"] = {
        "rouge1": 0.0,  # Fill in from your SFT results
        "rouge2": 0.0,
        "rougeL": 0.1820,
        "bleu": 0.0,
    }
    
    # SCST experiments
    for exp_name, result in scst_results.items():
        if result.get("success", False):
            scores = result["scores"]
            comparison_data[exp_name] = {
                "rouge1": scores["rouge1"],
                "rouge2": scores["rouge2"],
                "rougeL": scores["rougeL"],
                "bleu": scores["bleu"],
            }
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data).T
    df = df.round(4)
    
    print("\n" + "="*70)
    print("RESULTS COMPARISON TABLE")
    print("="*70)
    print(df)
    
    # Save to CSV
    csv_path = f"{CFG['results_dir']}/scst_comparison.csv"
    df.to_csv(csv_path)
    print(f"\n✅ Saved comparison table to {csv_path}")
    
except ImportError:
    print("Pandas not available, skipping table generation")

# ============================================================================
# ALTERNATIVE: Direct SCST Training (Without Experiment Runner)
# ============================================================================

# If you prefer to call SCST directly without the experiment runner:

"""
from src.scst import run_scst

# Run a single SCST experiment
scst_dir = run_scst(
    model_name=best_dir,  # Start from SFT model
    out_dir="./results/checkpoints/scst_direct",
    n_train=2000,
    batch_size=8,
    epochs=1,
    lr=1e-6,
    warmup_ratio=0.1,
    greedy_beams=4,
    greedy_max_new=64,
    sample_max_new=64,
    sample_min_new=8,
    top_k=30,
    top_p=0.8,
    temperature=0.7,
    no_repeat_ngram_size=4,
    max_input_len=512,
    max_target_len=128,
    advantage_normalize=True,
    reward_clip=0.5,
    dataset_name=CFG["dataset"],
    min_len=200,
    max_len=1200,
    seed=42,
    debug=True,  # Set to False for less output
)

# Evaluate
scores = eval_model(scst_dir, eval_raw, text_key=text_key, ref_key=ref_key)
print(f"ROUGE-L: {scores['rougeL']:.4f}")
"""

