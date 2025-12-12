# notebook_ablation_study.py
"""
Ablation study to test each component individually.
This demonstrates the contribution of each design decision.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.scst import run_scst
from src.eval_utils import eval_model, get_dataset_keys
from src.analysis import ablation_study

# ============================================================================
# CELL 1: Setup
# ============================================================================

CFG = {
    "dataset": "cnn_dailymail",
    "base_model": "google/flan-t5-small",
    "sft_model_path": "./checkpoints/flan_t5_sft/best",  # Update with your path
    "results_dir": "./results/ablation",
    "n_train": 1000,  # Smaller for faster ablation
    "n_eval": 500,    # Smaller for faster evaluation
    "seed": 42,
}

os.makedirs(CFG["results_dir"], exist_ok=True)

print("✅ Ablation study configuration loaded!")

# ============================================================================
# CELL 2: Load Data
# ============================================================================

print("\n" + "="*70)
print("Loading data...")
print("="*70)

if CFG["dataset"] == "cnn_dailymail":
    train_raw = load_dataset("cnn_dailymail", "3.0.0")["train"].select(range(CFG["n_train"]))
    eval_raw = load_dataset("cnn_dailymail", "3.0.0")["validation"].select(range(CFG["n_eval"]))
    text_key, ref_key = "article", "highlights"
else:
    raise ValueError(f"Unknown dataset: {CFG['dataset']}")

print(f"✅ Loaded {len(train_raw)} training and {len(eval_raw)} eval examples")

# ============================================================================
# CELL 3: Baseline Models (for comparison)
# ============================================================================

print("\n" + "="*70)
print("Evaluating baseline models...")
print("="*70)

baseline_scores = {}

# Base model
print("Evaluating base model...")
base_scores = eval_model(CFG["base_model"], eval_raw, text_key=text_key, ref_key=ref_key)
baseline_scores["base"] = base_scores
print(f"  Base ROUGE-L: {base_scores['rougeL']:.4f}")

# SFT model (if exists)
if os.path.exists(CFG["sft_model_path"]):
    print("Evaluating SFT model...")
    sft_scores = eval_model(CFG["sft_model_path"], eval_raw, text_key=text_key, ref_key=ref_key)
    baseline_scores["sft"] = sft_scores
    print(f"  SFT ROUGE-L: {sft_scores['rougeL']:.4f}")
else:
    print("⚠️  SFT model not found, skipping")

# ============================================================================
# CELL 4: Ablation 1: Warm-Starting (Base vs SFT)
# ============================================================================

print("\n" + "="*70)
print("ABLATION 1: Warm-Starting Effect")
print("="*70)
print("Testing SCST from base vs SCST from SFT")

ablation1_configs = []

# SCST from base (no warm-starting)
if True:  # Set to False to skip if already done
    print("\nTraining SCST from base...")
    out_dir_base = os.path.join(CFG["results_dir"], "scst_from_base")
    try:
        run_scst(
            model_name=CFG["base_model"],
            out_dir=out_dir_base,
            n_train=CFG["n_train"],
            batch_size=8,
            epochs=1,
            lr=1e-6,
            warmup_ratio=0.1,
            train_dataset=train_raw,
            text_key=text_key,
            ref_key=ref_key,
            seed=CFG["seed"],
        )
        scores_base = eval_model(out_dir_base, eval_raw, text_key=text_key, ref_key=ref_key)
        ablation1_configs.append(("scst_from_base", out_dir_base))
        print(f"  ✅ SCST from base ROUGE-L: {scores_base['rougeL']:.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# SCST from SFT (with warm-starting)
if os.path.exists(CFG["sft_model_path"]):
    print("\nTraining SCST from SFT...")
    out_dir_sft = os.path.join(CFG["results_dir"], "scst_from_sft")
    try:
        run_scst(
            model_name=CFG["sft_model_path"],
            out_dir=out_dir_sft,
            n_train=CFG["n_train"],
            batch_size=8,
            epochs=1,
            lr=1e-6,
            warmup_ratio=0.1,
            train_dataset=train_raw,
            text_key=text_key,
            ref_key=ref_key,
            seed=CFG["seed"],
        )
        scores_sft = eval_model(out_dir_sft, eval_raw, text_key=text_key, ref_key=ref_key)
        ablation1_configs.append(("scst_from_sft", out_dir_sft))
        print(f"  ✅ SCST from SFT ROUGE-L: {scores_sft['rougeL']:.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# ============================================================================
# CELL 5: Ablation 2: Stability Mechanisms
# ============================================================================

print("\n" + "="*70)
print("ABLATION 2: Stability Mechanisms")
print("="*70)
print("Testing effect of decoding constraints, reward clamping, data filtering")

if not os.path.exists(CFG["sft_model_path"]):
    print("⚠️  SFT model required for this ablation, skipping")
else:
    ablation2_configs = []
    
    # Full configuration (all stability mechanisms)
    print("\nTraining with ALL stability mechanisms...")
    out_dir_full = os.path.join(CFG["results_dir"], "scst_full_stability")
    try:
        run_scst(
            model_name=CFG["sft_model_path"],
            out_dir=out_dir_full,
            n_train=CFG["n_train"],
            batch_size=8,
            epochs=1,
            lr=1e-6,
            warmup_ratio=0.1,
            train_dataset=train_raw,
            text_key=text_key,
            ref_key=ref_key,
            # Full stability: decoding constraints + reward clamping + data filtering
            no_repeat_ngram_size=4,
            top_p=0.75,
            reward_clip=0.5,
            min_len=200,
            max_len=1200,
            seed=CFG["seed"],
        )
        scores_full = eval_model(out_dir_full, eval_raw, text_key=text_key, ref_key=ref_key)
        ablation2_configs.append(("full_stability", out_dir_full))
        print(f"  ✅ Full stability ROUGE-L: {scores_full['rougeL']:.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # No decoding constraints
    print("\nTraining WITHOUT decoding constraints...")
    out_dir_no_constraints = os.path.join(CFG["results_dir"], "scst_no_constraints")
    try:
        run_scst(
            model_name=CFG["sft_model_path"],
            out_dir=out_dir_no_constraints,
            n_train=CFG["n_train"],
            batch_size=8,
            epochs=1,
            lr=1e-6,
            warmup_ratio=0.1,
            train_dataset=train_raw,
            text_key=text_key,
            ref_key=ref_key,
            # No constraints: remove no_repeat_ngram, use greedy decoding
            no_repeat_ngram_size=0,
            top_p=1.0,  # Greedy
            reward_clip=0.5,
            min_len=200,
            max_len=1200,
            seed=CFG["seed"],
        )
        scores_no_const = eval_model(out_dir_no_constraints, eval_raw, text_key=text_key, ref_key=ref_key)
        ablation2_configs.append(("no_constraints", out_dir_no_constraints))
        print(f"  ✅ No constraints ROUGE-L: {scores_no_const['rougeL']:.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # No reward clamping
    print("\nTraining WITHOUT reward clamping...")
    out_dir_no_clamp = os.path.join(CFG["results_dir"], "scst_no_clamp")
    try:
        run_scst(
            model_name=CFG["sft_model_path"],
            out_dir=out_dir_no_clamp,
            n_train=CFG["n_train"],
            batch_size=8,
            epochs=1,
            lr=1e-6,
            warmup_ratio=0.1,
            train_dataset=train_raw,
            text_key=text_key,
            ref_key=ref_key,
            # No clamping: allow full reward range
            no_repeat_ngram_size=4,
            top_p=0.75,
            reward_clip=10.0,  # Effectively no clamp
            min_len=200,
            max_len=1200,
            seed=CFG["seed"],
        )
        scores_no_clamp = eval_model(out_dir_no_clamp, eval_raw, text_key=text_key, ref_key=ref_key)
        ablation2_configs.append(("no_clamp", out_dir_no_clamp))
        print(f"  ✅ No clamp ROUGE-L: {scores_no_clamp['rougeL']:.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # No data filtering
    print("\nTraining WITHOUT data filtering...")
    out_dir_no_filter = os.path.join(CFG["results_dir"], "scst_no_filter")
    try:
        run_scst(
            model_name=CFG["sft_model_path"],
            out_dir=out_dir_no_filter,
            n_train=CFG["n_train"],
            batch_size=8,
            epochs=1,
            lr=1e-6,
            warmup_ratio=0.1,
            train_dataset=train_raw,
            text_key=text_key,
            ref_key=ref_key,
            # No filtering: accept all lengths
            no_repeat_ngram_size=4,
            top_p=0.75,
            reward_clip=0.5,
            min_len=0,  # No minimum
            max_len=10000,  # No maximum
            seed=CFG["seed"],
        )
        scores_no_filter = eval_model(out_dir_no_filter, eval_raw, text_key=text_key, ref_key=ref_key)
        ablation2_configs.append(("no_filter", out_dir_no_filter))
        print(f"  ✅ No filter ROUGE-L: {scores_no_filter['rougeL']:.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# ============================================================================
# CELL 6: Ablation 3: Hyperparameters
# ============================================================================

print("\n" + "="*70)
print("ABLATION 3: Hyperparameter Sensitivity")
print("="*70)
print("Testing different learning rates and batch sizes")

if not os.path.exists(CFG["sft_model_path"]):
    print("⚠️  SFT model required for this ablation, skipping")
else:
    ablation3_configs = []
    
    # Different learning rates
    for lr in [5e-7, 1e-6, 2e-6]:
        print(f"\nTraining with LR={lr}...")
        out_dir_lr = os.path.join(CFG["results_dir"], f"scst_lr_{lr}")
        try:
            run_scst(
                model_name=CFG["sft_model_path"],
                out_dir=out_dir_lr,
                n_train=CFG["n_train"],
                batch_size=8,
                epochs=1,
                lr=lr,
                warmup_ratio=0.1,
                train_dataset=train_raw,
                text_key=text_key,
                ref_key=ref_key,
                seed=CFG["seed"],
            )
            scores_lr = eval_model(out_dir_lr, eval_raw, text_key=text_key, ref_key=ref_key)
            ablation3_configs.append((f"lr_{lr}", out_dir_lr))
            print(f"  ✅ LR={lr} ROUGE-L: {scores_lr['rougeL']:.4f}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

# ============================================================================
# CELL 7: Compile Ablation Results
# ============================================================================

print("\n" + "="*70)
print("Compiling ablation results...")
print("="*70)

# Collect all model paths
all_models = {}

# Baselines
all_models["base"] = CFG["base_model"]
if os.path.exists(CFG["sft_model_path"]):
    all_models["sft"] = CFG["sft_model_path"]

# Ablation 1
for name, path in ablation1_configs:
    if os.path.exists(path):
        all_models[name] = path

# Ablation 2
if 'ablation2_configs' in locals():
    for name, path in ablation2_configs:
        if os.path.exists(path):
            all_models[name] = path

# Ablation 3
if 'ablation3_configs' in locals():
    for name, path in ablation3_configs:
        if os.path.exists(path):
            all_models[name] = path

# Run ablation study
print(f"\nEvaluating {len(all_models)} configurations...")
ablation_df = ablation_study(
    model_paths=all_models,
    eval_dataset=eval_raw,
    text_key=text_key,
    ref_key=ref_key,
    save_path=os.path.join(CFG["results_dir"], "ablation_results.csv"),
)

print("\n" + "="*70)
print("ABLATION RESULTS SUMMARY")
print("="*70)
print(ablation_df.round(4))

# ============================================================================
# CELL 8: Analysis and Interpretation
# ============================================================================

print("\n" + "="*70)
print("ABLATION ANALYSIS")
print("="*70)

if "scst_from_base" in ablation_df["config"].values and "scst_from_sft" in ablation_df["config"].values:
    base_row = ablation_df[ablation_df["config"] == "scst_from_base"].iloc[0]
    sft_row = ablation_df[ablation_df["config"] == "scst_from_sft"].iloc[0]
    warm_start_gain = sft_row["rougeL"] - base_row["rougeL"]
    print(f"\n1. Warm-Starting Effect:")
    print(f"   SCST from base:  {base_row['rougeL']:.4f}")
    print(f"   SCST from SFT:   {sft_row['rougeL']:.4f}")
    print(f"   Gain:            {warm_start_gain:+.4f} ({warm_start_gain/base_row['rougeL']*100:+.1f}% relative)")

if "full_stability" in ablation_df["config"].values:
    full_row = ablation_df[ablation_df["config"] == "full_stability"].iloc[0]
    print(f"\n2. Full Stability Configuration:")
    print(f"   ROUGE-L: {full_row['rougeL']:.4f}")
    print(f"   Compression: {full_row['compression']:.4f}")
    print(f"   Repetition: {full_row['repetition']:.4f}")
    
    if "no_constraints" in ablation_df["config"].values:
        no_const_row = ablation_df[ablation_df["config"] == "no_constraints"].iloc[0]
        constraint_effect = full_row["rougeL"] - no_const_row["rougeL"]
        print(f"\n   vs. No Constraints:")
        print(f"   ROUGE-L: {no_const_row['rougeL']:.4f}")
        print(f"   Effect: {constraint_effect:+.4f}")

print("\n✅ Ablation study complete!")
print(f"   Results saved to: {os.path.join(CFG['results_dir'], 'ablation_results.csv')}")

