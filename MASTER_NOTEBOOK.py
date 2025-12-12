# MASTER_NOTEBOOK.py
"""
Complete end-to-end pipeline for RL fine-tuning experiments.
Run this notebook from start to finish to generate all results for your paper.

This notebook:
1. Sets up environment and loads data
2. Trains SFT baseline
3. Trains SCST models (from base and SFT)
4. Evaluates all models
5. Runs comprehensive analysis
6. Generates all tables and figures for paper
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Force reload modules (important for Jupyter/Colab)
import importlib
if 'src' in sys.modules:
    importlib.reload(sys.modules['src'])
if 'src.eval_utils' in sys.modules:
    importlib.reload(sys.modules['src.eval_utils'])
if 'src.sft' in sys.modules:
    importlib.reload(sys.modules['src.sft'])
if 'src.scst' in sys.modules:
    importlib.reload(sys.modules['src.scst'])
if 'src.analysis' in sys.modules:
    importlib.reload(sys.modules['src.analysis'])

import torch
import numpy as np
import pandas as pd
import json
import random
from datetime import datetime
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed

# Import our modules
try:
    from src.sft import run_sft
    from src.scst import run_scst
    from src.eval_utils import eval_model, get_dataset_keys, generate_summaries_batched, metrics_table
    from src.data import preprocess_cnn_dm, preprocess_xsum  # For tokenizing datasets
    from src.analysis import (
        analyze_errors,
        compare_examples,
        plot_training_dynamics,
        analyze_reward_distribution,
        ablation_study,
        statistical_significance_test,
        length_distribution_analysis,
        comprehensive_model_analysis,
    )
    pass
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    raise

print("="*80)
print("RL FINE-TUNING MASTER NOTEBOOK")
print("="*80)

# ============================================================================
# CELL 1: Configuration and Setup
# ============================================================================

# Global configuration
CFG = {
    # Dataset
    "dataset": "cnn_dailymail",
    "n_train": 2000,      # Training examples for RL
    "n_eval": 1000,       # Evaluation examples
    "n_sft_train": 5000,  # Training examples for SFT (can be larger)
    
    # Models
    "base_model": "google/flan-t5-small",
    
    # Directories
    "results_dir": "./results",
    "checkpoints_dir": "./checkpoints",
    "analysis_dir": "./analysis",
    "logs_dir": "./logs",
    
    # Training
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # SFT hyperparameters
    "sft_batch_size": 8,
    "sft_epochs": 3,
    "sft_lr": 5e-5,
    "sft_warmup_ratio": 0.1,
    
    # SCST hyperparameters
    "scst_batch_size": 8,
    "scst_epochs": 1,
    "scst_lr": 1e-6,
    "scst_warmup_ratio": 0.1,
    
    # Generation
    "max_input_len": 512,
    "max_new_tokens": 64,
    "num_beams": 4,
    
    # Data filtering
    "min_len": 200,
    "max_len": 1200,
    
    # Decoding constraints
    "no_repeat_ngram_size": 4,
    "top_k": 30,
    "top_p": 0.75,
    "temperature": 0.7,
    
    # Stability
    "reward_clip": 0.5,
    "advantage_normalize": True,
}

# Create directories
for dir_name in [CFG["results_dir"], CFG["checkpoints_dir"], CFG["analysis_dir"], CFG["logs_dir"]]:
    os.makedirs(dir_name, exist_ok=True)

# Set seeds for reproducibility
set_seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])

# Get dataset keys
text_key, ref_key = get_dataset_keys(CFG["dataset"])

# ============================================================================
# CELL 2: Load Datasets
# ============================================================================

if CFG["dataset"] == "cnn_dailymail":
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Training data for SFT
    train_raw_sft = dataset["train"].select(range(CFG["n_sft_train"]))
    
    # Training data for RL (subset, can overlap with SFT)
    train_raw_rl = dataset["train"].select(range(CFG["n_train"]))
    
    # Validation data for SFT
    val_raw_sft = dataset["validation"].select(range(min(500, CFG["n_eval"])))
    
    # Evaluation data (fixed set for all models)
    eval_raw = dataset["validation"].select(range(CFG["n_eval"]))
else:
    raise ValueError(f"Unknown dataset: {CFG['dataset']}")

# ============================================================================
# CELL 3: Train SFT Baseline
# ============================================================================

print("\n" + "="*80)
print("TRAINING SFT BASELINE")
print("="*80)

sft_output_dir = os.path.join(CFG["checkpoints_dir"], "flan_t5_sft")
# run_sft saves directly to output_dir, not a subdirectory
sft_best_dir = sft_output_dir

# Check if SFT already exists
if os.path.exists(sft_best_dir) and os.path.exists(os.path.join(sft_best_dir, "config.json")):
    print(f"WARNING: SFT model exists, skipping training")
    SKIP_SFT = True
else:
    SKIP_SFT = False

if not SKIP_SFT:
    print("Training SFT model (this may take 30-60 minutes)...")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(CFG["base_model"])
        
        # Tokenize datasets for SFT training
        if CFG["dataset"] == "cnn_dailymail":
            train_tok_sft, val_tok_sft = preprocess_cnn_dm(
                train_raw_sft,
                val_raw_sft,
                tokenizer=tokenizer,
                max_input_len=CFG["max_input_len"],
                max_target_len=CFG["max_new_tokens"] + 10,  # Some buffer
            )
        elif CFG["dataset"] == "xsum":
            train_tok_sft, val_tok_sft = preprocess_xsum(
                train_raw_sft,
                val_raw_sft,
                tokenizer=tokenizer,
                max_input_len=CFG["max_input_len"],
                max_target_len=CFG["max_new_tokens"] + 10,
            )
        else:
            raise ValueError(f"Unknown dataset: {CFG['dataset']}")
        
        sft_model, sft_history = run_sft(
            train_dataset=train_tok_sft,
            val_dataset=val_tok_sft,
            tokenizer=tokenizer,
            model_name=CFG["base_model"],
            batch_size=CFG["sft_batch_size"],
            num_epochs=CFG["sft_epochs"],
            lr=CFG["sft_lr"],
            warmup_ratio=CFG["sft_warmup_ratio"],
            output_dir=sft_output_dir,
            device=CFG["device"],
            # Evaluation during training (use raw dataset for generation)
            eval_dataset=eval_raw,
            text_key=text_key,
            ref_key=ref_key,
            eval_batch_size=16,
            eval_max_input_len=CFG["max_input_len"],
            eval_max_new_tokens=CFG["max_new_tokens"],
            eval_num_beams=CFG["num_beams"],
            print_metrics=True,
        )
        
        # Save training history
        history_path = os.path.join(CFG["logs_dir"], "sft_history.json")
        with open(history_path, 'w') as f:
            json.dump(sft_history, f, indent=2)
        
        print("SFT training complete!")
        
    except Exception as e:
        print(f"ERROR: Error during SFT training: {e}")
        import traceback
        traceback.print_exc()
        raise

# Evaluate SFT
print("Evaluating SFT...")
sft_scores = eval_model(
    sft_best_dir,
    eval_raw,
    text_key=text_key,
    ref_key=ref_key,
    save_path=os.path.join(CFG["results_dir"], "results_sft.json"),
)
print(f"SFT - ROUGE-L: {sft_scores['rougeL']:.4f}, ROUGE-1: {sft_scores['rouge1']:.4f}, BLEU: {sft_scores['bleu']:.4f}")

# ============================================================================
# CELL 4: Train SCST Models
# ============================================================================

print("\n" + "="*80)
print("TRAINING SCST MODELS")
print("="*80)

scst_experiments = [
    {
        "name": "scst_from_sft",
        "model_path": sft_best_dir,
        "description": "SCST starting from SFT checkpoint",
    },
    {
        "name": "scst_from_base",
        "model_path": CFG["base_model"],
        "description": "SCST starting from base model",
    },
]

scst_results = {}

for exp in scst_experiments:
    print(f"\n{exp['name']} ({exp['description']})...")
    
    out_dir = os.path.join(CFG["checkpoints_dir"], exp["name"])
    
    # Check if already exists
    if os.path.exists(out_dir):
        print(f"  WARNING: Model exists, skipping")
        skip = True
    else:
        skip = False
    
    if not skip:
        print(f"  Training (may take 20-40 minutes)...")
        
        try:
            run_scst(
                model_name=exp["model_path"],
                out_dir=out_dir,
                n_train=CFG["n_train"],
                batch_size=CFG["scst_batch_size"],
                epochs=CFG["scst_epochs"],
                lr=CFG["scst_lr"],
                warmup_ratio=CFG["scst_warmup_ratio"],
                # Generation parameters
                greedy_beams=CFG["num_beams"],
                greedy_max_new=CFG["max_new_tokens"],
                sample_max_new=CFG["max_new_tokens"],
                sample_min_new=8,
                top_k=CFG["top_k"],
                top_p=CFG["top_p"],
                temperature=CFG["temperature"],
                no_repeat_ngram_size=CFG["no_repeat_ngram_size"],
                max_input_len=CFG["max_input_len"],
                max_target_len=CFG["max_new_tokens"] + 10,
                # Stability
                advantage_normalize=CFG["advantage_normalize"],
                reward_clip=CFG["reward_clip"],
                # Dataset
                dataset_name=CFG["dataset"],
                # Data filtering
                min_len=CFG["min_len"],
                max_len=CFG["max_len"],
                seed=CFG["seed"],
            )
            
            print(f"  Training complete")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Evaluate
    try:
        scores = eval_model(
            out_dir,
            eval_raw,
            text_key=text_key,
            ref_key=ref_key,
            save_path=os.path.join(CFG["results_dir"], f"results_{exp['name']}.json"),
        )
        scst_results[exp["name"]] = scores
        print(f"  ROUGE-L: {scores['rougeL']:.4f}, ROUGE-1: {scores['rouge1']:.4f}, BLEU: {scores['bleu']:.4f}")
        
    except Exception as e:
        print(f"  ERROR: Evaluation error: {e}")

# ============================================================================
# CELL 5: Evaluate Base Model (for comparison)
# ============================================================================

print("\nEvaluating base model...")
base_scores = eval_model(
    CFG["base_model"],
    eval_raw,
    text_key=text_key,
    ref_key=ref_key,
    save_path=os.path.join(CFG["results_dir"], "results_base.json"),
)
print(f"Base - ROUGE-L: {base_scores['rougeL']:.4f}, ROUGE-1: {base_scores['rouge1']:.4f}, BLEU: {base_scores['bleu']:.4f}")

# ============================================================================
# CELL 6: Generate Predictions for All Models
# ============================================================================

print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

all_models = {
    "base": CFG["base_model"],
    "sft": sft_best_dir,
}

# Add SCST models
for exp in scst_experiments:
    out_dir = os.path.join(CFG["checkpoints_dir"], exp["name"])
    if os.path.exists(out_dir):
        all_models[exp["name"]] = out_dir

all_preds = {}
all_refs = None

for model_name, model_path in all_models.items():
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        preds, refs = generate_summaries_batched(
            model, tokenizer, eval_raw,
            text_key=text_key,
            ref_key=ref_key,
            batch_size=16,
            max_input_len=CFG["max_input_len"],
            max_new_tokens=CFG["max_new_tokens"],
            num_beams=CFG["num_beams"],
        )
        
        all_preds[model_name] = preds
        if all_refs is None:
            all_refs = refs
        
    except Exception as e:
        print(f"ERROR: Error generating {model_name}: {e}")

print(f"Generated predictions for {len(all_preds)} models")

# ============================================================================
# CELL 7: Comprehensive Analysis
# ============================================================================

print("\n" + "="*80)
print("RUNNING ANALYSIS")
print("="*80)

comparison_data = []
for model_name in all_models.keys():
    if model_name in all_preds:
        metrics = metrics_table(all_preds[model_name], all_refs)
        metrics["model"] = model_name
        comparison_data.append(metrics)

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df[["model"] + [c for c in comparison_df.columns if c != "model"]]

comparison_path = os.path.join(CFG["analysis_dir"], "model_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)

# Error Analysis

error_comparison = {}
articles = [ex[text_key] for ex in eval_raw] if hasattr(eval_raw[0], text_key) else None

for model_name, preds in all_preds.items():
    error_analysis = analyze_errors(
        preds, all_refs, articles,
        save_path=os.path.join(CFG["analysis_dir"], f"error_analysis_{model_name}.json")
    )
    error_comparison[model_name] = error_analysis

# Create error comparison table
error_df_data = []
for model_name, analysis in error_comparison.items():
    row = {"model": model_name}
    row.update(analysis["error_percentages"])
    row["avg_compression"] = analysis["avg_compression"]
    row["avg_repetition"] = analysis["avg_repetition"]
    row["avg_rouge_l"] = analysis["avg_rouge_l"]
    error_df_data.append(row)

error_df = pd.DataFrame(error_df_data)
error_path = os.path.join(CFG["analysis_dir"], "error_analysis_comparison.csv")
error_df.to_csv(error_path, index=False)

# Example Comparisons

examples = compare_examples(
    model_preds=all_preds,
    refs=all_refs,
    articles=articles,
    n_examples=10,
    save_path=os.path.join(CFG["analysis_dir"], "example_comparisons.json"),
)

# Length Distribution Analysis

length_comparison = {}
for model_name, preds in all_preds.items():
    length_analysis = length_distribution_analysis(
        preds, all_refs,
        save_path=os.path.join(CFG["analysis_dir"], f"length_analysis_{model_name}.json")
    )
    length_comparison[model_name] = length_analysis

length_df_data = []
for model_name, analysis in length_comparison.items():
    row = {"model": model_name}
    row["pred_mean"] = analysis["pred_mean"]
    row["pred_std"] = analysis["pred_std"]
    row["ref_mean"] = analysis["ref_mean"]
    row["compression_mean"] = analysis["compression_mean"]
    row["compression_std"] = analysis["compression_std"]
    length_df_data.append(row)

length_df = pd.DataFrame(length_df_data)
length_path = os.path.join(CFG["analysis_dir"], "length_analysis.csv")
length_df.to_csv(length_path, index=False)

# Statistical Significance Testing

try:
    import evaluate
    rouge = evaluate.load("rouge")
    
    # Compare SFT vs SCST from SFT
    if "sft" in all_preds and "scst_from_sft" in all_preds:
        sft_scores_list = rouge.compute(
            predictions=all_preds["sft"],
            references=all_refs,
            use_aggregator=False
        )["rougeL"]
        scst_scores_list = rouge.compute(
            predictions=all_preds["scst_from_sft"],
            references=all_refs,
            use_aggregator=False
        )["rougeL"]
        
        sig_test = statistical_significance_test(
            sft_scores_list, scst_scores_list, test_type="paired_t"
        )
        
        sig_path = os.path.join(CFG["analysis_dir"], "significance_test.json")
        with open(sig_path, 'w') as f:
            json.dump(sig_test, f, indent=2)
        
        print(f"Statistical test (SFT vs SCST): p={sig_test['p_value']:.6f}, significant={sig_test['significant']}")
    
    # Compare Base vs SCST from Base
    if "base" in all_preds and "scst_from_base" in all_preds:
        base_scores_list = rouge.compute(
            predictions=all_preds["base"],
            references=all_refs,
            use_aggregator=False
        )["rougeL"]
        scst_base_scores_list = rouge.compute(
            predictions=all_preds["scst_from_base"],
            references=all_refs,
            use_aggregator=False
        )["rougeL"]
        
        sig_test_base = statistical_significance_test(
            base_scores_list, scst_base_scores_list, test_type="paired_t"
        )
        
        print(f"Statistical test (Base vs SCST): p={sig_test_base['p_value']:.6f}, significant={sig_test_base['significant']}")

except Exception as e:
    pass

# Comprehensive Analysis for Each Model
for model_name, model_path in all_models.items():
    if model_name in all_preds:
        try:
            output_dir = os.path.join(CFG["analysis_dir"], model_name)
            comprehensive_model_analysis(
                model_path=model_path,
                eval_dataset=eval_raw,
                text_key=text_key,
                ref_key=ref_key,
                output_dir=output_dir,
                n_examples=10,
            )
        except Exception as e:
            pass

# ============================================================================
# CELL 8: Final Summary and Results
# ============================================================================

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print("\nModel Comparison:")
print(comparison_df.round(4).to_string(index=False))

print("\nKey Findings:")
if "sft" in comparison_df["model"].values and "scst_from_sft" in comparison_df["model"].values:
    sft_row = comparison_df[comparison_df["model"] == "sft"].iloc[0]
    scst_row = comparison_df[comparison_df["model"] == "scst_from_sft"].iloc[0]
    improvement = scst_row["rougeL"] - sft_row["rougeL"]
    rel_improvement = (improvement / sft_row["rougeL"]) * 100
    print(f"  • SCST from SFT improves ROUGE-L by {improvement:+.4f} ({rel_improvement:+.2f}% relative)")
    print(f"    SFT: {sft_row['rougeL']:.4f} → SCST: {scst_row['rougeL']:.4f}")

if "base" in comparison_df["model"].values and "scst_from_base" in comparison_df["model"].values:
    base_row = comparison_df[comparison_df["model"] == "base"].iloc[0]
    scst_base_row = comparison_df[comparison_df["model"] == "scst_from_base"].iloc[0]
    improvement_base = scst_base_row["rougeL"] - base_row["rougeL"]
    print(f"  • SCST from Base improves ROUGE-L by {improvement_base:+.4f}")
    print(f"    Base: {base_row['rougeL']:.4f} → SCST: {scst_base_row['rougeL']:.4f}")

if "scst_from_sft" in comparison_df["model"].values and "scst_from_base" in comparison_df["model"].values:
    scst_sft_row = comparison_df[comparison_df["model"] == "scst_from_sft"].iloc[0]
    scst_base_row = comparison_df[comparison_df["model"] == "scst_from_base"].iloc[0]
    warm_start_gain = scst_sft_row["rougeL"] - scst_base_row["rougeL"]
    rel_gain = (warm_start_gain / scst_base_row["rougeL"]) * 100
    print(f"  • Warm-starting from SFT provides {warm_start_gain:+.4f} ROUGE-L gain ({rel_gain:+.1f}% relative)")
    print(f"    SCST from Base: {scst_base_row['rougeL']:.4f} → SCST from SFT: {scst_sft_row['rougeL']:.4f}")

print("\nGenerated Files:")
print(f"  Results directory: {CFG['results_dir']}/")
print(f"    - results_base.json")
print(f"    - results_sft.json")
for exp in scst_experiments:
    print(f"    - results_{exp['name']}.json")

print(f"\n  Analysis directory: {CFG['analysis_dir']}/")
print(f"    - model_comparison.csv (main results table)")
print(f"    - error_analysis_comparison.csv (error breakdown)")
print(f"    - length_analysis.csv (length statistics)")
print(f"    - example_comparisons.json (qualitative examples)")
print(f"    - significance_test.json (statistical tests)")
print(f"    - Per-model directories with detailed analysis")

print(f"\n  Checkpoints directory: {CFG['checkpoints_dir']}/")
print(f"    - flan_t5_sft/ (SFT model)")
for exp in scst_experiments:
    print(f"    - {exp['name']}/ (SCST model)")

print("\nAll experiments and analysis complete!")

# ============================================================================
# CELL 9: Generate Visualizations
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 9.1: Model Comparison Bar Chart
print("Creating model comparison plots...")

if len(comparison_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROUGE scores
    models = comparison_df["model"].values
    rouge1 = comparison_df["rouge1"].values
    rouge2 = comparison_df["rouge2"].values
    rougeL = comparison_df["rougeL"].values
    bleu = comparison_df["bleu"].values
    
    x_pos = range(len(models))
    
    # ROUGE-1
    axes[0, 0].bar(x_pos, rouge1, alpha=0.7, color='steelblue')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].set_ylabel('ROUGE-1')
    axes[0, 0].set_title('ROUGE-1 Scores')
    axes[0, 0].grid(True, alpha=0.3)
    for i, v in enumerate(rouge1):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ROUGE-2
    axes[0, 1].bar(x_pos, rouge2, alpha=0.7, color='coral')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].set_ylabel('ROUGE-2')
    axes[0, 1].set_title('ROUGE-2 Scores')
    axes[0, 1].grid(True, alpha=0.3)
    for i, v in enumerate(rouge2):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ROUGE-L
    axes[1, 0].bar(x_pos, rougeL, alpha=0.7, color='mediumseagreen')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].set_ylabel('ROUGE-L')
    axes[1, 0].set_title('ROUGE-L Scores')
    axes[1, 0].grid(True, alpha=0.3)
    for i, v in enumerate(rougeL):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # BLEU
    axes[1, 1].bar(x_pos, bleu, alpha=0.7, color='gold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].set_ylabel('BLEU')
    axes[1, 1].set_title('BLEU Scores')
    axes[1, 1].grid(True, alpha=0.3)
    for i, v in enumerate(bleu):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(CFG["analysis_dir"], "model_comparison_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# 9.2: Compression and Repetition Comparison
if len(comparison_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    compression = comparison_df["compression"].values
    repetition = comparison_df["repetition"].values
    
    # Compression
    axes[0].bar(x_pos, compression, alpha=0.7, color='teal')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('Compression Ratio')
    axes[0].set_title('Compression Ratio (Article Length / Summary Length)')
    axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Equal Length')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for i, v in enumerate(compression):
        axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Repetition
    axes[1].bar(x_pos, repetition, alpha=0.7, color='indianred')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylabel('Repetition Rate')
    axes[1].set_title('Repetition Rate (1 - unique_words/total_words)')
    axes[1].grid(True, alpha=0.3)
    for i, v in enumerate(repetition):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(CFG["analysis_dir"], "compression_repetition.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# 9.3: Error Analysis Visualization
if len(error_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    error_models = error_df["model"].values
    
    # Too short
    if "too_short" in error_df.columns:
        axes[0, 0].bar(range(len(error_models)), error_df["too_short"].values, alpha=0.7, color='lightblue')
        axes[0, 0].set_xticks(range(len(error_models)))
        axes[0, 0].set_xticklabels(error_models, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].set_title('Too Short Errors (%)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Too long
    if "too_long" in error_df.columns:
        axes[0, 1].bar(range(len(error_models)), error_df["too_long"].values, alpha=0.7, color='lightcoral')
        axes[0, 1].set_xticks(range(len(error_models)))
        axes[0, 1].set_xticklabels(error_models, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].set_title('Too Long Errors (%)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # High repetition
    if "high_repetition" in error_df.columns:
        axes[1, 0].bar(range(len(error_models)), error_df["high_repetition"].values, alpha=0.7, color='plum')
        axes[1, 0].set_xticks(range(len(error_models)))
        axes[1, 0].set_xticklabels(error_models, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Percentage')
        axes[1, 0].set_title('High Repetition Errors (%)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Low coverage
    if "low_coverage" in error_df.columns:
        axes[1, 1].bar(range(len(error_models)), error_df["low_coverage"].values, alpha=0.7, color='khaki')
        axes[1, 1].set_xticks(range(len(error_models)))
        axes[1, 1].set_xticklabels(error_models, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].set_title('Low Coverage Errors (%)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(CFG["analysis_dir"], "error_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# 9.4: Length Distribution Comparison
if len(length_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    length_models = length_df["model"].values
    
    # Mean lengths
    pred_means = length_df["pred_mean"].values
    ref_mean = length_df["ref_mean"].iloc[0] if len(length_df) > 0 else 0
    
    axes[0].bar(range(len(length_models)), pred_means, alpha=0.7, color='steelblue', label='Predictions')
    axes[0].axhline(y=ref_mean, color='r', linestyle='--', alpha=0.7, label='Reference')
    axes[0].set_xticks(range(len(length_models)))
    axes[0].set_xticklabels(length_models, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Length (words)')
    axes[0].set_title('Mean Summary Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Compression ratios
    comp_means = length_df["compression_mean"].values
    axes[1].bar(range(len(length_models)), comp_means, alpha=0.7, color='teal')
    axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Equal Length')
    axes[1].set_xticks(range(len(length_models)))
    axes[1].set_xticklabels(length_models, rotation=45, ha='right')
    axes[1].set_ylabel('Compression Ratio')
    axes[1].set_title('Compression Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(CFG["analysis_dir"], "length_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# 9.5: SFT Training History (if available)
history_path = os.path.join(CFG["logs_dir"], "sft_history.json")
if os.path.exists(history_path):
    print("Creating SFT training curves...")
    try:
        with open(history_path, 'r') as f:
            sft_history = json.load(f)
        
        if "train_loss" in sft_history and "val_loss" in sft_history:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            epochs = range(1, len(sft_history["train_loss"]) + 1)
            ax.plot(epochs, sft_history["train_loss"], 'b-o', label='Train Loss', linewidth=2, markersize=6)
            ax.plot(epochs, sft_history["val_loss"], 'r-s', label='Val Loss', linewidth=2, markersize=6)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('SFT Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(CFG["analysis_dir"], "sft_training_curves.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
            plt.close()
        
        # Plot metrics over epochs if available
        if "metrics" in sft_history and len(sft_history["metrics"]) > 0:
            metrics_list = [m for m in sft_history["metrics"] if m is not None]
            if len(metrics_list) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                epochs_metrics = range(1, len(metrics_list) + 1)
                
                rouge1_vals = [m.get("rouge1", 0) for m in metrics_list]
                rouge2_vals = [m.get("rouge2", 0) for m in metrics_list]
                rougeL_vals = [m.get("rougeL", 0) for m in metrics_list]
                bleu_vals = [m.get("bleu", 0) for m in metrics_list]
                
                axes[0, 0].plot(epochs_metrics, rouge1_vals, 'b-o', linewidth=2, markersize=6)
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('ROUGE-1')
                axes[0, 0].set_title('ROUGE-1 Over Training')
                axes[0, 0].grid(True, alpha=0.3)
                
                axes[0, 1].plot(epochs_metrics, rouge2_vals, 'g-o', linewidth=2, markersize=6)
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('ROUGE-2')
                axes[0, 1].set_title('ROUGE-2 Over Training')
                axes[0, 1].grid(True, alpha=0.3)
                
                axes[1, 0].plot(epochs_metrics, rougeL_vals, 'r-o', linewidth=2, markersize=6)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('ROUGE-L')
                axes[1, 0].set_title('ROUGE-L Over Training')
                axes[1, 0].grid(True, alpha=0.3)
                
                axes[1, 1].plot(epochs_metrics, bleu_vals, 'm-o', linewidth=2, markersize=6)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('BLEU')
                axes[1, 1].set_title('BLEU Over Training')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(CFG["analysis_dir"], "sft_metrics_over_time.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"  Saved: {plot_path}")
                plt.close()
    except Exception as e:
        print(f"  WARNING: Could not plot SFT history: {e}")

# 9.6: Combined ROUGE Comparison (all models side by side)
if len(comparison_df) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, rouge1, width, label='ROUGE-1', alpha=0.8, color='steelblue')
    ax.bar(x, rouge2, width, label='ROUGE-2', alpha=0.8, color='coral')
    ax.bar(x + width, rougeL, width, label='ROUGE-L', alpha=0.8, color='mediumseagreen')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('ROUGE Scores Comparison Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(CFG["analysis_dir"], "rouge_comparison_combined.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# 9.7: Detailed Performance Comparison (Heatmap)
if len(comparison_df) > 0:
    print("Creating performance heatmap...")
    
    # Prepare data for heatmap
    metrics_to_plot = ['rouge1', 'rouge2', 'rougeL', 'bleu']
    heatmap_data = comparison_df.set_index('model')[metrics_to_plot].T
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score'}, linewidths=0.5, ax=ax)
    ax.set_title('Performance Metrics Heatmap Across Models')
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric')
    
    plt.tight_layout()
    plot_path = os.path.join(CFG["analysis_dir"], "performance_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# 9.8: Improvement Over Baseline
if len(comparison_df) > 0 and "base" in comparison_df["model"].values:
    print("Creating improvement over baseline plot...")
    
    base_row = comparison_df[comparison_df["model"] == "base"].iloc[0]
    other_models = comparison_df[comparison_df["model"] != "base"]
    
    if len(other_models) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu']
        improvements = {}
        for metric in metrics:
            improvements[metric] = []
            for _, row in other_models.iterrows():
                imp = ((row[metric] - base_row[metric]) / base_row[metric]) * 100
                improvements[metric].append(imp)
        
        x = np.arange(len(other_models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[0].bar(x + i*width, improvements[metric], width, 
                       label=metric.upper(), alpha=0.8)
        
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Relative Improvement (%)')
        axes[0].set_title('Relative Improvement Over Base Model')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(other_models["model"].values, rotation=45, ha='right')
        axes[0].legend()
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Absolute improvements
        abs_improvements = {}
        for metric in metrics:
            abs_improvements[metric] = []
            for _, row in other_models.iterrows():
                imp = row[metric] - base_row[metric]
                abs_improvements[metric].append(imp)
        
        for i, metric in enumerate(metrics):
            axes[1].bar(x + i*width, abs_improvements[metric], width,
                       label=metric.upper(), alpha=0.8)
        
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Absolute Improvement')
        axes[1].set_title('Absolute Improvement Over Base Model')
        axes[1].set_xticks(x + width * 1.5)
        axes[1].set_xticklabels(other_models["model"].values, rotation=45, ha='right')
        axes[1].legend()
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(CFG["analysis_dir"], "improvement_over_baseline.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()

# 9.9: Summary Statistics Table Visualization
if len(comparison_df) > 0:
    print("Creating summary statistics visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = comparison_df[['model', 'rouge1', 'rouge2', 'rougeL', 'bleu', 
                                 'compression', 'repetition']].copy()
    table_data = table_data.round(4)
    
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(table_data.columns)):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Model Performance Summary Table', pad=20, fontsize=14, weight='bold')
    
    plot_path = os.path.join(CFG["analysis_dir"], "summary_table.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# 9.10: Error Type Distribution (Stacked Bar Chart)
if len(error_df) > 0:
    print("Creating error distribution plot...")
    
    error_types = ['too_short', 'too_long', 'high_repetition', 'low_coverage']
    available_types = [et for et in error_types if et in error_df.columns]
    
    if len(available_types) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = error_df["model"].values
        x = np.arange(len(models))
        width = 0.6
        
        bottom = np.zeros(len(models))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for i, error_type in enumerate(available_types):
            values = error_df[error_type].values
            ax.bar(x, values, width, label=error_type.replace('_', ' ').title(),
                   bottom=bottom, color=colors[i % len(colors)], alpha=0.8)
            bottom += values
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Error Percentage (%)')
        ax.set_title('Error Type Distribution Across Models')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(CFG["analysis_dir"], "error_distribution_stacked.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()

# 9.11: ROUGE Score Radar Chart
if len(comparison_df) > 0:
    print("Creating radar chart...")
    
    try:
        from math import pi
        
        models = comparison_df["model"].values
        metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu']
        
        # Number of variables
        N = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for idx, model in enumerate(models):
            values = comparison_df[comparison_df["model"] == model][metrics].iloc[0].values.tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=14, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(CFG["analysis_dir"], "performance_radar.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"  WARNING: Could not create radar chart: {e}")

# 9.12: Length vs Quality Scatter Plot
if len(comparison_df) > 0:
    print("Creating length vs quality analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = comparison_df["model"].values
    compression = comparison_df["compression"].values
    rougeL = comparison_df["rougeL"].values
    repetition = comparison_df["repetition"].values
    
    # Compression vs ROUGE-L
    scatter1 = axes[0].scatter(compression, rougeL, s=200, alpha=0.6, c=range(len(models)), 
                               cmap='viridis', edgecolors='black', linewidth=2)
    for i, model in enumerate(models):
        axes[0].annotate(model, (compression[i], rougeL[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0].set_xlabel('Compression Ratio')
    axes[0].set_ylabel('ROUGE-L Score')
    axes[0].set_title('Compression vs Quality (ROUGE-L)')
    axes[0].grid(True, alpha=0.3)
    
    # Repetition vs ROUGE-L
    scatter2 = axes[1].scatter(repetition, rougeL, s=200, alpha=0.6, c=range(len(models)),
                               cmap='plasma', edgecolors='black', linewidth=2)
    for i, model in enumerate(models):
        axes[1].annotate(model, (repetition[i], rougeL[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1].set_xlabel('Repetition Rate')
    axes[1].set_ylabel('ROUGE-L Score')
    axes[1].set_title('Repetition vs Quality (ROUGE-L)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(CFG["analysis_dir"], "length_quality_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# 9.13: Example Prediction Quality Distribution
if len(all_preds) > 0 and all_refs is not None:
    print("Creating prediction quality distribution...")
    
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx, (model_name, preds) in enumerate(all_preds.items()):
            if idx >= 4:  # Limit to 4 models
                break
            
            row = idx // 2
            col = idx % 2
            
            scores = rouge.compute(
                predictions=preds,
                references=all_refs,
                use_aggregator=False
            )["rougeL"]
            
            scores = [float(s) for s in scores]
            
            axes[row, col].hist(scores, bins=30, alpha=0.7, edgecolor='black')
            axes[row, col].axvline(np.mean(scores), color='r', linestyle='--', 
                                   linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
            axes[row, col].set_xlabel('ROUGE-L Score')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].set_title(f'{model_name} - ROUGE-L Distribution')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(all_preds), 4):
            row = idx // 2
            col = idx % 2
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(CFG["analysis_dir"], "quality_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"  WARNING: Could not create quality distribution: {e}")

print("\nAll plots generated!")
print(f"\nResults saved to: {CFG['analysis_dir']}/")
print("CSV/JSON files:")
print("  - model_comparison.csv (main results)")
print("  - error_analysis_comparison.csv")
print("  - example_comparisons.json")
print("\nVisualization files:")
print("  - model_comparison_metrics.png")
print("  - compression_repetition.png")
print("  - error_analysis.png")
print("  - length_distribution.png")
print("  - rouge_comparison_combined.png")
print("  - performance_heatmap.png")
print("  - improvement_over_baseline.png")
print("  - summary_table.png")
print("  - error_distribution_stacked.png")
print("  - performance_radar.png")
print("  - length_quality_analysis.png")
print("  - quality_distribution.png")
if os.path.exists(history_path):
    print("  - sft_training_curves.png")
    print("  - sft_metrics_over_time.png")
print("="*80)

