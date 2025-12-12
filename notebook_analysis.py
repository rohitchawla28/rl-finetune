# notebook_analysis.py
"""
Comprehensive analysis notebook for RL fine-tuning experiments.
Run this after training to generate all analysis for the paper.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

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
from src.eval_utils import eval_model, get_dataset_keys, generate_summaries_batched

# ============================================================================
# CELL 1: Setup and Configuration
# ============================================================================

CFG = {
    "dataset": "cnn_dailymail",
    "model_name": "google/flan-t5-small",
    "results_dir": "./results",
    "analysis_dir": "./analysis",
    "n_eval": 1000,  # Use 1000 examples for analysis
}

# Create analysis directory
os.makedirs(CFG["analysis_dir"], exist_ok=True)

print("✅ Configuration loaded!")
print(f"   Dataset: {CFG['dataset']}")
print(f"   Analysis directory: {CFG['analysis_dir']}")

# ============================================================================
# CELL 2: Load Evaluation Dataset
# ============================================================================

print("\n" + "="*70)
print("Loading evaluation dataset...")
print("="*70)

if CFG["dataset"] == "cnn_dailymail":
    eval_raw = load_dataset("cnn_dailymail", "3.0.0")["validation"].select(range(CFG["n_eval"]))
    text_key, ref_key = "article", "highlights"
else:
    raise ValueError(f"Unknown dataset: {CFG['dataset']}")

print(f"✅ Loaded {len(eval_raw)} evaluation examples")
print(f"   Text key: {text_key}, Reference key: {ref_key}")

# ============================================================================
# CELL 3: Load Model Checkpoints
# ============================================================================

print("\n" + "="*70)
print("Loading model checkpoints...")
print("="*70)

# Update these paths to your actual checkpoint directories
MODEL_PATHS = {
    "base": CFG["model_name"],
    "sft": "./checkpoints/flan_t5_sft/best",  # Update with your SFT path
    "scst_sft": "./checkpoints/scst_sft_rouge_l",  # Update with your SCST path
    "scst_base": "./checkpoints/scst_base_rouge_l",  # Update if you have this
}

# Verify which models exist
available_models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path) or path.startswith("google/"):
        available_models[name] = path
        print(f"✅ {name}: {path}")
    else:
        print(f"⚠️  {name}: {path} (not found, skipping)")

print(f"\n✅ Found {len(available_models)} available models")

# ============================================================================
# CELL 4: Comprehensive Analysis for Each Model
# ============================================================================

print("\n" + "="*70)
print("Running comprehensive analysis for each model...")
print("="*70)

all_results = {}

for model_name, model_path in available_models.items():
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*70}")
    
    output_dir = os.path.join(CFG["analysis_dir"], model_name)
    
    try:
        results = comprehensive_model_analysis(
            model_path=model_path,
            eval_dataset=eval_raw,
            text_key=text_key,
            ref_key=ref_key,
            output_dir=output_dir,
            n_examples=10,
        )
        all_results[model_name] = results
        print(f"✅ Analysis complete for {model_name}")
    except Exception as e:
        print(f"❌ Error analyzing {model_name}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# CELL 5: Compare Models Side-by-Side
# ============================================================================

print("\n" + "="*70)
print("Comparing all models...")
print("="*70)

# Generate predictions for all models
all_preds = {}
all_refs = []

for model_name, model_path in available_models.items():
    print(f"Generating predictions for {model_name}...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        preds, refs = generate_summaries_batched(
            model, tokenizer, eval_raw,
            text_key=text_key, ref_key=ref_key,
            batch_size=16,
        )
        all_preds[model_name] = preds
        if not all_refs:
            all_refs = refs
    except Exception as e:
        print(f"❌ Error generating predictions for {model_name}: {e}")

# Compare metrics
print("\n" + "="*70)
print("METRICS COMPARISON")
print("="*70)

comparison_data = []
for model_name, preds in all_preds.items():
    from src.eval_utils import metrics_table
    metrics = metrics_table(preds, all_refs)
    metrics["model"] = model_name
    comparison_data.append(metrics)

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df[["model"] + [c for c in comparison_df.columns if c != "model"]]
print(comparison_df.round(4))

# Save comparison
comparison_path = os.path.join(CFG["analysis_dir"], "model_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)
print(f"\n✅ Saved comparison to {comparison_path}")

# ============================================================================
# CELL 6: Error Analysis Comparison
# ============================================================================

print("\n" + "="*70)
print("Error Analysis Comparison")
print("="*70)

error_comparison = {}
for model_name, preds in all_preds.items():
    print(f"Analyzing errors for {model_name}...")
    articles = [ex[text_key] for ex in eval_raw] if hasattr(eval_raw[0], text_key) else None
    error_analysis = analyze_errors(preds, all_refs, articles)
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
print("\nError Analysis:")
print(error_df.round(2))

error_path = os.path.join(CFG["analysis_dir"], "error_analysis_comparison.csv")
error_df.to_csv(error_path, index=False)
print(f"\n✅ Saved error analysis to {error_path}")

# ============================================================================
# CELL 7: Example Comparisons
# ============================================================================

print("\n" + "="*70)
print("Generating example comparisons...")
print("="*70)

articles = [ex[text_key] for ex in eval_raw] if hasattr(eval_raw[0], text_key) else None

examples = compare_examples(
    model_preds=all_preds,
    refs=all_refs,
    articles=articles,
    n_examples=10,
)

examples_path = os.path.join(CFG["analysis_dir"], "example_comparisons.json")
import json
with open(examples_path, 'w') as f:
    json.dump(examples, f, indent=2)

print(f"✅ Saved {len(examples)} example comparisons to {examples_path}")

# Print a few examples
print("\n" + "="*70)
print("SAMPLE EXAMPLES (first 2)")
print("="*70)
for i, ex in enumerate(examples[:2]):
    print(f"\n--- Example {i+1} (Index {ex['index']}) ---")
    print(f"\nArticle (first 200 chars):")
    if "article" in ex:
        print(ex["article"][:200] + "...")
    print(f"\nReference:")
    print(ex["reference"])
    print(f"\nPredictions:")
    for model_name, pred in ex["predictions"].items():
        print(f"  {model_name}: {pred[:150]}...")

# ============================================================================
# CELL 8: Statistical Significance Testing
# ============================================================================

print("\n" + "="*70)
print("Statistical Significance Testing")
print("="*70)

if len(all_preds) >= 2:
    import evaluate
    rouge = evaluate.load("rouge")
    
    model_names = list(all_preds.keys())
    
    # Compare SFT vs SCST from SFT if available
    if "sft" in model_names and "scst_sft" in model_names:
        print("\nComparing SFT vs SCST from SFT...")
        sft_scores = rouge.compute(
            predictions=all_preds["sft"],
            references=all_refs,
            use_aggregator=False
        )["rougeL"]
        scst_scores = rouge.compute(
            predictions=all_preds["scst_sft"],
            references=all_refs,
            use_aggregator=False
        )["rougeL"]
        
        sig_test = statistical_significance_test(sft_scores, scst_scores, test_type="paired_t")
        print(f"  Test: {sig_test['test_name']}")
        print(f"  Mean SFT: {sig_test['mean2']:.4f}")
        print(f"  Mean SCST: {sig_test['mean1']:.4f}")
        print(f"  Difference: {sig_test['mean_diff']:.4f}")
        print(f"  p-value: {sig_test['p_value']:.6f}")
        print(f"  Significant: {sig_test['significant']}")
        
        # Save
        sig_path = os.path.join(CFG["analysis_dir"], "significance_test.json")
        with open(sig_path, 'w') as f:
            json.dump(sig_test, f, indent=2)
        print(f"  ✅ Saved to {sig_path}")

# ============================================================================
# CELL 9: Length Distribution Analysis
# ============================================================================

print("\n" + "="*70)
print("Length Distribution Analysis")
print("="*70)

length_comparison = {}
for model_name, preds in all_preds.items():
    length_analysis = length_distribution_analysis(preds, all_refs)
    length_comparison[model_name] = length_analysis

# Create length comparison table
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
print("\nLength Statistics:")
print(length_df.round(2))

length_path = os.path.join(CFG["analysis_dir"], "length_analysis.csv")
length_df.to_csv(length_path, index=False)
print(f"\n✅ Saved length analysis to {length_path}")

# ============================================================================
# CELL 10: Training Dynamics (if log files exist)
# ============================================================================

print("\n" + "="*70)
print("Plotting training dynamics...")
print("="*70)

# Look for CSV log files
log_files = {}
for model_name in available_models.keys():
    # Check common log file locations
    possible_paths = [
        os.path.join(CFG["results_dir"], f"{model_name}_training_log.csv"),
        os.path.join(CFG["results_dir"], "checkpoints", model_name, "training_log.csv"),
        f"./checkpoints/{model_name}/training_log.csv",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            log_files[model_name] = path
            break

if log_files:
    try:
        fig = plot_training_dynamics(
            log_files,
            metrics=["mean_reward", "kl", "policy_loss"],
            save_path=os.path.join(CFG["analysis_dir"], "training_dynamics.png"),
        )
        print(f"✅ Saved training dynamics plot")
        # plt.show()  # Uncomment to display
    except Exception as e:
        print(f"⚠️  Could not plot training dynamics: {e}")
else:
    print("⚠️  No training log files found")

# ============================================================================
# CELL 11: Summary Report
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*70)

print(f"\nAll analysis files saved to: {CFG['analysis_dir']}")
print("\nGenerated files:")
print("  - model_comparison.csv: Metrics comparison across models")
print("  - error_analysis_comparison.csv: Error type analysis")
print("  - example_comparisons.json: Side-by-side example comparisons")
print("  - length_analysis.csv: Length distribution statistics")
if log_files:
    print("  - training_dynamics.png: Training curves")

print("\nPer-model analysis directories:")
for model_name in available_models.keys():
    model_dir = os.path.join(CFG["analysis_dir"], model_name)
    if os.path.exists(model_dir):
        print(f"  - {model_dir}/")
        print(f"    - metrics.json")
        print(f"    - error_analysis.json")
        print(f"    - length_analysis.json")
        print(f"    - examples.json")

print("\n✅ All analysis complete! Use these results for your paper.")

