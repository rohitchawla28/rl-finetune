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
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the correct directory (rl-finetune/)")
    print("2. Try restarting the kernel/runtime")
    print("3. Verify src/eval_utils.py contains get_dataset_keys function")
    print("4. Run: import importlib; importlib.reload(sys.modules.get('src.eval_utils', None))")
    raise

# Verify get_dataset_keys works
try:
    test_keys = get_dataset_keys("cnn_dailymail")
    print(f"✅ Verified get_dataset_keys works: {test_keys}")
except Exception as e:
    print(f"⚠️  Warning: get_dataset_keys test failed: {e}")

print("="*80)
print("RL FINE-TUNING MASTER NOTEBOOK")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# CELL 1: Configuration and Setup
# ============================================================================

print("\n" + "="*80)
print("CELL 1: Configuration and Setup")
print("="*80)

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

print("✅ Configuration loaded!")
print(f"   Dataset: {CFG['dataset']}")
print(f"   Base model: {CFG['base_model']}")
print(f"   Device: {CFG['device']}")
print(f"   Training examples: {CFG['n_train']}")
print(f"   Evaluation examples: {CFG['n_eval']}")
print(f"   Text key: {text_key}, Reference key: {ref_key}")

# ============================================================================
# CELL 2: Load Datasets
# ============================================================================

print("\n" + "="*80)
print("CELL 2: Load Datasets")
print("="*80)

print("Loading CNN/DailyMail dataset...")

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
    
    print(f"✅ Dataset loaded!")
    print(f"   SFT training: {len(train_raw_sft)} examples")
    print(f"   RL training: {len(train_raw_rl)} examples")
    print(f"   SFT validation: {len(val_raw_sft)} examples")
    print(f"   Evaluation: {len(eval_raw)} examples")
else:
    raise ValueError(f"Unknown dataset: {CFG['dataset']}")

# ============================================================================
# CELL 3: Train SFT Baseline
# ============================================================================

print("\n" + "="*80)
print("CELL 3: Train SFT Baseline")
print("="*80)

sft_output_dir = os.path.join(CFG["checkpoints_dir"], "flan_t5_sft")
sft_best_dir = os.path.join(sft_output_dir, "best")

# Check if SFT already exists
if os.path.exists(sft_best_dir):
    print(f"⚠️  SFT model already exists at {sft_best_dir}")
    print("   Skipping SFT training. Set SKIP_SFT=False to retrain.")
    SKIP_SFT = True
else:
    SKIP_SFT = False

if not SKIP_SFT:
    print("Training SFT model...")
    print(f"   Output directory: {sft_output_dir}")
    print(f"   This may take 30-60 minutes...")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(CFG["base_model"])
        
        sft_model, sft_history = run_sft(
            train_dataset=train_raw_sft,
            val_dataset=val_raw_sft,
            tokenizer=tokenizer,
            model_name=CFG["base_model"],
            batch_size=CFG["sft_batch_size"],
            num_epochs=CFG["sft_epochs"],
            lr=CFG["sft_lr"],
            warmup_ratio=CFG["sft_warmup_ratio"],
            output_dir=sft_output_dir,
            device=CFG["device"],
            # Evaluation during training
            eval_dataset=eval_raw,
            text_key=text_key,
            ref_key=ref_key,
            eval_batch_size=16,
            eval_max_input_len=CFG["max_input_len"],
            eval_max_new_tokens=CFG["max_new_tokens"],
            eval_num_beams=CFG["num_beams"],
            print_metrics=True,
        )
        
        print(f"✅ SFT training complete!")
        print(f"   Best model saved to: {sft_best_dir}")
        
        # Save training history
        history_path = os.path.join(CFG["logs_dir"], "sft_history.json")
        with open(history_path, 'w') as f:
            json.dump(sft_history, f, indent=2)
        print(f"   Training history saved to: {history_path}")
        
    except Exception as e:
        print(f"❌ Error during SFT training: {e}")
        import traceback
        traceback.print_exc()
        raise

# Evaluate SFT
print("\nEvaluating SFT model...")
sft_scores = eval_model(
    sft_best_dir,
    eval_raw,
    text_key=text_key,
    ref_key=ref_key,
    save_path=os.path.join(CFG["results_dir"], "results_sft.json"),
)
print(f"✅ SFT Results:")
print(f"   ROUGE-L: {sft_scores['rougeL']:.4f}")
print(f"   ROUGE-1: {sft_scores['rouge1']:.4f}")
print(f"   ROUGE-2: {sft_scores['rouge2']:.4f}")
print(f"   BLEU: {sft_scores['bleu']:.4f}")
print(f"   Compression: {sft_scores['compression']:.4f}")
print(f"   Repetition: {sft_scores['repetition']:.4f}")

# ============================================================================
# CELL 4: Train SCST Models
# ============================================================================

print("\n" + "="*80)
print("CELL 4: Train SCST Models")
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
    print(f"\n{'='*80}")
    print(f"Training: {exp['name']}")
    print(f"{'='*80}")
    print(f"Description: {exp['description']}")
    
    out_dir = os.path.join(CFG["checkpoints_dir"], exp["name"])
    
    # Check if already exists
    if os.path.exists(out_dir):
        print(f"⚠️  Model already exists at {out_dir}")
        print("   Skipping training. Delete directory to retrain.")
        skip = True
    else:
        skip = False
    
    if not skip:
        print(f"   Starting model: {exp['model_path']}")
        print(f"   Output directory: {out_dir}")
        print(f"   This may take 20-40 minutes...")
        
        try:
            run_scst(
                model_name=exp["model_path"],
                out_dir=out_dir,
                n_train=CFG["n_train"],
                batch_size=CFG["scst_batch_size"],
                epochs=CFG["scst_epochs"],
                lr=CFG["scst_lr"],
                warmup_ratio=CFG["scst_warmup_ratio"],
                train_dataset=train_raw_rl,
                text_key=text_key,
                ref_key=ref_key,
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
                # Data filtering
                min_len=CFG["min_len"],
                max_len=CFG["max_len"],
                seed=CFG["seed"],
            )
            
            print(f"✅ {exp['name']} training complete!")
            
        except Exception as e:
            print(f"❌ Error during {exp['name']} training: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Evaluate
    print(f"\nEvaluating {exp['name']}...")
    try:
        scores = eval_model(
            out_dir,
            eval_raw,
            text_key=text_key,
            ref_key=ref_key,
            save_path=os.path.join(CFG["results_dir"], f"results_{exp['name']}.json"),
        )
        scst_results[exp["name"]] = scores
        
        print(f"✅ {exp['name']} Results:")
        print(f"   ROUGE-L: {scores['rougeL']:.4f}")
        print(f"   ROUGE-1: {scores['rouge1']:.4f}")
        print(f"   ROUGE-2: {scores['rouge2']:.4f}")
        print(f"   BLEU: {scores['bleu']:.4f}")
        print(f"   Compression: {scores['compression']:.4f}")
        print(f"   Repetition: {scores['repetition']:.4f}")
        
    except Exception as e:
        print(f"❌ Error evaluating {exp['name']}: {e}")

# ============================================================================
# CELL 5: Evaluate Base Model (for comparison)
# ============================================================================

print("\n" + "="*80)
print("CELL 5: Evaluate Base Model")
print("="*80)

print("Evaluating base model...")
base_scores = eval_model(
    CFG["base_model"],
    eval_raw,
    text_key=text_key,
    ref_key=ref_key,
    save_path=os.path.join(CFG["results_dir"], "results_base.json"),
)

print(f"✅ Base Model Results:")
print(f"   ROUGE-L: {base_scores['rougeL']:.4f}")
print(f"   ROUGE-1: {base_scores['rouge1']:.4f}")
print(f"   ROUGE-2: {base_scores['rouge2']:.4f}")
print(f"   BLEU: {base_scores['bleu']:.4f}")
print(f"   Compression: {base_scores['compression']:.4f}")
print(f"   Repetition: {base_scores['repetition']:.4f}")

# ============================================================================
# CELL 6: Generate Predictions for All Models
# ============================================================================

print("\n" + "="*80)
print("CELL 6: Generate Predictions for All Models")
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

print(f"Generating predictions for {len(all_models)} models...")

all_preds = {}
all_refs = None

for model_name, model_path in all_models.items():
    print(f"\n  Generating for {model_name}...")
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
        
        print(f"    ✅ Generated {len(preds)} predictions")
        
    except Exception as e:
        print(f"    ❌ Error: {e}")

print(f"\n✅ Generated predictions for {len(all_preds)} models")

# ============================================================================
# CELL 7: Comprehensive Analysis
# ============================================================================

print("\n" + "="*80)
print("CELL 7: Comprehensive Analysis")
print("="*80)

# 7.1: Model Comparison Table
print("\n7.1: Creating model comparison table...")

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
print(f"✅ Saved to {comparison_path}")
print("\nModel Comparison:")
print(comparison_df.round(4).to_string(index=False))

# 7.2: Error Analysis
print("\n7.2: Running error analysis...")

error_comparison = {}
articles = [ex[text_key] for ex in eval_raw] if hasattr(eval_raw[0], text_key) else None

for model_name, preds in all_preds.items():
    print(f"  Analyzing {model_name}...")
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
print(f"✅ Saved to {error_path}")
print("\nError Analysis:")
print(error_df.round(2).to_string(index=False))

# 7.3: Example Comparisons
print("\n7.3: Generating example comparisons...")

examples = compare_examples(
    model_preds=all_preds,
    refs=all_refs,
    articles=articles,
    n_examples=10,
    save_path=os.path.join(CFG["analysis_dir"], "example_comparisons.json"),
)

print(f"✅ Generated {len(examples)} example comparisons")
print("\nSample Example (first one):")
if examples:
    ex = examples[0]
    print(f"\nIndex: {ex['index']}")
    print(f"\nReference:")
    print(f"  {ex['reference'][:200]}...")
    print(f"\nPredictions:")
    for model_name, pred in ex["predictions"].items():
        print(f"  {model_name}: {pred[:150]}...")

# 7.4: Length Distribution Analysis
print("\n7.4: Analyzing length distributions...")

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
print(f"✅ Saved to {length_path}")

# 7.5: Statistical Significance Testing
print("\n7.5: Testing statistical significance...")

try:
    import evaluate
    rouge = evaluate.load("rouge")
    
    # Compare SFT vs SCST from SFT
    if "sft" in all_preds and "scst_from_sft" in all_preds:
        print("  Testing SFT vs SCST from SFT...")
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
        
        print(f"  ✅ Test results:")
        print(f"     Mean SFT: {sig_test['mean2']:.4f}")
        print(f"     Mean SCST: {sig_test['mean1']:.4f}")
        print(f"     Difference: {sig_test['mean_diff']:+.4f}")
        print(f"     p-value: {sig_test['p_value']:.6f}")
        print(f"     Significant: {sig_test['significant']}")
        print(f"     Saved to {sig_path}")
    
    # Compare Base vs SCST from Base
    if "base" in all_preds and "scst_from_base" in all_preds:
        print("  Testing Base vs SCST from Base...")
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
        
        print(f"  ✅ Test results:")
        print(f"     Mean Base: {sig_test_base['mean2']:.4f}")
        print(f"     Mean SCST: {sig_test_base['mean1']:.4f}")
        print(f"     Difference: {sig_test_base['mean_diff']:+.4f}")
        print(f"     p-value: {sig_test_base['p_value']:.6f}")
        print(f"     Significant: {sig_test_base['significant']}")

except Exception as e:
    print(f"  ⚠️  Could not run significance tests: {e}")

# 7.6: Comprehensive Analysis for Each Model
print("\n7.6: Running comprehensive analysis for each model...")

for model_name, model_path in all_models.items():
    if model_name in all_preds:
        print(f"  Analyzing {model_name}...")
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
            print(f"    ✅ Complete")
        except Exception as e:
            print(f"    ⚠️  Error: {e}")

# ============================================================================
# CELL 8: Final Summary and Results
# ============================================================================

print("\n" + "="*80)
print("CELL 8: Final Summary and Results")
print("="*80)

print("\n" + "="*80)
print("EXPERIMENT COMPLETE - FINAL RESULTS")
print("="*80)

print("\n📊 Model Comparison:")
print(comparison_df.round(4).to_string(index=False))

print("\n📈 Key Findings:")
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

print("\n📁 Generated Files:")
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

print("\n✅ All experiments and analysis complete!")
print(f"   Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "="*80)
print("Next steps:")
print("  1. Review model_comparison.csv for main results")
print("  2. Review error_analysis_comparison.csv for error insights")
print("  3. Review example_comparisons.json for qualitative examples")
print("  4. Use these files to populate your paper tables and figures")
print("="*80)

