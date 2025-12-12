# generate_final_plots.py
"""
Generate only the most important plots for the final report.
Creates a 'final_report_plots' folder with 4-5 key visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Configuration
CFG = {
    "analysis_dir": "./analysis",
    "logs_dir": "./logs",
    "final_plots_dir": "./final_report_plots",  # New folder for final plots
}

# Create final plots directory
os.makedirs(CFG["final_plots_dir"], exist_ok=True)

print("="*80)
print("GENERATING FINAL REPORT PLOTS")
print("="*80)
print(f"Output directory: {CFG['final_plots_dir']}/")

# Load data
print("\nLoading data files...")

comparison_path = os.path.join(CFG["analysis_dir"], "model_comparison.csv")
error_path = os.path.join(CFG["analysis_dir"], "error_analysis_comparison.csv")
history_path = os.path.join(CFG["logs_dir"], "sft_history.json")

comparison_df = None
error_df = None
sft_history = None

if os.path.exists(comparison_path):
    comparison_df = pd.read_csv(comparison_path)
    print(f"  Loaded: model_comparison.csv")
else:
    print(f"  ERROR: model_comparison.csv not found")
    print("  Run MASTER_NOTEBOOK.py first to generate data")
    sys.exit(1)

if os.path.exists(error_path):
    error_df = pd.read_csv(error_path)
    print(f"  Loaded: error_analysis_comparison.csv")

if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        sft_history = json.load(f)
    print(f"  Loaded: sft_history.json")

# ============================================================================
# PLOT 1: Model Comparison - All Metrics (Most Important)
# ============================================================================

print("\n1. Creating model comparison (all metrics)...")

if comparison_df is not None and len(comparison_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
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
    plot_path = os.path.join(CFG["final_plots_dir"], "1_model_comparison_all_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# ============================================================================
# PLOT 2: Improvement Over Baseline (Shows Contribution)
# ============================================================================

print("2. Creating improvement over baseline...")

if comparison_df is not None and len(comparison_df) > 0 and "base" in comparison_df["model"].values:
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
        plot_path = os.path.join(CFG["final_plots_dir"], "2_improvement_over_baseline.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()

# ============================================================================
# PLOT 3: Error Analysis (Shows Failure Modes)
# ============================================================================

print("3. Creating error analysis...")

if error_df is not None and len(error_df) > 0:
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
    plot_path = os.path.join(CFG["final_plots_dir"], "3_error_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# ============================================================================
# PLOT 4: Performance Heatmap (Clean Summary)
# ============================================================================

print("4. Creating performance heatmap...")

if comparison_df is not None and len(comparison_df) > 0:
    metrics_to_plot = ['rouge1', 'rouge2', 'rougeL', 'bleu']
    heatmap_data = comparison_df.set_index('model')[metrics_to_plot].T
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score'}, linewidths=0.5, ax=ax)
    ax.set_title('Performance Metrics Heatmap Across Models', fontsize=14, weight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(CFG["final_plots_dir"], "4_performance_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# ============================================================================
# PLOT 5: Compression and Repetition (Shows Quality Trade-offs)
# ============================================================================

print("5. Creating compression and repetition analysis...")

if comparison_df is not None and len(comparison_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = comparison_df["model"].values
    compression = comparison_df["compression"].values
    repetition = comparison_df["repetition"].values
    
    x_pos = range(len(models))
    
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
    plot_path = os.path.join(CFG["final_plots_dir"], "5_compression_repetition.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# ============================================================================
# Optional: SFT Training Curves (if available)
# ============================================================================

if sft_history is not None:
    print("6. Creating SFT training curves (bonus)...")
    
    if "train_loss" in sft_history and "val_loss" in sft_history:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(sft_history["train_loss"]) + 1)
        ax.plot(epochs, sft_history["train_loss"], 'b-o', label='Train Loss', linewidth=2, markersize=6)
        ax.plot(epochs, sft_history["val_loss"], 'r-s', label='Val Loss', linewidth=2, markersize=6)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('SFT Training and Validation Loss', fontsize=14, weight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(CFG["final_plots_dir"], "6_sft_training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()

# ============================================================================
# Create README for the folder
# ============================================================================

readme_content = """# Final Report Plots

This folder contains the key visualizations for the final project report.

## Plots Included:

1. **1_model_comparison_all_metrics.png**
   - Shows ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores for all models
   - Primary results visualization

2. **2_improvement_over_baseline.png**
   - Shows relative and absolute improvements over base model
   - Demonstrates contribution of each method

3. **3_error_analysis.png**
   - Breakdown of error types (too short, too long, high repetition, low coverage)
   - Shows what failure modes each model exhibits

4. **4_performance_heatmap.png**
   - Clean heatmap visualization of all metrics across models
   - Easy-to-read summary

5. **5_compression_repetition.png**
   - Compression ratio and repetition rate comparison
   - Shows quality trade-offs

6. **6_sft_training_curves.png** (if available)
   - Training and validation loss over epochs
   - Shows training dynamics

## Usage in Paper:

These plots are ready to include in your final report. All plots are saved at 300 DPI for high-quality printing.

Suggested figure captions:
- Figure 1: Model comparison (Plot 1)
- Figure 2: Improvement over baseline (Plot 2)
- Figure 3: Error analysis (Plot 3)
- Figure 4: Performance heatmap (Plot 4)
- Figure 5: Compression and repetition (Plot 5)
"""

readme_path = os.path.join(CFG["final_plots_dir"], "README.md")
with open(readme_path, 'w') as f:
    f.write(readme_content)

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("FINAL REPORT PLOTS GENERATED")
print("="*80)
print(f"\nAll plots saved to: {CFG['final_plots_dir']}/")
print("\nGenerated plots:")
print("  1. model_comparison_all_metrics.png - All key metrics comparison")
print("  2. improvement_over_baseline.png - Shows contribution of methods")
print("  3. error_analysis.png - Error type breakdown")
print("  4. performance_heatmap.png - Clean summary visualization")
print("  5. compression_repetition.png - Quality trade-offs")
if sft_history is not None:
    print("  6. sft_training_curves.png - Training dynamics")
print("\nThese are the plots to include in your final report!")
print("="*80)

