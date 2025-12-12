# generate_plots.py
"""
Standalone script to generate all visualization plots for the final report.
Run this after MASTER_NOTEBOOK.py has completed to generate all plots.

Usage:
    python generate_plots.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Configuration
CFG = {
    "results_dir": "./results",
    "analysis_dir": "./analysis",
    "logs_dir": "./logs",
}

# Create directories if they don't exist
for dir_name in [CFG["results_dir"], CFG["analysis_dir"], CFG["logs_dir"]]:
    os.makedirs(dir_name, exist_ok=True)

print("="*80)
print("GENERATING ALL PLOTS")
print("="*80)

# Load data
print("\nLoading data files...")

comparison_path = os.path.join(CFG["analysis_dir"], "model_comparison.csv")
error_path = os.path.join(CFG["analysis_dir"], "error_analysis_comparison.csv")
length_path = os.path.join(CFG["analysis_dir"], "length_analysis.csv")
history_path = os.path.join(CFG["logs_dir"], "sft_history.json")

comparison_df = None
error_df = None
length_df = None
sft_history = None

if os.path.exists(comparison_path):
    comparison_df = pd.read_csv(comparison_path)
    print(f"  Loaded: {comparison_path}")
else:
    print(f"  WARNING: {comparison_path} not found")

if os.path.exists(error_path):
    error_df = pd.read_csv(error_path)
    print(f"  Loaded: {error_path}")
else:
    print(f"  WARNING: {error_path} not found")

if os.path.exists(length_path):
    length_df = pd.read_csv(length_path)
    print(f"  Loaded: {length_path}")
else:
    print(f"  WARNING: {length_path} not found")

if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        sft_history = json.load(f)
    print(f"  Loaded: {history_path}")
else:
    print(f"  WARNING: {history_path} not found (SFT training curves will be skipped)")

# ============================================================================
# Plot 1: Model Comparison Metrics (2x2 grid)
# ============================================================================

if comparison_df is not None and len(comparison_df) > 0:
    print("\n1. Creating model comparison plots...")
    
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
    plot_path = os.path.join(CFG["analysis_dir"], "model_comparison_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

# ============================================================================
# Plot 2: Compression and Repetition
# ============================================================================

if comparison_df is not None and len(comparison_df) > 0:
    print("2. Creating compression and repetition plots...")
    
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

# ============================================================================
# Plot 3: Error Analysis
# ============================================================================

if error_df is not None and len(error_df) > 0:
    print("3. Creating error analysis plots...")
    
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

# ============================================================================
# Plot 4: Length Distribution
# ============================================================================

if length_df is not None and len(length_df) > 0:
    print("4. Creating length distribution plots...")
    
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

# ============================================================================
# Plot 5: SFT Training Curves
# ============================================================================

if sft_history is not None:
    print("5. Creating SFT training curves...")
    
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

# ============================================================================
# Plot 6: Combined ROUGE Comparison
# ============================================================================

if comparison_df is not None and len(comparison_df) > 0:
    print("6. Creating combined ROUGE comparison...")
    
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

# ============================================================================
# Plot 7: Performance Heatmap
# ============================================================================

if comparison_df is not None and len(comparison_df) > 0:
    print("7. Creating performance heatmap...")
    
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

# ============================================================================
# Plot 8: Improvement Over Baseline
# ============================================================================

if comparison_df is not None and len(comparison_df) > 0 and "base" in comparison_df["model"].values:
    print("8. Creating improvement over baseline plot...")
    
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

# ============================================================================
# Plot 9: Summary Statistics Table
# ============================================================================

if comparison_df is not None and len(comparison_df) > 0:
    print("9. Creating summary statistics table...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
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

# ============================================================================
# Plot 10: Error Distribution Stacked
# ============================================================================

if error_df is not None and len(error_df) > 0:
    print("10. Creating error distribution stacked plot...")
    
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

# ============================================================================
# Plot 11: Performance Radar Chart
# ============================================================================

if comparison_df is not None and len(comparison_df) > 0:
    print("11. Creating radar chart...")
    
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

# ============================================================================
# Plot 12: Length vs Quality Scatter
# ============================================================================

if comparison_df is not None and len(comparison_df) > 0:
    print("12. Creating length vs quality analysis...")
    
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

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("ALL PLOTS GENERATED")
print("="*80)
print(f"\nAll plots saved to: {CFG['analysis_dir']}/")
print("\nGenerated files:")
print("  1. model_comparison_metrics.png")
print("  2. compression_repetition.png")
print("  3. error_analysis.png")
print("  4. length_distribution.png")
print("  5. sft_training_curves.png (if SFT history available)")
print("  6. sft_metrics_over_time.png (if SFT history available)")
print("  7. rouge_comparison_combined.png")
print("  8. performance_heatmap.png")
print("  9. improvement_over_baseline.png")
print("  10. summary_table.png")
print("  11. error_distribution_stacked.png")
print("  12. performance_radar.png")
print("  13. length_quality_analysis.png")
print("\nAll plots are saved at 300 DPI for paper inclusion.")
print("="*80)

