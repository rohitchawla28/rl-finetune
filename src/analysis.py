# src/analysis.py
"""
Comprehensive analysis tools for RL fine-tuning experiments.
Provides error analysis, example comparisons, training dynamics, and ablation studies.
"""

from typing import List, Dict, Tuple, Optional, Any
import json
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5Tokenizer, T5ForConditionalGeneration, PreTrainedModel, PreTrainedTokenizerBase
import torch

from .eval_utils import generate_summaries_batched, metrics_table, get_dataset_keys


def analyze_errors(
    preds: List[str],
    refs: List[str],
    articles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze types of errors in predictions.
    
    Error categories:
    - Length errors (too short/long)
    - Repetition errors (high repetition)
    - Coverage errors (low ROUGE overlap)
    - Hallucination (content not in article)
    
    Args:
        preds: List of predictions
        refs: List of references
        articles: Optional list of source articles
        save_path: Optional path to save analysis JSON
    
    Returns:
        Dictionary with error analysis
    """
    import evaluate
    rouge = evaluate.load("rouge")
    
    # Compute per-example metrics
    rouge_scores = rouge.compute(predictions=preds, references=refs, use_aggregator=False)
    
    pred_lens = [len(p.split()) for p in preds]
    ref_lens = [len(r.split()) for r in refs]
    compression_ratios = [p / max(1, r) for p, r in zip(pred_lens, ref_lens)]
    
    # Repetition rates
    def _rep_rate(text: str) -> float:
        toks = text.lower().split()
        return 0.0 if len(toks) <= 1 else 1.0 - (len(set(toks)) / len(toks))
    
    rep_rates = [_rep_rate(p) for p in preds]
    
    # Categorize errors
    errors = {
        "too_short": [],  # compression > 0.5 (much shorter than reference)
        "too_long": [],   # compression > 3.0 (much longer than reference)
        "high_repetition": [],  # repetition > 0.4
        "low_coverage": [],  # ROUGE-L < 0.1
        "combined": [],  # multiple issues
    }
    
    for i, (pred, ref, comp, rep, rouge_l) in enumerate(zip(
        preds, refs, compression_ratios, rep_rates, rouge_scores["rougeL"]
    )):
        issues = []
        if comp < 0.5:
            issues.append("too_short")
            errors["too_short"].append(i)
        if comp > 3.0:
            issues.append("too_long")
            errors["too_long"].append(i)
        if rep > 0.4:
            issues.append("high_repetition")
            errors["high_repetition"].append(i)
        if rouge_l < 0.1:
            issues.append("low_coverage")
            errors["low_coverage"].append(i)
        
        if len(issues) > 1:
            errors["combined"].append(i)
    
    # Summary statistics
    analysis = {
        "total_examples": len(preds),
        "error_counts": {k: len(v) for k, v in errors.items()},
        "error_percentages": {k: len(v) / len(preds) * 100 for k, v in errors.items()},
        "avg_compression": float(np.mean(compression_ratios)),
        "avg_repetition": float(np.mean(rep_rates)),
        "avg_rouge_l": float(np.mean(rouge_scores["rougeL"])),
        "compression_std": float(np.std(compression_ratios)),
        "repetition_std": float(np.std(rep_rates)),
        "rouge_l_std": float(np.std(rouge_scores["rougeL"])),
        "error_indices": errors,
    }
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    return analysis


def compare_examples(
    model_preds: Dict[str, List[str]],  # {model_name: [predictions]}
    refs: List[str],
    articles: Optional[List[str]] = None,
    indices: Optional[List[int]] = None,
    n_examples: int = 5,
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Compare example predictions across models.
    
    Args:
        model_preds: Dictionary mapping model names to prediction lists
        refs: Reference summaries
        articles: Optional source articles
        indices: Optional specific indices to compare
        n_examples: Number of examples to return
        save_path: Optional path to save comparison JSON
    
    Returns:
        List of example dictionaries
    """
    if indices is None:
        # Select diverse examples (high, medium, low ROUGE)
        import evaluate
        rouge = evaluate.load("rouge")
        
        # Use first model for selection
        first_model = list(model_preds.keys())[0]
        scores = rouge.compute(
            predictions=model_preds[first_model],
            references=refs,
            use_aggregator=False
        )["rougeL"]
        
        # Select examples across score range
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        step = len(sorted_indices) // n_examples
        indices = [sorted_indices[i * step] for i in range(n_examples)]
    
    examples = []
    for idx in indices[:n_examples]:
        ex = {
            "index": idx,
            "reference": refs[idx],
            "predictions": {name: preds[idx] for name, preds in model_preds.items()},
        }
        if articles:
            ex["article"] = articles[idx][:500] + "..." if len(articles[idx]) > 500 else articles[idx]
        examples.append(ex)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(examples, f, indent=2)
    
    return examples


def plot_training_dynamics(
    log_files: Dict[str, str],  # {model_name: path_to_csv}
    metrics: List[str] = ["mean_reward", "kl", "policy_loss"],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Plot training dynamics from CSV log files.
    
    Args:
        log_files: Dictionary mapping model names to CSV file paths
        metrics: List of metrics to plot
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for metric in metrics:
        ax = axes[metrics.index(metric)]
        for name, log_file in log_files.items():
            if os.path.exists(log_file):
                df = pd.read_csv(log_file)
                if metric in df.columns:
                    # Convert to numeric, handling empty strings
                    values = pd.to_numeric(df[metric], errors='coerce')
                    steps = df.get("step", range(len(values)))
                    ax.plot(steps, values, label=name, alpha=0.7)
        
        ax.set_xlabel("Step")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_reward_distribution(
    rewards: List[float],
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze reward distribution (for PPO/SCST training).
    
    Args:
        rewards: List of reward values
        save_path: Optional path to save analysis JSON
    
    Returns:
        Dictionary with reward statistics
    """
    rewards = np.array(rewards)
    
    analysis = {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards)),
        "q25": float(np.percentile(rewards, 25)),
        "q75": float(np.percentile(rewards, 75)),
        "variance": float(np.var(rewards)),
        "outliers": int(np.sum(np.abs(rewards - np.mean(rewards)) > 3 * np.std(rewards))),
    }
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    return analysis


def ablation_study(
    model_paths: Dict[str, str],  # {config_name: model_path}
    eval_dataset,
    text_key: str = "article",
    ref_key: str = "highlights",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run ablation study comparing different model configurations.
    
    Args:
        model_paths: Dictionary mapping configuration names to model paths
        eval_dataset: Evaluation dataset
        text_key: Key for input text
        ref_key: Key for reference
        save_path: Optional path to save results CSV
    
    Returns:
        DataFrame with results for each configuration
    """
    from .eval_utils import eval_model
    
    results = []
    for config_name, model_path in model_paths.items():
        print(f"Evaluating {config_name}...")
        scores = eval_model(
            model_path,
            eval_dataset,
            text_key=text_key,
            ref_key=ref_key,
        )
        scores["config"] = config_name
        results.append(scores)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ["config"] + [c for c in df.columns if c != "config"]
    df = df[cols]
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        df.to_csv(save_path, index=False)
    
    return df


def statistical_significance_test(
    scores1: List[float],
    scores2: List[float],
    test_type: str = "paired_t",
) -> Dict[str, Any]:
    """
    Test statistical significance between two sets of scores.
    
    Args:
        scores1: First set of scores (e.g., per-example ROUGE-L)
        scores2: Second set of scores
        test_type: Type of test ("paired_t", "wilcoxon", "mannwhitney")
    
    Returns:
        Dictionary with test results
    
    Note: Requires scipy to be installed.
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy is required for statistical significance testing. Install with: pip install scipy")
    
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    
    if test_type == "paired_t":
        statistic, p_value = stats.ttest_rel(scores1, scores2)
        test_name = "Paired t-test"
    elif test_type == "wilcoxon":
        statistic, p_value = stats.wilcoxon(scores1, scores2)
        test_name = "Wilcoxon signed-rank test"
    elif test_type == "mannwhitney":
        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    mean_diff = np.mean(scores1) - np.mean(scores2)
    std_diff = np.std(scores1 - scores2)
    
    result = {
        "test_name": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "mean1": float(np.mean(scores1)),
        "mean2": float(np.mean(scores2)),
        "std1": float(np.std(scores1)),
        "std2": float(np.std(scores2)),
    }
    
    return result


def length_distribution_analysis(
    preds: List[str],
    refs: List[str],
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze length distributions of predictions vs references.
    
    Args:
        preds: List of predictions
        refs: List of references
        save_path: Optional path to save analysis JSON
    
    Returns:
        Dictionary with length statistics
    """
    pred_lens = [len(p.split()) for p in preds]
    ref_lens = [len(r.split()) for r in refs]
    
    analysis = {
        "pred_mean": float(np.mean(pred_lens)),
        "pred_std": float(np.std(pred_lens)),
        "pred_min": int(np.min(pred_lens)),
        "pred_max": int(np.max(pred_lens)),
        "pred_median": float(np.median(pred_lens)),
        "ref_mean": float(np.mean(ref_lens)),
        "ref_std": float(np.std(ref_lens)),
        "ref_min": int(np.min(ref_lens)),
        "ref_max": int(np.max(ref_lens)),
        "ref_median": float(np.median(ref_lens)),
        "compression_mean": float(np.mean([p / max(1, r) for p, r in zip(pred_lens, ref_lens)])),
        "compression_std": float(np.std([p / max(1, r) for p, r in zip(pred_lens, ref_lens)])),
    }
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    return analysis


def ngram_analysis(
    texts: List[str],
    n: int = 1,
    top_k: int = 20,
) -> List[Tuple[str, int]]:
    """
    Analyze n-gram frequencies in texts.
    
    Args:
        texts: List of texts
        n: N-gram size (1=unigrams, 2=bigrams, etc.)
        top_k: Number of top n-grams to return
    
    Returns:
        List of (ngram, count) tuples, sorted by frequency
    """
    ngrams = []
    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            ngrams.append(ngram)
    
    counter = Counter(ngrams)
    return counter.most_common(top_k)


def comprehensive_model_analysis(
    model_path: str,
    eval_dataset,
    text_key: str = "article",
    ref_key: str = "highlights",
    output_dir: Optional[str] = None,
    n_examples: int = 10,
) -> Dict[str, Any]:
    """
    Run comprehensive analysis on a single model.
    
    Args:
        model_path: Path to model
        eval_dataset: Evaluation dataset
        text_key: Key for input text
        ref_key: Key for reference
        output_dir: Optional directory to save all analysis files
        n_examples: Number of example comparisons to generate
    
    Returns:
        Dictionary with all analysis results
    """
    from .eval_utils import generate_summaries_batched, metrics_table
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    
    print(f"Loading model from {model_path}...")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    print("Generating predictions...")
    preds, refs = generate_summaries_batched(
        model, tokenizer, eval_dataset,
        text_key=text_key, ref_key=ref_key,
    )
    
    # Get articles if available
    articles = None
    if hasattr(eval_dataset, text_key):
        articles = [ex[text_key] for ex in eval_dataset]
    
    print("Computing metrics...")
    metrics = metrics_table(preds, refs)
    
    print("Analyzing errors...")
    error_analysis = analyze_errors(preds, refs, articles)
    
    print("Analyzing length distribution...")
    length_analysis = length_distribution_analysis(preds, refs)
    
    print("Analyzing n-grams...")
    top_unigrams = ngram_analysis(preds, n=1, top_k=20)
    top_bigrams = ngram_analysis(preds, n=2, top_k=20)
    
    results = {
        "metrics": metrics,
        "error_analysis": error_analysis,
        "length_analysis": length_analysis,
        "top_unigrams": top_unigrams,
        "top_bigrams": top_bigrams,
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save error analysis
        with open(os.path.join(output_dir, "error_analysis.json"), 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        # Save length analysis
        with open(os.path.join(output_dir, "length_analysis.json"), 'w') as f:
            json.dump(length_analysis, f, indent=2)
        
        # Save examples
        examples = compare_examples(
            {"model": preds},
            refs,
            articles,
            n_examples=n_examples,
        )
        with open(os.path.join(output_dir, "examples.json"), 'w') as f:
            json.dump(examples, f, indent=2)
        
        print(f"Analysis saved to {output_dir}")
    
    return results

