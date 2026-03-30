"""
Evaluation and Comparison Module
Computes QA metrics (Exact Match, F1) and generates comparison visualizations.
"""

import os
import json
import collections
import string
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, ground_truth):
    """Compute token-level F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_exact_match(prediction, ground_truth):
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def evaluate_qa_model(model, val_dataset, val_raw, tokenizer):
    """
    Evaluate a QA model on the validation set.
    
    Args:
        model: Trained QA model
        val_dataset: Preprocessed validation dataset (tokenized)
        val_raw: Raw validation data with answers
        tokenizer: BERT tokenizer
    
    Returns:
        metrics: Dict with exact_match, f1, and per-example scores
    """
    print("\n[EVAL] Evaluating QA model...")
    
    model.eval()
    model.to(config.DEVICE)
    
    # Store columns that are needed
    example_ids = val_dataset["example_id"]
    offset_mappings = val_dataset["offset_mapping"]
    
    # Remove non-model columns for inference
    eval_dataset = val_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataset.set_format(type="torch")
    
    eval_loader = DataLoader(eval_dataset, batch_size=config.QA_BATCH_SIZE)
    
    all_start_logits = []
    all_end_logits = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", unit="batch"):
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())
    
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)
    
    # Map predictions back to examples
    example_to_features = collections.defaultdict(list)
    for idx, example_id in enumerate(example_ids):
        example_to_features[example_id].append(idx)
    
    predictions = {}
    
    for example in val_raw:
        example_id = example["id"]
        context = example["context"]
        
        if example_id not in example_to_features:
            predictions[example_id] = ""
            continue
        
        feature_indices = example_to_features[example_id]
        
        best_score = -float("inf")
        best_answer = ""
        
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offsets = offset_mappings[feature_index]
            
            # Get top start and end positions
            start_indices = np.argsort(start_logits)[-20:][::-1]
            end_indices = np.argsort(end_logits)[-20:][::-1]
            
            for start_index in start_indices:
                for end_index in end_indices:
                    # Skip invalid spans
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > 30:
                        continue
                    
                    score = start_logits[start_index] + end_logits[end_index]
                    
                    if score > best_score:
                        best_score = score
                        start_char = offsets[start_index][0]
                        end_char = offsets[end_index][1]
                        best_answer = context[start_char:end_char]
        
        predictions[example_id] = best_answer
    
    # Compute metrics
    exact_matches = []
    f1_scores = []
    
    for example in val_raw:
        example_id = example["id"]
        gold_answers = example["answers"]["text"]
        prediction = predictions.get(example_id, "")
        
        if not gold_answers:
            continue
        
        # Take max over all gold answers
        em = max(compute_exact_match(prediction, ans) for ans in gold_answers)
        f1 = max(compute_f1(prediction, ans) for ans in gold_answers)
        
        exact_matches.append(em)
        f1_scores.append(f1)
    
    metrics = {
        "exact_match": np.mean(exact_matches) * 100,
        "f1": np.mean(f1_scores) * 100,
        "num_examples": len(exact_matches),
        "per_example_em": exact_matches,
        "per_example_f1": f1_scores,
    }
    
    print(f"[EVAL] Exact Match: {metrics['exact_match']:.2f}%")
    print(f"[EVAL] F1 Score: {metrics['f1']:.2f}%")
    print(f"[EVAL] Evaluated on {metrics['num_examples']} examples")
    
    return metrics


def generate_comparison_plots(baseline_results, adapted_results,
                               baseline_history, adapted_history,
                               mlm_history=None):
    """
    Generate comprehensive comparison visualizations.
    
    Args:
        baseline_results: Dict with baseline metrics
        adapted_results: Dict with domain-adapted metrics
        baseline_history: Baseline training history
        adapted_history: Domain-adapted training history
        mlm_history: MLM pretraining history (optional)
    """
    print("\n[VIZ] Generating comparison plots...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.figsize': (14, 10),
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline BERT vs Domain-Adapted BERT: Comparison Results",
                 fontsize=16, fontweight='bold', y=1.02)
    
    # ---- Plot 1: Accuracy & F1 Comparison ----
    ax1 = axes[0, 0]
    metrics_names = ['Exact Match (%)', 'F1 Score (%)']
    baseline_vals = [baseline_results['exact_match'], baseline_results['f1']]
    adapted_vals = [adapted_results['exact_match'], adapted_results['f1']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline BERT',
                     color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, adapted_vals, width, label='Domain-Adapted BERT',
                     color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax1.set_ylabel('Score (%)')
    ax1.set_title('QA Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.set_ylim(0, max(max(baseline_vals), max(adapted_vals)) * 1.3)
    
    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)
    
    # ---- Plot 2: Training Loss Curves ----
    ax2 = axes[0, 1]
    ax2.plot(baseline_history['epoch_losses'], 'b-o', label='Baseline BERT',
             linewidth=2, markersize=8, color='#3498db')
    ax2.plot(adapted_history['epoch_losses'], 'r-s', label='Domain-Adapted BERT',
             linewidth=2, markersize=8, color='#e74c3c')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Loss')
    ax2.set_title('QA Training Loss Over Epochs')
    ax2.legend()
    ax2.set_xticks(range(len(baseline_history['epoch_losses'])))
    ax2.set_xticklabels([str(i+1) for i in range(len(baseline_history['epoch_losses']))])
    
    # ---- Plot 3: Training Time Comparison ----
    ax3 = axes[1, 0]
    time_labels = ['Baseline\nBERT']
    time_values = [baseline_results['training_time']]
    colors = ['#3498db']
    
    if mlm_history is not None:
        time_labels.append('Domain MLM\nPretraining')
        time_values.append(adapted_results.get('mlm_time', 0))
        colors.append('#2ecc71')
    
    time_labels.append('Domain-Adapted\nQA Fine-tuning')
    time_values.append(adapted_results['training_time'])
    colors.append('#e74c3c')
    
    bars = ax3.bar(time_labels, time_values, color=colors, alpha=0.85,
                    edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Training Time Breakdown')
    
    for bar, val in zip(bars, time_values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
    
    # ---- Plot 4: F1 Score Distribution ----
    ax4 = axes[1, 1]
    baseline_f1s = baseline_results.get('per_example_f1', [])
    adapted_f1s = adapted_results.get('per_example_f1', [])
    
    if baseline_f1s and adapted_f1s:
        ax4.hist(baseline_f1s, bins=20, alpha=0.6, label='Baseline BERT',
                 color='#3498db', edgecolor='black', linewidth=0.5)
        ax4.hist(adapted_f1s, bins=20, alpha=0.6, label='Domain-Adapted BERT',
                 color='#e74c3c', edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('F1 Score')
        ax4.set_ylabel('Number of Examples')
        ax4.set_title('F1 Score Distribution')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No per-example\nscores available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('F1 Score Distribution')
    
    plt.tight_layout()
    plot_path = os.path.join(config.PLOTS_DIR, "comparison_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Comparison plot saved to: {plot_path}")
    
    # ---- Additional Plot: MLM Training Loss ----
    if mlm_history is not None and mlm_history.get('epoch_losses'):
        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.plot(mlm_history['epoch_losses'], 'g-o', linewidth=2, markersize=8,
                color='#2ecc71')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average MLM Loss')
        ax.set_title('Domain-Specific MLM Pretraining Loss')
        ax.set_xticks(range(len(mlm_history['epoch_losses'])))
        ax.set_xticklabels([str(i+1) for i in range(len(mlm_history['epoch_losses']))])
        plt.tight_layout()
        mlm_plot_path = os.path.join(config.PLOTS_DIR, "mlm_pretraining_loss.png")
        plt.savefig(mlm_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIZ] MLM loss plot saved to: {mlm_plot_path}")
    
    # ---- Step-level loss comparison ----
    fig3, ax = plt.subplots(figsize=(10, 5))
    ax.plot(baseline_history['step_losses'], alpha=0.7, label='Baseline BERT',
            color='#3498db', linewidth=0.8)
    ax.plot(adapted_history['step_losses'], alpha=0.7, label='Domain-Adapted BERT',
            color='#e74c3c', linewidth=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Step-Level QA Training Loss')
    ax.legend()
    plt.tight_layout()
    step_plot_path = os.path.join(config.PLOTS_DIR, "step_loss_comparison.png")
    plt.savefig(step_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Step loss plot saved to: {step_plot_path}")
    
    return plot_path


def save_results(baseline_results, adapted_results, 
                  baseline_history, adapted_history, mlm_history=None):
    """Save all results to JSON for the report."""
    results = {
        "baseline": {
            "exact_match": baseline_results["exact_match"],
            "f1": baseline_results["f1"],
            "training_time": baseline_results["training_time"],
            "parameters": baseline_results["parameters"],
            "num_examples": baseline_results["num_examples"],
            "final_loss": baseline_history["epoch_losses"][-1] if baseline_history["epoch_losses"] else None,
        },
        "domain_adapted": {
            "exact_match": adapted_results["exact_match"],
            "f1": adapted_results["f1"],
            "training_time": adapted_results["training_time"],
            "total_time": adapted_results.get("total_time", adapted_results["training_time"]),
            "mlm_time": adapted_results.get("mlm_time", 0),
            "parameters": adapted_results["parameters"],
            "num_examples": adapted_results["num_examples"],
            "final_loss": adapted_history["epoch_losses"][-1] if adapted_history["epoch_losses"] else None,
        },
        "improvement": {
            "em_diff": adapted_results["exact_match"] - baseline_results["exact_match"],
            "f1_diff": adapted_results["f1"] - baseline_results["f1"],
        },
        "config": {
            "model_name": config.MODEL_NAME,
            "qa_train_samples": config.QA_TRAIN_SAMPLES,
            "qa_val_samples": config.QA_VAL_SAMPLES,
            "qa_epochs": config.QA_EPOCHS,
            "qa_learning_rate": config.QA_LEARNING_RATE,
            "qa_batch_size": config.QA_BATCH_SIZE,
            "mlm_train_samples": config.MLM_TRAIN_SAMPLES,
            "mlm_epochs": config.MLM_EPOCHS,
            "mlm_learning_rate": config.MLM_LEARNING_RATE,
            "device": str(config.DEVICE),
        }
    }
    
    results_path = os.path.join(config.RESULTS_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[RESULTS] Results saved to: {results_path}")
    
    # Print summary table
    print("\n" + "="*75)
    print("RESULTS SUMMARY TABLE")
    print("="*75)
    print(f"{'Model':<25} {'EM (%)':<12} {'F1 (%)':<12} {'Train Time':<15} {'Parameters':<15}")
    print("-"*75)
    print(f"{'Baseline BERT':<25} {baseline_results['exact_match']:<12.2f} "
          f"{baseline_results['f1']:<12.2f} {baseline_results['training_time']:<15.1f} "
          f"{baseline_results['parameters']:<15,}")
    
    total_adapted_time = adapted_results.get('total_time', adapted_results['training_time'])
    print(f"{'Domain-Adapted BERT':<25} {adapted_results['exact_match']:<12.2f} "
          f"{adapted_results['f1']:<12.2f} {total_adapted_time:<15.1f} "
          f"{adapted_results['parameters']:<15,}")
    print("-"*75)
    print(f"{'Improvement':<25} {results['improvement']['em_diff']:<+12.2f} "
          f"{results['improvement']['f1_diff']:<+12.2f}")
    print("="*75)
    
    return results
