"""
Main Entry Point - NLP Assignment 2
Part 1: Domain Adaptation Gap
Roll Number: 2305990

This script orchestrates the complete experiment pipeline:
1. Load and preprocess SQuAD data
2. Train baseline BERT QA model
3. Perform domain-specific MLM pretraining (medical corpus)
4. Train domain-adapted BERT QA model
5. Evaluate both models
6. Generate comparison results and visualizations

Usage:
    python run_all.py
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import BertTokenizerFast

import config
from data_utils import set_seed, load_squad_data, preprocess_qa_training, preprocess_qa_validation
from baseline_bert_qa import train_baseline_qa
from domain_adapted_bert_qa import mlm_pretrain, train_domain_adapted_qa
from evaluate_and_compare import evaluate_qa_model, generate_comparison_plots, save_results


def main():
    """Run the complete experiment pipeline."""
    total_start = time.time()
    
    print("="*70)
    print("  NLP ASSIGNMENT 2: DOMAIN ADAPTATION GAP")
    print("  Part 1 - Roll Number: 2305990")
    print("  Advancing BERT: Domain-Specific Pretraining for QA")
    print("="*70)
    print(f"\nDevice: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Seed: {config.SEED}")
    print()
    
    # Set reproducibility
    set_seed()
    
    # ============================================================
    # Step 1: Load Tokenizer and Data
    # ============================================================
    print("\n" + "="*60)
    print("STEP 1: LOADING TOKENIZER AND DATA")
    print("="*60)
    
    tokenizer = BertTokenizerFast.from_pretrained(config.MODEL_NAME)
    train_raw, val_raw = load_squad_data()
    
    # Preprocess training data
    print("[PREPROCESS] Tokenizing training data for QA...")
    train_tokenized = train_raw.map(
        lambda x: preprocess_qa_training(x, tokenizer),
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenizing train",
    )
    
    # Preprocess validation data
    print("[PREPROCESS] Tokenizing validation data for QA...")
    val_tokenized = val_raw.map(
        lambda x: preprocess_qa_validation(x, tokenizer),
        batched=True,
        remove_columns=val_raw.column_names,
        desc="Tokenizing val",
    )
    
    print(f"[PREPROCESS] Training features: {len(train_tokenized)}")
    print(f"[PREPROCESS] Validation features: {len(val_tokenized)}")
    
    # ============================================================
    # Step 2: Train Baseline BERT QA
    # ============================================================
    baseline_model, baseline_time, baseline_history, baseline_params = train_baseline_qa(
        train_tokenized, tokenizer
    )
    
    # ============================================================
    # Step 3: Evaluate Baseline
    # ============================================================
    print("\n" + "="*60)
    print("STEP 3: EVALUATING BASELINE MODEL")
    print("="*60)
    
    # Need fresh validation tokenization for evaluation
    val_eval = val_raw.map(
        lambda x: preprocess_qa_validation(x, tokenizer),
        batched=True,
        remove_columns=val_raw.column_names,
        desc="Re-tokenizing val for baseline eval",
    )
    
    baseline_metrics = evaluate_qa_model(baseline_model, val_eval, val_raw, tokenizer)
    
    # Save baseline model
    baseline_save_path = os.path.join(config.MODELS_DIR, "baseline_bert_qa")
    print(f"[SAVE] Saving baseline model to {baseline_save_path}")
    baseline_model.save_pretrained(baseline_save_path)
    tokenizer.save_pretrained(baseline_save_path)
    print(f"[SAVE] Baseline model saved successfully!")
    
    # Free memory
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ============================================================
    # Step 4: Domain-Specific MLM Pretraining
    # ============================================================
    mlm_model, mlm_time, mlm_history = mlm_pretrain(tokenizer)
    
    # Save MLM pretrained model
    mlm_save_path = os.path.join(config.MODELS_DIR, "domain_mlm_pretrained")
    print(f"[SAVE] Saving MLM pretrained model to {mlm_save_path}")
    mlm_model.save_pretrained(mlm_save_path)
    tokenizer.save_pretrained(mlm_save_path)
    print(f"[SAVE] MLM pretrained model saved successfully!")
    
    # ============================================================
    # Step 5: Train Domain-Adapted BERT QA
    # ============================================================
    # Re-tokenize training data (fresh copy for the domain-adapted model)
    train_tokenized_da = train_raw.map(
        lambda x: preprocess_qa_training(x, tokenizer),
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenizing train for domain-adapted",
    )
    
    adapted_model, adapted_time, adapted_history, adapted_params = train_domain_adapted_qa(
        train_tokenized_da, tokenizer, mlm_model=mlm_model
    )
    
    # Free MLM model memory
    del mlm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ============================================================
    # Step 6: Evaluate Domain-Adapted Model
    # ============================================================
    print("\n" + "="*60)
    print("STEP 6: EVALUATING DOMAIN-ADAPTED MODEL")
    print("="*60)
    
    val_eval_da = val_raw.map(
        lambda x: preprocess_qa_validation(x, tokenizer),
        batched=True,
        remove_columns=val_raw.column_names,
        desc="Re-tokenizing val for adapted eval",
    )
    
    adapted_metrics = evaluate_qa_model(adapted_model, val_eval_da, val_raw, tokenizer)
    
    # Save domain-adapted QA model
    adapted_save_path = os.path.join(config.MODELS_DIR, "domain_adapted_bert_qa")
    print(f"[SAVE] Saving domain-adapted QA model to {adapted_save_path}")
    adapted_model.save_pretrained(adapted_save_path)
    tokenizer.save_pretrained(adapted_save_path)
    print(f"[SAVE] Domain-adapted QA model saved successfully!")
    
    # Free adapted model memory
    del adapted_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ============================================================
    # Step 7: Compare and Visualize
    # ============================================================
    print("\n" + "="*60)
    print("STEP 7: GENERATING COMPARISON RESULTS")
    print("="*60)
    
    # Compile results
    baseline_results = {
        **baseline_metrics,
        "training_time": baseline_time,
        "parameters": baseline_params,
    }
    
    adapted_results = {
        **adapted_metrics,
        "training_time": adapted_time,
        "total_time": mlm_time + adapted_time,
        "mlm_time": mlm_time,
        "parameters": adapted_params,
    }
    
    # Generate plots
    generate_comparison_plots(
        baseline_results, adapted_results,
        baseline_history, adapted_history,
        mlm_history=mlm_history
    )
    
    # Save results
    results = save_results(
        baseline_results, adapted_results,
        baseline_history, adapted_history,
        mlm_history=mlm_history
    )
    
    # ============================================================
    # Final Summary
    # ============================================================
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("  EXPERIMENT COMPLETE")
    print("="*70)
    print(f"  Total experiment time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Results saved to: {config.RESULTS_DIR}")
    print(f"  Plots saved to: {config.PLOTS_DIR}")
    print(f"  ")
    print(f"  Baseline BERT:        EM={baseline_results['exact_match']:.2f}%  F1={baseline_results['f1']:.2f}%")
    print(f"  Domain-Adapted BERT:  EM={adapted_results['exact_match']:.2f}%  F1={adapted_results['f1']:.2f}%")
    print(f"  Improvement:          EM={results['improvement']['em_diff']:+.2f}%  F1={results['improvement']['f1_diff']:+.2f}%")
    print("="*70)
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Experiment was interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
