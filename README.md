# NLP Assignment 2: Advancing BERT - Domain Adaptation Gap

**Course:** CSE 37 - Natural Language Processing  
**Student Roll Number:** 2305990  
**Assignment Part:** Part 1 - Domain Adaptation Gap (Roll Numbers Ending with 0 and 5)  
**Instructor:** Dr. Sambit Praharaj, Assistant Professor (II), KIIT

## Objective

This project investigates the **Domain Adaptation Gap** in BERT-based models. The core hypothesis is that BERT's general-purpose pretraining may not capture the specialized vocabulary and relationships present in domain-specific text (e.g., medical, legal, or financial). By continuing BERT's pretraining on domain-specific corpora using **Masked Language Modeling (MLM)**, we aim to improve its performance on downstream Question Answering tasks.

## Project Structure

```
NLP Assignment/
├── config.py                    # Configuration and hyperparameters
├── data_utils.py                # Data loading and preprocessing
├── baseline_bert_qa.py          # Baseline BERT QA model
├── domain_adapted_bert_qa.py    # Domain-adapted BERT (MLM + QA)
├── evaluate_and_compare.py      # Evaluation metrics and visualizations
├── run_all.py                   # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── results/                     # Experiment results (JSON)
├── plots/                       # Visualization outputs
└── saved_models/                # Saved model checkpoints
```

## Methodology

### 1. Baseline Model
- **Model:** `bert-base-uncased` fine-tuned on SQuAD v1.1 subset
- **Task:** Extractive Question Answering
- **Metrics:** Exact Match (EM) and F1 Score

### 2. Domain-Adapted Model
- **Step 1 - MLM Pretraining:** Continue pretraining BERT on a medical/biomedical text corpus using Masked Language Modeling
- **Step 2 - QA Fine-tuning:** Fine-tune the domain-adapted BERT on SQuAD v1.1
- **Comparison:** Against the baseline model on same evaluation set

### 3. Extensions Implemented
- ✅ Continue pretraining using Masked Language Modeling (MLM)
- ✅ Comparison with baseline (BioBERT-style approach)

## Setup & Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete experiment
python run_all.py
```

## Results

Results are saved to `results/experiment_results.json` and visualizations to `plots/`.

| Model | Accuracy (EM) | F1 Score | Training Time | Parameters |
|-------|--------------|----------|---------------|------------|
| Baseline BERT | See results | See results | See results | ~110M |
| Domain-Adapted BERT | See results | See results | See results | ~110M |

## Key Files

- **`config.py`** — All hyperparameters (learning rates, batch sizes, subset sizes, etc.)
- **`data_utils.py`** — SQuAD data loading, QA preprocessing with sliding windows, medical corpus loading
- **`baseline_bert_qa.py`** — Standard BERT QA fine-tuning
- **`domain_adapted_bert_qa.py`** — MLM pretraining on domain text + QA fine-tuning
- **`evaluate_and_compare.py`** — EM/F1 computation, comparison plots, results export
- **`run_all.py`** — Orchestrates the full experiment pipeline

## Hardware

- Tested on CPU (compatible with CUDA GPUs for faster execution)
- Approximate runtime: ~30-60 minutes on CPU with default subset sizes
