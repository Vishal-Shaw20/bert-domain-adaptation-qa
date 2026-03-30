"""
Configuration file for NLP Assignment 2 - Domain Adaptation Gap
Roll Number: 2305990
Part 1: Domain Adaptation Gap (Roll Numbers Ending with 5 and 0)
"""

import os
import torch

# ============================================================
# Model Configuration
# ============================================================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128

# ============================================================
# QA Fine-tuning Configuration
# ============================================================
QA_TRAIN_SAMPLES = 200       # SQuAD training subset size (small for CPU)
QA_VAL_SAMPLES = 50          # SQuAD validation subset size
QA_BATCH_SIZE = 8
QA_LEARNING_RATE = 3e-5
QA_EPOCHS = 2
QA_WEIGHT_DECAY = 0.01
QA_WARMUP_RATIO = 0.1

# ============================================================
# MLM Pretraining Configuration (Domain Adaptation)
# ============================================================
MLM_TRAIN_SAMPLES = 300      # Domain corpus samples for MLM (small for CPU)
MLM_BATCH_SIZE = 8
MLM_LEARNING_RATE = 5e-5
MLM_EPOCHS = 2
MLM_MASK_PROB = 0.15
MLM_MAX_LENGTH = 128

# ============================================================
# Device & Reproducibility
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Create directories if they don't exist
for d in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)
