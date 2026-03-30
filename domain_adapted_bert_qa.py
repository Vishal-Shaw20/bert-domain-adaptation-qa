"""
Domain-Adapted BERT Question Answering Model
Part 1: Domain Adaptation Gap

This module implements:
1. Masked Language Model (MLM) pretraining on domain-specific (medical) corpus
2. Fine-tuning the domain-adapted BERT on SQuAD for QA

The hypothesis is that domain-specific pretraining improves BERT's understanding
of specialized vocabulary and relationships, leading to better QA performance.
"""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    BertForQuestionAnswering,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

import config
from data_utils import set_seed, load_medical_corpus, MLMDataset


def mlm_pretrain(tokenizer, epochs=config.MLM_EPOCHS,
                  batch_size=config.MLM_BATCH_SIZE,
                  learning_rate=config.MLM_LEARNING_RATE):
    """
    Continue pretraining BERT with Masked Language Modeling on medical corpus.
    
    This implements the domain adaptation step where we expose BERT to
    domain-specific text before fine-tuning on the downstream QA task.
    
    Args:
        tokenizer: BERT tokenizer
        epochs: Number of MLM pretraining epochs
        batch_size: Batch size for MLM training
        learning_rate: Learning rate for MLM
    
    Returns:
        model: Domain-pretrained BERT model (BertForMaskedLM)
        mlm_time: Time taken for MLM pretraining
        mlm_history: Training history
    """
    set_seed()
    print("\n" + "="*60)
    print("DOMAIN-SPECIFIC MLM PRETRAINING (Medical Corpus)")
    print("="*60)
    
    # Load medical corpus
    medical_texts = load_medical_corpus()
    
    # Create MLM dataset
    mlm_dataset = MLMDataset(medical_texts, tokenizer)
    mlm_loader = DataLoader(mlm_dataset, batch_size=batch_size, shuffle=True)
    
    # Load BERT for MLM
    model = BertForMaskedLM.from_pretrained(config.MODEL_NAME)
    model.to(config.DEVICE)
    
    print(f"[MLM] Model loaded on {config.DEVICE}")
    print(f"[MLM] Training samples: {len(mlm_dataset)}")
    print(f"[MLM] Epochs: {epochs}, Batch size: {batch_size}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(mlm_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Training loop
    mlm_history = {"epoch_losses": [], "step_losses": []}
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(mlm_loader, desc=f"MLM Epoch {epoch+1}/{epochs}",
                          unit="batch")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            mlm_history["step_losses"].append(loss.item())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_epoch_loss = epoch_loss / len(mlm_loader)
        mlm_history["epoch_losses"].append(avg_epoch_loss)
        print(f"[MLM] Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")
    
    mlm_time = time.time() - start_time
    print(f"\n[MLM] Domain pretraining completed in {mlm_time:.1f} seconds")
    
    return model, mlm_time, mlm_history


def train_domain_adapted_qa(train_dataset, tokenizer, mlm_model=None,
                             epochs=config.QA_EPOCHS,
                             batch_size=config.QA_BATCH_SIZE,
                             learning_rate=config.QA_LEARNING_RATE):
    """
    Fine-tune domain-adapted BERT for Question Answering.
    
    Uses the BERT encoder weights from MLM pretraining and adds a QA head.
    
    Args:
        train_dataset: Preprocessed QA training dataset
        tokenizer: BERT tokenizer
        mlm_model: Domain-pretrained BertForMaskedLM model
        epochs: QA fine-tuning epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        model: Trained QA model
        qa_time: QA fine-tuning time
        qa_history: Training history
        total_params: Total model parameters
    """
    set_seed()
    print("\n" + "="*60)
    print("FINE-TUNING DOMAIN-ADAPTED BERT FOR QA")
    print("="*60)
    
    # Initialize QA model
    qa_model = BertForQuestionAnswering.from_pretrained(config.MODEL_NAME)
    
    # Transfer domain-adapted weights from MLM model to QA model
    if mlm_model is not None:
        print("[MODEL] Transferring domain-adapted BERT encoder weights...")
        
        # Get the BERT encoder state dict from MLM model
        mlm_bert_state = mlm_model.bert.state_dict()
        
        # Load into QA model's BERT encoder
        qa_model.bert.load_state_dict(mlm_bert_state)
        print("[MODEL] Domain-adapted weights loaded successfully!")
    
    qa_model.to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in qa_model.parameters())
    trainable_params = sum(p.numel() for p in qa_model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,}")
    
    # Prepare data
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask",
                                                      "token_type_ids", "start_positions",
                                                      "end_positions"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and scheduler
    optimizer = AdamW(qa_model.parameters(), lr=learning_rate,
                      weight_decay=config.QA_WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.QA_WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    # Training loop
    qa_history = {"epoch_losses": [], "step_losses": []}
    start_time = time.time()
    
    qa_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"QA Epoch {epoch+1}/{epochs}",
                          unit="batch")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            
            outputs = qa_model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qa_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            qa_history["step_losses"].append(loss.item())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        qa_history["epoch_losses"].append(avg_epoch_loss)
        print(f"[QA] Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")
    
    qa_time = time.time() - start_time
    print(f"\n[QA] Domain-adapted QA training completed in {qa_time:.1f} seconds")
    
    return qa_model, qa_time, qa_history, total_params
