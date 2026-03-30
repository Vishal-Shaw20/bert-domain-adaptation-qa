"""
Baseline BERT Question Answering Model
Fine-tunes bert-base-uncased directly on SQuAD v1.1 subset.
This serves as the baseline for comparison with domain-adapted models.
"""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

import config
from data_utils import set_seed


def train_baseline_qa(train_dataset, tokenizer, epochs=config.QA_EPOCHS,
                       batch_size=config.QA_BATCH_SIZE,
                       learning_rate=config.QA_LEARNING_RATE):
    """
    Train baseline BERT for Question Answering on SQuAD.
    
    Args:
        train_dataset: Preprocessed training dataset
        tokenizer: BERT tokenizer
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    
    Returns:
        model: Trained model
        training_time: Total training time in seconds
        training_history: Dict with loss history
    """
    set_seed()
    print("\n" + "="*60)
    print("TRAINING BASELINE BERT FOR QUESTION ANSWERING")
    print("="*60)
    
    # Load pretrained BERT for QA
    model = BertForQuestionAnswering.from_pretrained(config.MODEL_NAME)
    model.to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,}")
    
    # Prepare data
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 
                                                      "token_type_ids", "start_positions", 
                                                      "end_positions"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, 
                      weight_decay=config.QA_WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.QA_WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    # Training loop
    training_history = {"epoch_losses": [], "step_losses": []}
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
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
            training_history["step_losses"].append(loss.item())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        training_history["epoch_losses"].append(avg_epoch_loss)
        print(f"[TRAIN] Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")
    
    training_time = time.time() - start_time
    
    print(f"\n[TRAIN] Baseline training completed in {training_time:.1f} seconds")
    print(f"[TRAIN] Final loss: {training_history['epoch_losses'][-1]:.4f}")
    
    return model, training_time, training_history, total_params
