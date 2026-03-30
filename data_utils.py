"""
Data Utilities for NLP Assignment 2
Handles loading and preprocessing of SQuAD and domain-specific datasets.
"""

import random
import collections
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizerFast
import config


def set_seed(seed=config.SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_squad_data(train_samples=config.QA_TRAIN_SAMPLES,
                    val_samples=config.QA_VAL_SAMPLES):
    """
    Load SQuAD v1.1 dataset subset.
    
    Returns:
        train_dataset: Training split (subset)
        val_dataset: Validation split (subset)
    """
    print(f"[DATA] Loading SQuAD v1.1 dataset...")
    dataset = load_dataset("squad", trust_remote_code=True)
    
    # Take subsets
    train_data = dataset["train"].shuffle(seed=config.SEED).select(range(min(train_samples, len(dataset["train"]))))
    val_data = dataset["validation"].shuffle(seed=config.SEED).select(range(min(val_samples, len(dataset["validation"]))))
    
    print(f"[DATA] Training samples: {len(train_data)}")
    print(f"[DATA] Validation samples: {len(val_data)}")
    
    return train_data, val_data


def preprocess_qa_training(examples, tokenizer, max_length=config.MAX_LENGTH, 
                           doc_stride=config.DOC_STRIDE):
    """
    Preprocess SQuAD examples for QA training.
    Handles context splitting with sliding window for long documents.
    """
    questions = [q.strip() for q in examples["question"]]
    
    tokenized = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # Find start and end of context in tokens
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            # Check if answer is within this span
            if not (offsets[token_start_index][0] <= start_char and 
                    offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # Find token positions of answer
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)
    
    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    
    return tokenized


def preprocess_qa_validation(examples, tokenizer, max_length=config.MAX_LENGTH,
                              doc_stride=config.DOC_STRIDE):
    """
    Preprocess SQuAD examples for QA validation/evaluation.
    Keeps example IDs and offset mappings for answer extraction.
    """
    questions = [q.strip() for q in examples["question"]]
    
    tokenized = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    
    example_ids = []
    for i in range(len(tokenized["input_ids"])):
        sample_index = sample_mapping[i]
        example_ids.append(examples["id"][sample_index])
        
        # Set offset mapping to None for non-context tokens
        sequence_ids = tokenized.sequence_ids(i)
        offset = tokenized["offset_mapping"][i]
        tokenized["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None
            for k, o in enumerate(offset)
        ]
    
    tokenized["example_id"] = example_ids
    return tokenized


def load_medical_corpus(num_samples=config.MLM_TRAIN_SAMPLES):
    """
    Load a medical/biomedical text corpus for domain-specific MLM pretraining.
    Uses PubMed QA dataset abstracts as biomedical text source.
    """
    print(f"[DATA] Loading medical corpus for MLM pretraining...")
    
    try:
        # Try loading PubMed QA dataset
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)
        texts = []
        for item in dataset:
            # Combine context paragraphs
            if "context" in item and "contexts" in item["context"]:
                text = " ".join(item["context"]["contexts"])
                texts.append(text)
            elif "long_answer" in item:
                texts.append(item["long_answer"])
        
        if len(texts) > num_samples:
            random.seed(config.SEED)
            texts = random.sample(texts, num_samples)
        
        print(f"[DATA] Loaded {len(texts)} medical text samples from PubMed QA")
        return texts
        
    except Exception as e:
        print(f"[DATA] PubMed QA not available ({e}), trying alternative...")
        
        try:
            # Alternative: use medical_questions_pairs
            dataset = load_dataset("medical_questions_pairs", split="train", trust_remote_code=True)
            texts = [item["question_1"] + " " + item["question_2"] for item in dataset]
            
            if len(texts) > num_samples:
                random.seed(config.SEED)
                texts = random.sample(texts, num_samples)
            
            print(f"[DATA] Loaded {len(texts)} medical text samples from medical_questions_pairs")
            return texts
            
        except Exception as e2:
            print(f"[DATA] Falling back to synthetic medical corpus ({e2})")
            return _generate_synthetic_medical_corpus(num_samples)


def _generate_synthetic_medical_corpus(num_samples):
    """
    Generate synthetic medical text for MLM pretraining demonstration.
    This is a fallback when external datasets are unavailable.
    """
    medical_templates = [
        "The patient presented with symptoms of {condition} including {symptom1} and {symptom2}. "
        "Treatment with {drug} was initiated at a dosage of {dose}. Follow-up showed improvement.",
        
        "Clinical trials for {drug} demonstrated efficacy in treating {condition}. "
        "The study enrolled {n} participants across multiple sites. "
        "Primary endpoints included reduction in {symptom1} and {symptom2}.",
        
        "Pathological examination revealed {finding} consistent with {condition}. "
        "Immunohistochemical staining was positive for {marker}. "
        "The patient was referred for {treatment} therapy.",
        
        "A retrospective analysis of {n} patients with {condition} showed that "
        "combination therapy with {drug} and {drug2} resulted in superior outcomes "
        "compared to monotherapy. Adverse events included {symptom1}.",
        
        "Magnetic resonance imaging of the {organ} demonstrated {finding}. "
        "Differential diagnosis includes {condition} and {condition2}. "
        "Laboratory results showed elevated {biomarker} levels.",
        
        "The mechanism of action of {drug} involves inhibition of {pathway}, "
        "leading to decreased {process}. This has implications for treatment of "
        "{condition} and related disorders.",
        
        "Epidemiological data suggest that {condition} affects approximately "
        "{prevalence} of the population. Risk factors include {risk1} and {risk2}. "
        "Early detection through {screening} is recommended.",
        
        "Post-operative recovery following {procedure} for {condition} typically "
        "requires {duration}. Complications may include {complication1} and "
        "{complication2}. Physical therapy is recommended."
    ]
    
    conditions = ["type 2 diabetes", "hypertension", "chronic kidney disease", 
                  "coronary artery disease", "rheumatoid arthritis", "asthma",
                  "hepatitis B", "pulmonary fibrosis", "multiple sclerosis",
                  "Parkinson's disease", "Alzheimer's disease", "pneumonia",
                  "sepsis", "acute myocardial infarction", "stroke"]
    
    symptoms = ["fatigue", "dyspnea", "chest pain", "elevated blood pressure",
                "peripheral edema", "joint stiffness", "cognitive decline",
                "muscle weakness", "nausea", "headache", "tachycardia",
                "hypotension", "fever", "chronic cough", "weight loss"]
    
    drugs = ["metformin", "lisinopril", "atorvastatin", "aspirin", 
             "omeprazole", "amlodipine", "prednisone", "rituximab",
             "insulin glargine", "warfarin", "clopidogrel", "metoprolol"]
    
    organs = ["brain", "liver", "kidney", "heart", "lung", "pancreas", "spleen"]
    
    findings = ["calcification", "mass lesion", "edema", "fibrosis",
                "necrosis", "hyperplasia", "atrophy", "inflammation"]
    
    markers = ["CD20", "Ki-67", "HER2", "PD-L1", "p53", "EGFR"]
    
    biomarkers = ["troponin", "creatinine", "BNP", "CRP", "HbA1c", "LDL cholesterol"]
    
    texts = []
    random.seed(config.SEED)
    
    for i in range(num_samples):
        template = random.choice(medical_templates)
        text = template.format(
            condition=random.choice(conditions),
            condition2=random.choice(conditions),
            symptom1=random.choice(symptoms),
            symptom2=random.choice(symptoms),
            drug=random.choice(drugs),
            drug2=random.choice(drugs),
            dose=f"{random.randint(1, 500)}mg",
            n=random.randint(50, 5000),
            finding=random.choice(findings),
            marker=random.choice(markers),
            treatment=random.choice(["chemotherapy", "radiation", "immunotherapy", "surgical"]),
            organ=random.choice(organs),
            biomarker=random.choice(biomarkers),
            pathway=random.choice(["JAK-STAT", "PI3K/AKT", "NF-κB", "MAPK", "mTOR"]),
            process=random.choice(["inflammation", "cell proliferation", "apoptosis", "angiogenesis"]),
            prevalence=f"{random.uniform(0.1, 15.0):.1f}%",
            risk1=random.choice(["smoking", "obesity", "sedentary lifestyle", "genetic predisposition"]),
            risk2=random.choice(["advanced age", "family history", "high cholesterol", "diabetes"]),
            screening=random.choice(["blood tests", "imaging", "biopsy", "genetic testing"]),
            procedure=random.choice(["CABG", "hip replacement", "appendectomy", "cholecystectomy"]),
            duration=random.choice(["2-4 weeks", "6-8 weeks", "3-6 months", "1-2 weeks"]),
            complication1=random.choice(["infection", "hemorrhage", "thrombosis", "wound dehiscence"]),
            complication2=random.choice(["pain", "restricted mobility", "scarring", "nerve damage"]),
        )
        texts.append(text)
    
    print(f"[DATA] Generated {len(texts)} synthetic medical text samples")
    return texts


class MLMDataset:
    """Dataset class for Masked Language Modeling."""
    
    def __init__(self, texts, tokenizer, max_length=config.MLM_MAX_LENGTH, 
                 mask_prob=config.MLM_MASK_PROB):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        
        print(f"[DATA] Tokenizing {len(texts)} texts for MLM...")
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        import torch
        
        input_ids = self.encodings["input_ids"][idx].clone()
        attention_mask = self.encodings["attention_mask"][idx].clone()
        
        # Create labels (copy of original input_ids)
        labels = input_ids.clone()
        
        # Create mask for MLM
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        
        # Don't mask special tokens
        special_tokens_mask = [
            1 if token in [self.tokenizer.cls_token_id, 
                          self.tokenizer.sep_token_id, 
                          self.tokenizer.pad_token_id] else 0
            for token in input_ids.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        
        # Don't mask padding
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% [MASK], 10% random, 10% original
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8)
        ).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), input_ids.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
