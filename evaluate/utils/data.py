import os
import csv
import json
import random
from copy import deepcopy
from collections import Counter

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import normalize

from .get_embeddings import calculate_embedding


class SingleSetenceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, seq_labels, ner_labels=None):
        self.encodings = encodings
        self.seq_labels = seq_labels
        self.ner_labels = ner_labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['seq_labels'] = torch.tensor(self.seq_labels[idx])
        item['ner_labels'] = torch.tensor(self.ner_labels[idx]) if self.ner_labels else []
        return item

    def __len__(self):
        return len(self.seq_labels)


class BinaryClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for binary classification tasks
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def prepare_binary_classification_data(texts, labels, tokenizer, max_length=128):
    """
    Prepare data for binary classification
    
    Args:
        texts: List of text strings
        labels: List of binary labels (0 or 1)
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        
    Returns:
        BinaryClassificationDataset
    """
    # Tokenize texts
    encodings = tokenizer(texts, padding="max_length", truncation=True, 
                          return_tensors="pt", max_length=max_length)
    
    # Create dataset
    dataset = BinaryClassificationDataset(encodings, labels)
    
    return dataset

def create_similarity_datasets(text_pairs, labels, tokenizer, max_length=128, val_split=0.1):
    """
    Create train and validation datasets for similarity tasks
    
    Args:
        text_pairs: List of tuples (text1, text2)
        labels: List of binary labels (0 or 1)
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        val_split: Fraction of data to use for validation
        
    Returns:
        train_dataset, val_dataset
    """
    # Combine text pairs with special tokens
    combined_texts = [f"{pair[0]} [SEP] {pair[1]}" for pair in text_pairs]
    
    # Convert labels to numpy array
    labels = np.array(labels)
    
    # Split data into train and validation
    indices = np.random.permutation(len(combined_texts))
    val_size = int(len(combined_texts) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Create train and validation datasets
    train_texts = [combined_texts[i] for i in train_indices]
    train_labels = labels[train_indices]
    val_texts = [combined_texts[i] for i in val_indices]
    val_labels = labels[val_indices]
    
    # Create datasets
    train_dataset = prepare_binary_classification_data(train_texts, train_labels, tokenizer, max_length)
    val_dataset = prepare_binary_classification_data(val_texts, val_labels, tokenizer, max_length)
    
    return train_dataset, val_dataset

