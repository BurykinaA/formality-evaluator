import os
import csv
import time
import sys


from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score



def evaluate_binary_classification(predictions, labels, threshold=0.5):
    """
    Evaluate binary classification predictions
    
    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    if len(predictions.shape) > 1 and predictions.shape[1] == 2:
        probs = np.exp(predictions[:, 1]) / np.sum(np.exp(predictions), axis=1)
    else:
        probs = 1 / (1 + np.exp(-predictions))
    
    binary_preds = (probs >= threshold).astype(int)
    
    accuracy = accuracy_score(labels, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_preds, average='binary'
    )
    
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = None
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return metrics

def evaluate_similarity(embeddings1, embeddings2, labels, threshold=0.5):
    """
    Evaluate similarity between pairs of embeddings
    
    Args:
        embeddings1: Embeddings of first texts
        embeddings2: Embeddings of second texts
        labels: Ground truth labels (1 for similar, 0 for dissimilar)
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    norm_emb1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm_emb2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    similarities = np.sum(norm_emb1 * norm_emb2, axis=1)
    
    binary_preds = (similarities >= threshold).astype(int)
    
    metrics = evaluate_binary_classification(similarities, labels, threshold)
    
    return metrics



