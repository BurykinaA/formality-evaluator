import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def evaluate_binary_classification(predictions, labels, threshold=0.5):
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

    norm_emb1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm_emb2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    similarities = np.sum(norm_emb1 * norm_emb2, axis=1)
    
    metrics = evaluate_binary_classification((similarities + 1) / 2, labels, threshold = np.median(similarities))
    
    return metrics



