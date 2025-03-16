import os
import argparse
import logging
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from utils.get_embeddings import calculate_embedding
from utils.test_utils import evaluate_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_similarity_data(data_path):
    """Load data for similarity evaluation"""
    text1_list = []
    text2_list = []
    labels = []
    
    with open(data_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                text1_list.append('Response: '+parts[0])
                text2_list.append('Query: Is this text written in a formal tone?')
                labels.append(int(parts[1]))
    
    return text1_list, text2_list, np.array(labels)

def evaluate_model_similarity(model, tokenizer, data_path, output_dir, max_length=128, batch_size=32):
    """Evaluate model on similarity task"""
    logger.info(f"Evaluating similarity on {data_path}")
    
    # Load data
    text1_list, text2_list, labels = load_similarity_data(data_path)
    
    # Calculate embeddings
    logger.info("Calculating embeddings for first texts...")
    embeddings1 = calculate_embedding(text1_list, model, tokenizer, device=device, max_length=max_length, batch_size=batch_size, verbose=True)
    
    logger.info("Calculating embeddings for second texts...")
    embeddings2 = calculate_embedding(text2_list, model, tokenizer, device=device, max_length=max_length, batch_size=batch_size, verbose=True)
    
    logger.info("Evaluating similarity...")
    metrics = evaluate_similarity(embeddings1, embeddings2, labels)
    
    logger.info(f"Similarity metrics: {metrics}")
    
    with open(os.path.join(output_dir, "similarity_results.txt"), "w") as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1: {metrics['f1']:.4f}\n")
        if metrics['auc'] is not None:
            f.write(f"AUC: {metrics['auc']:.4f}\n")
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to the similarity data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the pre-trained model")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding calculation")
    
    args = parser.parse_args()

    model_name_clean = os.path.basename(args.model_name)
    dataset_name_clean = os.path.splitext(os.path.basename(args.data_path))[0]
    output_dir = os.path.join(args.output_dir, f"{dataset_name_clean}", f"{model_name_clean}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    metrics = evaluate_model_similarity(
        model,
        tokenizer,
        args.data_path,
        output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
    
