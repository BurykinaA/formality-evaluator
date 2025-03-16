from tqdm import tqdm
import logging
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from transformers import AutoConfig, AutoTokenizer, AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from utils.data import prepare_binary_classification_data
from utils.model import GeneralModelForSequenceClassification
from utils.test_utils import evaluate_binary_classification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, eval_dataloader, device):
    """Evaluate model on a dataset"""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            
            logits = outputs[1] if isinstance(outputs, tuple) else outputs['logits']
            labels = batch['labels']
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    metrics = evaluate_binary_classification(all_logits, all_labels)
    
    return metrics, np.mean(all_logits)

def train(args, model, train_dataset, val_dataset, test_dataset):
    """Train model on a dataset"""
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)
    
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    model.to(args.device)
    
    logger.info("Starting training...")
    best_val_metric = 0
    best_model = None
    early_stop_count = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs[0] if isinstance(outputs, tuple) else outputs['loss']
            
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        val_metrics, _ = evaluate(model, val_dataloader, args.device)
        val_metric = val_metrics['accuracy']
        
        logger.info(f"Epoch {epoch+1}: Train loss = {epoch_loss/len(train_dataloader):.4f}, "
                   f"Val accuracy = {val_metric:.4f}")
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_model = model.state_dict().copy()
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        if early_stop_count >= args.patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    model.load_state_dict(best_model)
    
    test_metrics, _ = evaluate(model, test_dataloader, args.device)
    
    logger.info(f"Test metrics: {test_metrics}")
    
    model_name_clean = os.path.basename(args.model_name)
    dataset_name_clean = os.path.splitext(os.path.basename(args.data_dir))[0]
    output_dir = os.path.join(args.output_dir, f"{dataset_name_clean}", f"{model_name_clean}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "finetune_results.txt"), "w") as f:
        f.write(f"Test accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Test precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Test recall: {test_metrics['recall']:.4f}\n")
        f.write(f"Test F1: {test_metrics['f1']:.4f}\n")
        if test_metrics['auc'] is not None:
            f.write(f"Test AUC: {test_metrics['auc']:.4f}\n")
    
    if args.save_model:
        logger.info(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
    
    return test_metrics

def load_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    texts, labels = [], []
    
    with open(args.data_dir, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                texts.append(parts[0])
                labels.append(int(parts[1]))
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.4, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    train_dataset = prepare_binary_classification_data(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = prepare_binary_classification_data(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = prepare_binary_classification_data(test_texts, test_labels, tokenizer, args.max_length)
    
    return train_dataset, val_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model and results")
    
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Name or path of the pre-trained model")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "average"], help="Pooling method")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_model", action="store_true", help="Whether to save the model")
    
    args = parser.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    set_seed(args.seed)
    
    train_dataset, val_dataset, test_dataset = load_data(args)
    
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = 2  # Binary classification
    
    model = GeneralModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        model_name=args.model_name,
        weights=None,
        pooling=args.pooling
    )
    
    test_metrics = train(args, model, train_dataset, val_dataset, test_dataset)
    
    logger.info(f"Training completed. Test accuracy: {test_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()