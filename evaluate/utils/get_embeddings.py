import os
import argparse
import json
import time
import pickle
from tqdm import tqdm, trange
import numpy as np
from sklearn.preprocessing import normalize

from transformers import AutoModel, AutoTokenizer

import torch
from torch.utils.data import DataLoader, SequentialSampler



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def calculate_embedding(texts, model, tokenizer, device='cuda', batch_size=32, max_length=128, verbose=False):
    """
    Calculate embeddings for text using a pre-trained model.
    
    Args:
        texts: List of text strings
        model: Pre-trained model
        tokenizer: Tokenizer for the model
        device: Device to run the model on
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        verbose: Whether to show progress bar
    
    Returns:
        numpy array of embeddings
    """

    model.to(device)
    
    encoding = tokenizer(texts, padding="max_length", truncation=True, 
                         return_tensors="pt", max_length=max_length)
    
    emb_dataset = TextDataset(encoding)
    emb_sampler = SequentialSampler(emb_dataset)
    emb_dataloader = DataLoader(emb_dataset, batch_size=batch_size, sampler=emb_sampler)
    
    emb_iterator = tqdm(emb_dataloader, desc="Calculating embeddings") if verbose else emb_dataloader

    all_embeddings = []

    with torch.no_grad():
        model.eval()
        for inputs in emb_iterator:
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # print('outputs', outputs)
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                #embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)


            # print(embeddings)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size of bert embedding calculation", default=0)
    parser.add_argument("--n_gpu", type=int, help="Batch size of bert embedding calculation", default=1)
    parser.add_argument("--text_dir", type=str, help="path to the original text file", default="/mnt/efs/ToD-BERT/data_tod/all_sent_cleaned.txt")
    parser.add_argument("--embedding_dir", type=str, help="path to the embedding file", default="data/embedding.npy")
    parser.add_argument("--model_dir", type=str, help="path to the model used for calculating embedding", default="TODBERT/TOD-BERT-JNT-V1")
    parser.add_argument("--average_embedding", action="store_true", help="", default=False)
    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()
    batch_size = args.batch_size if args.batch_size else 10

    root_path = '/'.join(args.index_dir.split("/")[:-1])
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    with open(args.text_dir, "r") as f:
        texts = f.readlines()
    texts = [t.strip("\n") for t in texts]
    texts = [t.split('\t')[-1] for t in texts]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModel.from_pretrained(args.model_dir)

    task_type = "average_embedding" if args.average_embedding else "cls_embedding"
    ori_embedding = calculate_embedding(texts, model, tokenizer, task_type=task_type, verbose=True)


        


    

