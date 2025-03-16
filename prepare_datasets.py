import os
import random
from datasets import load_dataset
import re

def clean_text(text):
    """Clean text by removing extra whitespace and newlines."""
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_length=200):
    """Split text into chunks of approximately max_length characters."""
    if not text or len(text) <= max_length:
        return [text] if text else []
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            if len(sentence) > max_length:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_length:
                        current_chunk += " " + word if current_chunk else word
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = word
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_reddit_dataset(sample_size=50000):
    """Process the Reddit dataset and return the samples."""
    print("Loading Reddit dataset...")
    dataset = load_dataset('SocialGrep/the-reddit-dataset-dataset', 'comments', split='train[:100000]')
    
    print(f"Reddit dataset size: {len(dataset)}")
    print(f"Reddit dataset columns: {dataset.column_names}")
    
    texts = []
    for item in dataset:
        if 'body' in item and item['body']:
            clean = clean_text(item['body'])
            if clean and len(clean) > 20:  # Filter out very short comments
                chunks = chunk_text(clean[:200])
                texts.extend(chunks)
                
                if len(texts) >= sample_size:
                    break
    
    random.shuffle(texts)
    texts = [f"{text}\t0\n" for text in texts if text][:sample_size]  # 0 - informal text
    
    print(f"Reddit dataset processed: {len(texts)} samples")
    return texts

def process_enron_dataset(sample_size=50000):
    """Process the Enron email dataset and return the samples."""
    print("Loading Enron email dataset...")
    dataset = load_dataset("LLM-PBE/enron-email", split="train[:100000]")
    
    print(f"Enron dataset size: {len(dataset)}")
    print(f"Enron dataset columns: {dataset.column_names}")
    
    texts = []
    for item in dataset:
        content = item.get('text', '')
        if content:
            clean = clean_text(content)
            if clean and len(clean) > 20:  # Filter out very short emails
                chunks = chunk_text(clean[:200])
                texts.extend(chunks)
                
                if len(texts) >= sample_size:
                    break
    
    random.shuffle(texts)
    texts = [f"{text}\t1\n" for text in texts if text][:sample_size]  # 1 - formal text
    
    print(f"Enron dataset processed: {len(texts)} samples")
    return texts

def create_combined_dataset(output_file):
    """Create a combined dataset from Reddit and Enron data."""
    reddit_samples = process_reddit_dataset()
    enron_samples = process_enron_dataset()
    
    combined_lines = reddit_samples + enron_samples
    random.shuffle(combined_lines)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(combined_lines)
    
    print(f"Combined dataset saved to {output_file} with {len(combined_lines)} lines")

if __name__ == "__main__":
    output_file = "data/reddit_enron_combined.txt"
    create_combined_dataset(output_file) 