import torch

class BinaryClassificationDataset(torch.utils.data.Dataset):
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
    # Tokenize texts
    encodings = tokenizer(texts, padding="max_length", truncation=True, 
                          return_tensors="pt", max_length=max_length)
    
    # Create dataset
    dataset = BinaryClassificationDataset(encodings, labels)
    
    return dataset
