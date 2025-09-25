import torch
from torch.utils.data import Dataset
from tokenizer import SimpleTokenizer

class LanguageModelingDataset(Dataset):
    def __init__(self, text_path, tokenizer, seq_len=32):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        with open(text_path, encoding='utf-8') as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        self.samples = []
        for i in range(0, len(tokens) - seq_len):
            input_seq = tokens[i:i+seq_len]
            target_seq = tokens[i+1:i+seq_len+1]
            self.samples.append((input_seq, target_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)

