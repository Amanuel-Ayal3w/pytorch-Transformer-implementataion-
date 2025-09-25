import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tokenizer import SimpleTokenizer
from lm_dataset import LanguageModelingDataset
from main import inputEmbedding, PositionalEncoding

# Simple GPT-style decoder-only Transformer for language modeling
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_len=32, dropout=0.1):
        super().__init__()
        self.embedding = inputEmbedding(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)
        emb = self.pos_encoding(emb)
        # Generate causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        out = self.decoder(emb, emb, tgt_mask=mask)
        logits = self.lm_head(out)
        return logits

# Config
DATA_PATH = 'english_corpus.txt' 
BATCH_SIZE = 32
EPOCHS = 10
# block_size
MAX_LEN = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'language_model.pt'
TOKENIZER_PATH = 'tokenizer.pt'

def train_gpt_model():
    """Main training function for GPT language model."""
    # 1. Prepare Tokenizer and Dataset
    print('Fitting tokenizer...')
    with open(DATA_PATH, encoding='utf-8') as f:
        text = f.read()
    tokenizer = SimpleTokenizer()
    tokenizer.fit([text])
    dataset = LanguageModelingDataset(DATA_PATH, tokenizer, seq_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Build Model
    vocab_size = tokenizer.vocab_size()
    model = GPTLanguageModel(vocab_size, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_len=MAX_LEN, dropout=0.1)
    model = model.to(DEVICE)

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. Training Loop
    print('Starting training...')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for input_seq, target_seq in dataloader:
            input_seq, target_seq = input_seq.to(DEVICE), target_seq.to(DEVICE)
            logits = model(input_seq)
            loss = criterion(logits.view(-1, logits.size(-1)), target_seq.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), MODEL_PATH)
        torch.save(tokenizer, TOKENIZER_PATH)

    print('Training complete!')

if __name__ == "__main__":
    train_gpt_model()
