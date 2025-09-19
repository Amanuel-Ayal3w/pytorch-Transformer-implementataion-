# GPT-Style Conversational AI

Train your own ChatGPT-like model using PyTorch transformers. Simple decoder-only architecture that learns from conversation data.

## What It Does

- **Trains a GPT-style model** on English conversation data
- **Learns conversation patterns** from 3,690+ question-answer pairs
- **Generates responses** like ChatGPT using attention mechanisms
- **Runs locally** on your machine (8GB RAM friendly)

## Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy
```

### 2. Train Your Model
```bash
python3 train_lm.py
```
*Takes ~15 minutes on CPU*

### 3. Chat with Your Model
```bash
python3 lm_chat.py
```

## Example Chat
```
👤 You: Hi, how are you doing?
🤖 GPT: I'm doing well, thank you! How about yourself?

👤 You: What school do you go to?
🤖 GPT: I go to PCC.
```

## Files
- `train_lm.py` - Train the GPT model
- `lm_chat.py` - Chat interface
- `main.py` - Transformer architecture
- `tokenizer.py` - Text processing
- `lm_dataset.py` - Data preparation
- `english_corpus.txt` - Training data

## Troubleshooting

**"Model files not found"** → Run `python3 train_lm.py` first

**Out of memory** → Reduce `BATCH_SIZE` in `train_lm.py`

**Poor responses** → Train longer (increase `EPOCHS`)

---

Built with PyTorch • Decoder-only architecture • "Attention is all you need"