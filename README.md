# Conversational AI with PyTorch Transformers

A comprehensive implementation of conversational AI using PyTorch with two different transformer architectures: **GPT-style (decoder-only)** and **full encoder-decoder**. Train your own ChatGPT-like model on English conversation data!

## 🚀 Features

- **Two AI Architectures**: GPT-style (like ChatGPT) and Encoder-Decoder
- **Complete Training Pipeline**: From raw text to trained conversational AI
- **Interactive Chat Interfaces**: Natural conversation with your trained models
- **English Conversation Dataset**: 3,690+ conversation pairs for training
- **Memory Optimized**: Runs on 8GB RAM systems
- **Comprehensive Logging**: Track training progress and model performance

## 🏗️ Architecture Overview

### GPT-Style (Decoder-Only) - Like ChatGPT
```
User Input → [GPT Model] → Response
```
- **Simpler architecture** - One model does everything
- **Autoregressive generation** - Predicts next tokens
- **Better for open-ended conversation**
- **Files**: `train_lm.py`, `lm_chat.py`

### Encoder-Decoder - Like Google Translate
```
User Input → [Encoder] → Context → [Decoder] → Response
```
- **Two-stage processing** - Understand then respond
- **Better for structured Q&A**
- **More complex but powerful**
- **Files**: `train_chat.py`, `chatbot.py`

## 📁 Project Structure

```
├── main.py                 # Core transformer architecture
├── tokenizer.py           # Text tokenization and vocabulary
├── lm_dataset.py          # Dataset for GPT-style training
├── chat_dataset.py        # Dataset for conversation training
├── train_lm.py            # Train GPT-style model
├── train_chat.py          # Train encoder-decoder model
├── lm_chat.py             # Chat with GPT-style model
├── chatbot.py             # Chat with encoder-decoder model
├── english_corpus.txt     # Conversation training data
```

## 🛠️ Setup

### 1. Environment Setup
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source ./venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy psutil
```

### 2. Data Preparation
Your `english_corpus.txt` should contain tab-separated conversation pairs:
```
Hi, how are you doing?	I'm fine. How about yourself?
I'm pretty good, thanks for asking.	No problem. So how have you been?
```

## 🚀 Quick Start

### Option 1: GPT-Style (Recommended)
```bash
# 1. Train GPT model (like ChatGPT)
python3 train_lm.py

# 2. Chat with your GPT model
python3 lm_chat.py
```

### Option 2: Encoder-Decoder
```bash
# 1. Train encoder-decoder model
python3 train_chat.py

# 2. Chat with your encoder-decoder model
python3 chatbot.py
```

## 📊 Training Details

### GPT Model Configuration
- **Architecture**: Decoder-only transformer
- **Parameters**: ~2M (lightweight)
- **Training Time**: ~15 minutes on CPU
- **Memory Usage**: ~200MB
- **Dataset Format**: Sequential text

### Encoder-Decoder Configuration
- **Architecture**: Full transformer (encoder + decoder)
- **Parameters**: ~10M (more powerful)
- **Training Time**: ~35 minutes on CPU
- **Memory Usage**: ~400MB
- **Dataset Format**: Question-answer pairs

## 🔧 Advanced Usage

### Training Monitoring
```bash
# Check training progress
tail -f training_log.txt

# Monitor model status
python3 chatbot.py status
python3 lm_chat.py status
```

### Testing Without Training
```bash
# Test GPT interface
python3 lm_chat.py test

# Test encoder-decoder interface
python3 chatbot.py test

# See demo conversations
python3 chatbot.py demo
```

### Custom Training Parameters
Edit the CONFIG section in training files:
```python
CONFIG = {
    'batch_size': 16,      # Reduce for less memory
    'epochs': 20,          # More epochs = better quality
    'd_model': 256,        # Model size
    'learning_rate': 3e-4, # Learning speed
}
```

## 🧠 Model Architecture Details

### Core Components (main.py)
- **Input Embedding**: Converts tokens to vectors
- **Positional Encoding**: Adds position information
- **Multi-Head Attention**: The "attention is all you need" mechanism
- **Feed Forward Networks**: Non-linear transformations
- **Layer Normalization**: Stabilizes training
- **Encoder/Decoder Blocks**: Complete transformer layers

### Attention Mechanisms
- **Self-Attention**: Words attend to other words in same sequence
- **Cross-Attention**: Decoder attends to encoder output
- **Causal Masking**: Prevents seeing future tokens during generation

## 📈 Performance Tips

### For Better Results
1. **More Training Data**: Add more conversation pairs to `english_corpus.txt`
2. **Longer Training**: Increase epochs in config
3. **Larger Model**: Increase `d_model` and `n_layers`
4. **Better Hardware**: Use GPU for faster training

### Memory Optimization
1. **Reduce Batch Size**: Lower `batch_size` in config
2. **Shorter Sequences**: Reduce `max_len`
3. **Smaller Model**: Decrease `d_model` and `n_layers`

## 🐛 Troubleshooting

### Common Issues

**"LR: 0.000000" (Learning rate zero)**
- Fixed in current version with improved scheduler

**"Model files not found"**
```bash
# Train the model first
python3 train_lm.py  # or train_chat.py
```

**Out of memory errors**
- Reduce batch_size in training config
- Use smaller model dimensions

**Poor conversation quality**
- Train for more epochs
- Add more conversation data
- Try the other architecture

## 📚 Learning Resources

### Understanding Transformers
- **"Attention Is All You Need"** - Original transformer paper
- **"The Illustrated Transformer"** - Visual explanation
- **OpenAI GPT papers** - Evolution of language models

### Code Structure
- `main.py` - Study the transformer implementation
- `train_*.py` - See how training loops work
- `*_chat.py` - Understand inference and generation

## 🎯 Use Cases

### Educational
- **Learn Transformers**: Hands-on implementation
- **Understand Attention**: See how attention mechanisms work
- **Compare Architectures**: GPT vs Encoder-Decoder

### Practical
- **Custom Chatbots**: Train on your own conversation data
- **Text Generation**: Creative writing assistance
- **Q&A Systems**: Domain-specific question answering

## 🔮 Future Enhancements

- **Multi-language Support**: Train on different languages
- **Larger Models**: Scale up for better performance
- **Fine-tuning**: Adapt pre-trained models
- **Web Interface**: Browser-based chat interface
- **Voice Integration**: Speech-to-text and text-to-speech

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 👨‍💻 Author

**Amanuel Ayalew**

Built with ❤️ using PyTorch and the power of attention mechanisms.

---

*"Attention is all you need" - and now you have it!* 🚀
