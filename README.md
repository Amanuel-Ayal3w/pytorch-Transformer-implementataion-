# GPT-Style Conversational AI with PyTorch

A clean implementation of GPT-style conversational AI using PyTorch transformers. Train your own ChatGPT-like model on English conversation data with a simple, decoder-only architecture!

## 🚀 Features

- **GPT-Style Architecture**: Decoder-only transformer (like ChatGPT)
- **Complete Training Pipeline**: From raw text to trained conversational AI
- **Interactive Chat Interface**: Natural conversation with your trained model
- **English Conversation Dataset**: 3,690+ conversation pairs for training
- **Memory Optimized**: Runs efficiently on 8GB RAM systems
- **Simple & Clean**: Focus on what works best for conversation

## 🏗️ Architecture Overview

### GPT-Style (Decoder-Only) - Like ChatGPT
```
User Input → [GPT Model] → Response
```
- **Simple architecture** - One model does everything
- **Autoregressive generation** - Predicts next tokens naturally
- **Proven approach** - Same as ChatGPT, GPT-4, Claude
- **Files**: `train_lm.py`, `lm_chat.py`

## 📁 Project Structure

```
├── main.py                 # Core transformer architecture components
├── tokenizer.py           # Text tokenization and vocabulary
├── lm_dataset.py          # Dataset for GPT-style training
├── train_lm.py            # Train GPT-style model
├── lm_chat.py             # Chat with your trained GPT model
├── english_corpus.txt     # Conversation training data (3,690+ pairs)
├── test_main.py           # Unit tests for transformer components
└── README.md              # This file
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

### Train Your GPT Model
```bash
# Train the model (takes ~15 minutes on CPU)
python3 train_lm.py
```

### Chat with Your Model
```bash
# Start interactive chat
python3 lm_chat.py
```

## 💬 Chat Interface Example

```
🤖 GPT-STYLE CONVERSATIONAL AI
============================================================
👤 You: Hi, how are you doing?
🤖 GPT: I'm doing well, thank you! How about yourself?

👤 You: I'm pretty good, thanks for asking.
🤖 GPT: That's great to hear! What have you been up to?

👤 You: generate Once upon a time
🤖 GPT: Once upon a time there was a young girl who lived in a small village...

👤 You: quit
🤖 GPT: Goodbye! Thanks for chatting!
```

## 📊 Model Details

### GPT Model Configuration
- **Architecture**: Decoder-only transformer (like ChatGPT)
- **Parameters**: ~2M (lightweight but effective)
- **Training Time**: ~15 minutes on CPU
- **Memory Usage**: ~200MB during training
- **Dataset**: Sequential text from conversation pairs
- **Vocabulary**: ~5,000 tokens from your conversation data

### Training Process
1. **Tokenization**: Converts text to numerical tokens
2. **Language Modeling**: Learns to predict next token in sequence
3. **Autoregressive Training**: Each token predicts the following token
4. **Conversation Learning**: Learns patterns from 3,690+ conversation pairs

## 🔧 Advanced Usage

### Chat Commands
```bash
# Start normal chat
python3 lm_chat.py

# Test interface without trained model
python3 lm_chat.py test

# Check model status
python3 lm_chat.py status

# Show help
python3 lm_chat.py help
```

### Training Monitoring
```bash
# Watch training progress
python3 train_lm.py

# Check if model files exist
python3 lm_chat.py status
```

### Custom Training Parameters
Edit the configuration in `train_lm.py`:
```python
# Config
DATA_PATH = 'english_corpus.txt' 
BATCH_SIZE = 32           # Reduce for less memory usage
EPOCHS = 10               # More epochs = better quality
MAX_LEN = 32              # Maximum sequence length
d_model = 128             # Model dimension (larger = more powerful)
n_layers = 2              # Number of transformer layers
n_heads = 4               # Number of attention heads
```

## 🧠 How It Works

### Core Components (main.py)
- **Input Embedding**: Converts tokens to vectors
- **Positional Encoding**: Adds position information to tokens
- **Multi-Head Attention**: The core "attention is all you need" mechanism
- **Feed Forward Networks**: Non-linear transformations
- **Layer Normalization**: Stabilizes training
- **Decoder Blocks**: Complete transformer decoder layers

### GPT Training Process
1. **Text Processing**: Converts conversations to token sequences
2. **Next Token Prediction**: Model learns to predict what comes next
3. **Attention Learning**: Learns which words to focus on
4. **Pattern Recognition**: Discovers conversation patterns and responses
5. **Generation**: Uses learned patterns to create new responses

## 📈 Performance Tips

### For Better Conversation Quality
1. **More Training Data**: Add more conversation pairs to `english_corpus.txt`
2. **Longer Training**: Increase `EPOCHS` in `train_lm.py`
3. **Larger Model**: Increase `d_model` and `n_layers`
4. **Better Hardware**: Use GPU for faster training

### Memory Optimization
1. **Reduce Batch Size**: Lower `BATCH_SIZE` in config
2. **Shorter Sequences**: Reduce `MAX_LEN`
3. **Smaller Model**: Decrease `d_model` and `n_layers`

## 🐛 Troubleshooting

### Common Issues

**"Model files not found"**
```bash
# Train the model first
python3 train_lm.py
```

**Out of memory errors**
- Reduce `BATCH_SIZE` from 32 to 16 or 8
- Reduce `MAX_LEN` from 32 to 16
- Use smaller model dimensions

**Poor conversation quality**
- Train for more epochs (increase `EPOCHS`)
- Add more conversation data to `english_corpus.txt`
- Use larger model (increase `d_model`)

**Training too slow**
- Reduce `EPOCHS` for faster results
- Use smaller `BATCH_SIZE`
- Consider using GPU if available

## 📚 Understanding the Code

### Key Files Explained
- **`main.py`**: Core transformer building blocks (study this to understand transformers)
- **`train_lm.py`**: GPT training loop (see how language models are trained)
- **`lm_chat.py`**: Inference and chat interface (understand text generation)
- **`tokenizer.py`**: Text processing (learn about tokenization)
- **`lm_dataset.py`**: Data preparation (see how training data is formatted)

### Learning Path
1. **Start with `main.py`** - Understand transformer components
2. **Read `train_lm.py`** - See how training works
3. **Explore `lm_chat.py`** - Learn about text generation
4. **Experiment** - Modify parameters and see what happens!

## 🎯 Use Cases

### Educational
- **Learn Transformers**: Hands-on implementation of "Attention Is All You Need"
- **Understand GPT**: See how ChatGPT-style models work
- **Study Attention**: Visualize how attention mechanisms focus on different words

### Practical
- **Custom Chatbots**: Train on your own conversation data
- **Text Generation**: Creative writing assistance
- **Domain-Specific AI**: Train on specialized conversations (customer service, technical support, etc.)

## 🔮 Future Enhancements

- **Larger Models**: Scale up for better performance
- **Multi-language Support**: Train on different languages
- **Fine-tuning**: Adapt pre-trained models
- **Web Interface**: Browser-based chat interface
- **Voice Integration**: Speech-to-text and text-to-speech

## 📄 Files Generated During Training

After training, you'll have:
- `language_model.pt` - Your trained GPT model (2-5MB)
- `tokenizer.pt` - Vocabulary and tokenization rules (~100KB)

## 🧪 Testing

Run the unit tests to verify transformer components:
```bash
python3 test_main.py
```

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 👨‍💻 Author

**Amanuel Ayalew**

Built with ❤️ using PyTorch and the power of attention mechanisms.

---

*"Attention is all you need" - and now you have it in its simplest, most effective form!* 🚀