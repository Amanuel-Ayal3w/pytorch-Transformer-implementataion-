import torch
import sys
import os
from chat_model import ConversationalTransformer
from tokenizer import SimpleTokenizer

class ChatBot:
    """
    Interactive chatbot using trained conversational transformer.
    """
    
    def __init__(self, model_path='chat_model.pt', tokenizer_path='chat_tokenizer.pt'):
        """
        Initialize chatbot with trained model and tokenizer.
        
        Args:
            model_path (str): Path to trained model
            tokenizer_path (str): Path to tokenizer
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🤖 Loading ChatBot...")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = self.load_tokenizer()
        
        # Load model
        self.model = self.load_model()
        
        print("✅ ChatBot ready!")
    
    def load_tokenizer(self):
        """Load the tokenizer from file."""
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {self.tokenizer_path}")
        
        print(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = torch.load(self.tokenizer_path, map_location=self.device, weights_only=False)
        print(f"Vocabulary size: {tokenizer.vocab_size()}")
        return tokenizer
    
    def load_model(self):
        """Load the trained model from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        
        # Create model with same architecture as training
        vocab_size = self.tokenizer.vocab_size()
        model = ConversationalTransformer(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=4,
            n_heads=8,
            d_ff=1024,
            max_len=64,
            dropout=0.1
        )
        
        # Set special tokens
        model.pad_token = self.tokenizer.word2idx['<pad>']
        model.sos_token = self.tokenizer.word2idx['<sos>']
        model.eos_token = self.tokenizer.word2idx['<eos>']
        model.unk_token = self.tokenizer.word2idx['<unk>']
        
        # Load trained weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        model = model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        return model
    
    def chat(self, user_input, max_gen_len=50, temperature=1.0, top_k=None):
        """
        Generate response to user input.
        
        Args:
            user_input (str): User's message
            max_gen_len (int): Maximum response length
            temperature (float): Sampling temperature (1.0 = normal, <1.0 = focused)
            top_k (int): Top-k sampling (None for greedy)
            
        Returns:
            str: Generated response
        """
        if not user_input.strip():
            return "Please say something!"
        
        try:
            # Clean input
            user_input = user_input.strip().lower()
            
            # Generate response using model's chat method
            response = self.model.chat(
                user_input, 
                self.tokenizer, 
                max_gen_len=max_gen_len
            )
            
            # Clean response
            response = response.strip()
            if not response:
                return "I'm not sure how to respond to that."
            
            # Capitalize first letter
            if response:
                response = response[0].upper() + response[1:] if len(response) > 1 else response.upper()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I had trouble understanding that."
    
    def interactive_chat(self):
        """
        Start interactive chat session.
        """
        print("\n" + "="*60)
        print("🤖 CONVERSATIONAL AI CHATBOT")
        print("="*60)
        print("Type your messages and press Enter to chat!")
        print("Commands:")
        print("  'quit' or 'exit' - End conversation")
        print("  'help' - Show this help")
        print("  'clear' - Clear screen")
        print("="*60)
        
        conversation_count = 0
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n🤖 Bot: Goodbye! Thanks for chatting!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n🤖 Bot: I'm a conversational AI trained on English conversations.")
                    print("Just type naturally and I'll try to respond appropriately!")
                    continue
                
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print("🤖 CONVERSATIONAL AI CHATBOT")
                    print("="*60)
                    continue
                
                elif not user_input:
                    print("🤖 Bot: Please say something!")
                    continue
                
                # Generate response
                print("🤖 Bot: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
                conversation_count += 1
                
                # Encourage longer conversations
                if conversation_count % 10 == 0:
                    print(f"\n💬 We've had {conversation_count} exchanges! Keep the conversation going!")
                
            except KeyboardInterrupt:
                print("\n\n🤖 Bot: Goodbye! Thanks for chatting!")
                break
            except EOFError:
                print("\n\n🤖 Bot: Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("🤖 Bot: Sorry, something went wrong. Let's keep chatting!")

def main():
    """
    Main function to start the chatbot.
    """
    try:
        # Check if model files exist
        model_path = 'chat_model.pt'
        tokenizer_path = 'chat_tokenizer.pt'
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Please train the model first by running: python3 train_chat.py")
            return
        
        if not os.path.exists(tokenizer_path):
            print(f"❌ Tokenizer file not found: {tokenizer_path}")
            print("Please train the model first by running: python3 train_chat.py")
            return
        
        # Create and start chatbot
        chatbot = ChatBot(model_path, tokenizer_path)
        chatbot.interactive_chat()
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please make sure you have trained the model first!")
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        print("Please check your model files and try again.")

if __name__ == "__main__":
    main()

# Additional utility functions for enhanced chat experience

def test_chatbot_without_training():
    """
    Test chatbot interface with a dummy model (for development/testing).
    """
    print("🧪 TESTING CHATBOT INTERFACE (No trained model)")
    print("="*60)
    print("This is a test mode - responses will be random!")
    print("Type 'quit' to exit")
    print("="*60)
    
    import random
    
    # Sample responses for testing
    sample_responses = [
        "That's interesting!",
        "I see what you mean.",
        "Can you tell me more about that?",
        "That sounds great!",
        "I'm not sure I understand.",
        "What do you think about it?",
        "That's a good point.",
        "I'd like to know more.",
        "How does that make you feel?",
        "That's really cool!"
    ]
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n🤖 Bot: Goodbye from test mode!")
                break
            
            if not user_input:
                continue
            
            # Random response for testing
            response = random.choice(sample_responses)
            print(f"🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye from test mode!")
            break

def create_demo_conversation():
    """
    Create a demo conversation to show expected behavior.
    """
    demo_conversations = [
        ("Hi, how are you doing?", "I'm doing well, thank you! How about yourself?"),
        ("I'm pretty good, thanks for asking.", "That's great to hear! What have you been up to?"),
        ("I've been working on some projects.", "That sounds interesting! What kind of projects?"),
        ("Just some programming stuff.", "Programming is fun! What language are you using?"),
        ("Mostly Python these days.", "Python is a great choice! Very versatile."),
    ]
    
    print("\n" + "="*60)
    print("🎭 DEMO CONVERSATION")
    print("="*60)
    print("Here's what a conversation might look like:")
    print("="*60)
    
    for question, answer in demo_conversations:
        print(f"\n👤 You: {question}")
        print(f"🤖 Bot: {answer}")
    
    print("\n" + "="*60)
    print("This is what your trained model will be able to do!")
    print("="*60)

def check_model_status():
    """
    Check if model files exist and provide helpful information.
    """
    model_path = 'chat_model.pt'
    tokenizer_path = 'chat_tokenizer.pt'
    checkpoint_path = 'checkpoint.pt'
    
    print("\n📋 MODEL STATUS CHECK")
    print("="*40)
    
    # Check model files
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"✅ Model file: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Model file: {model_path} (not found)")
    
    if os.path.exists(tokenizer_path):
        size_kb = os.path.getsize(tokenizer_path) / 1024
        print(f"✅ Tokenizer file: {tokenizer_path} ({size_kb:.1f} KB)")
    else:
        print(f"❌ Tokenizer file: {tokenizer_path} (not found)")
    
    # Check training progress
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
        print(f"📊 Checkpoint file: {checkpoint_path} ({size_mb:.1f} MB)")
        print("   (Training in progress or completed)")
    else:
        print(f"📊 Checkpoint file: {checkpoint_path} (not found)")
    
    # Check training log
    if os.path.exists('training_log.txt'):
        with open('training_log.txt', 'r') as f:
            lines = f.readlines()
        print(f"📝 Training log: {len(lines)-2} epochs logged")
        if len(lines) > 2:
            last_line = lines[-1].strip()
            print(f"   Last entry: {last_line}")
    else:
        print("📝 Training log: not found")
    
    print("="*40)
    
    # Provide recommendations
    if not os.path.exists(model_path):
        print("\n💡 RECOMMENDATIONS:")
        print("1. Start training: python3 train_chat.py")
        print("2. Wait for training to complete")
        print("3. Then run: python3 chatbot.py")
    else:
        print("\n🚀 READY TO CHAT:")
        print("Run: python3 chatbot.py")

# Enhanced main function with options
def main():
    """
    Enhanced main function with multiple options.
    """
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            test_chatbot_without_training()
            return
        elif command == 'demo':
            create_demo_conversation()
            return
        elif command == 'status':
            check_model_status()
            return
        elif command == 'help':
            print("🤖 CHATBOT USAGE:")
            print("python3 chatbot.py          - Start chatbot (requires trained model)")
            print("python3 chatbot.py test     - Test interface without model")
            print("python3 chatbot.py demo     - Show demo conversation")
            print("python3 chatbot.py status   - Check model files")
            print("python3 chatbot.py help     - Show this help")
            return
    
    # Default behavior - start chatbot
    try:
        # Check if model files exist
        model_path = 'chat_model.pt'
        tokenizer_path = 'chat_tokenizer.pt'
        
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            print("❌ Model files not found!")
            print("\nOptions:")
            print("1. Train the model: python3 train_chat.py")
            print("2. Test interface: python3 chatbot.py test")
            print("3. Check status: python3 chatbot.py status")
            return
        
        # Create and start chatbot
        chatbot = ChatBot(model_path, tokenizer_path)
        chatbot.interactive_chat()
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please make sure you have trained the model first!")
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        print("Please check your model files and try again.")

if __name__ == "__main__":
    main()