import torch
import sys
import os
from train_lm import GPTLanguageModel
from tokenizer import SimpleTokenizer

class GPTChatBot:
   

    
    def __init__(self, model_path='language_model.pt', tokenizer_path='tokenizer.pt'):
   
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🤖 Loading GPT ChatBot...")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = self.load_tokenizer()
        
        # Load model
        self.model = self.load_model()
        
        print(" GPT ChatBot ready!")
    
    def load_tokenizer(self):
        """Load the tokenizer from file."""
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {self.tokenizer_path}")
        
        print(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = torch.load(self.tokenizer_path, map_location=self.device, weights_only=False)
        print(f"Vocabulary size: {tokenizer.vocab_size()}")
        return tokenizer
    
    def load_model(self):
        """Load the trained GPT model from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading GPT model from {self.model_path}")
        
        # Create model with same architecture as training
        vocab_size = self.tokenizer.vocab_size()
        model = GPTLanguageModel(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            max_len=32,
            dropout=0.1
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        model = model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        return model
    
    def generate_text(self, prompt, max_gen=50, temperature=1.0, top_k=None):
        """
        Generate text continuation using GPT model.
        
        Args:
            prompt (str): Input prompt
            max_gen (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling (None for greedy)
            
        Returns:
            str: Generated text
        """
        self.model.eval()
        
        with torch.no_grad():
            # Encode prompt
            input_ids = [self.tokenizer.word2idx.get('<sos>', 1)] + self.tokenizer.encode(prompt)
            input_ids = input_ids[-32:]  # Keep last 32 tokens (model's max_len)
            
            # Generate tokens
            for _ in range(max_gen):
                # Prepare input tensor
                x = torch.tensor([input_ids[-32:]], device=self.device)
                
                # Forward pass
                logits = self.model(x)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample next token
                if top_k is not None:
                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1).item()
                    next_token = top_k_indices[next_token_idx].item()
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits).item()
                
                # Add to sequence
                input_ids.append(next_token)
                
                # Stop if we generate <eos> token
                if next_token == self.tokenizer.word2idx.get('<eos>', 2):
                    break
            
            # Decode generated text (remove original prompt)
            prompt_len = len(self.tokenizer.encode(prompt)) + 1  # +1 for <sos>
            generated_tokens = input_ids[prompt_len:]
            
            # Remove <eos> if present
            if generated_tokens and generated_tokens[-1] == self.tokenizer.word2idx.get('<eos>', 2):
                generated_tokens = generated_tokens[:-1]
            
            return self.tokenizer.decode(generated_tokens)
    
    def chat(self, user_input, max_gen=30):
        """
        Generate conversational response.
        
        Args:
            user_input (str): User's message
            max_gen (int): Maximum response length
            
        Returns:
            str: Generated response
        """
        if not user_input.strip():
            return "Please say something!"
        
        try:
            # Clean input
            user_input = user_input.strip().lower()
            
            # For conversation, we can try different approaches:
            # Approach 1: Direct continuation
            response = self.generate_text(user_input, max_gen=max_gen, top_k=10)
            
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
        print("🤖 GPT-STYLE CONVERSATIONAL AI")
        print("="*60)
        print("Type your messages and press Enter to chat!")
        print("This model uses GPT-style (decoder-only) architecture")
        print("Commands:")
        print("  'quit' or 'exit' - End conversation")
        print("  'help' - Show this help")
        print("  'clear' - Clear screen")
        print("  'generate <text>' - Generate text continuation")
        print("="*60)
        
        conversation_count = 0
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n🤖 GPT: Goodbye! Thanks for chatting!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n🤖 GPT: I'm a GPT-style conversational AI (decoder-only architecture).")
                    print("I generate text by predicting the next tokens, just like ChatGPT!")
                    print("Try asking questions or starting conversations.")
                    continue
                
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print("🤖 GPT-STYLE CONVERSATIONAL AI")
                    print("="*60)
                    continue
                
                elif user_input.lower().startswith('generate '):
                    # Text generation mode
                    prompt = user_input[9:]  # Remove 'generate '
                    print("🤖 GPT: ", end="", flush=True)
                    response = self.generate_text(prompt, max_gen=50)
                    print(f"{prompt}{response}")
                    continue
                
                elif not user_input:
                    print("🤖 GPT: Please say something!")
                    continue
                
                # Generate conversational response
                print("🤖 GPT: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
                conversation_count += 1
                
                # Encourage longer conversations
                if conversation_count % 10 == 0:
                    print(f"\n💬 We've had {conversation_count} exchanges! Keep the conversation going!")
                
            except KeyboardInterrupt:
                print("\n\n🤖 GPT: Goodbye! Thanks for chatting!")
                break
            except EOFError:
                print("\n\n🤖 GPT: Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("🤖 GPT: Sorry, something went wrong. Let's keep chatting!")

def test_gpt_without_training():
    """
    Test GPT interface with dummy responses (for development/testing).
    """
    print("🧪 TESTING GPT INTERFACE (No trained model)")
    print("="*60)
    print("This is a test mode - responses will be random!")
    print("Type 'quit' to exit")
    print("="*60)
    
    import random
    
    # Sample GPT-style responses
    sample_responses = [
        "That's really interesting to think about.",
        "I understand what you're saying.",
        "Can you elaborate on that?",
        "That makes a lot of sense.",
        "I'm curious about your perspective on this.",
        "What do you think would happen if...?",
        "That's a fascinating point.",
        "I hadn't considered that before.",
        "How do you feel about that?",
        "That's quite thought-provoking."
    ]
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n🤖 GPT: Goodbye from test mode!")
                break
            
            if not user_input:
                continue
            
            # Random response for testing
            response = random.choice(sample_responses)
            print(f"🤖 GPT: {response}")
            
        except KeyboardInterrupt:
            print("\n\n🤖 GPT: Goodbye from test mode!")
            break

def check_gpt_model_status():
    """
    Check if GPT model files exist and provide helpful information.
    """
    model_path = 'language_model.pt'
    tokenizer_path = 'tokenizer.pt'
    
    print("\n📋 GPT MODEL STATUS CHECK")
    print("="*40)
    
    # Check model files
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"✅ GPT Model: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"❌ GPT Model: {model_path} (not found)")
    
    if os.path.exists(tokenizer_path):
        size_kb = os.path.getsize(tokenizer_path) / 1024
        print(f"✅ Tokenizer: {tokenizer_path} ({size_kb:.1f} KB)")
    else:
        print(f"❌ Tokenizer: {tokenizer_path} (not found)")
    
    print("="*40)
    
    # Provide recommendations
    if not os.path.exists(model_path):
        print("\n💡 RECOMMENDATIONS:")
        print("1. Train GPT model: python3 train_lm.py")
        print("2. Wait for training to complete")
        print("3. Then run: python3 lm_chat.py")
    else:
        print("\n🚀 READY TO CHAT:")
        print("Run: python3 lm_chat.py")

def main():
    """
    Main function to start the GPT chatbot.
    """
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            test_gpt_without_training()
            return
        elif command == 'status':
            check_gpt_model_status()
            return
        elif command == 'help':
            print("🤖 GPT CHATBOT USAGE:")
            print("python3 lm_chat.py          - Start GPT chatbot (requires trained model)")
            print("python3 lm_chat.py test     - Test interface without model")
            print("python3 lm_chat.py status   - Check model files")
            print("python3 lm_chat.py help     - Show this help")
            return
    
    # Default behavior - start GPT chatbot
    try:
        # Check if model files exist
        model_path = 'language_model.pt'
        tokenizer_path = 'tokenizer.pt'
        
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            print("❌ GPT model files not found!")
            print("\nOptions:")
            print("1. Train the GPT model: python3 train_lm.py")
            print("2. Test interface: python3 lm_chat.py test")
            print("3. Check status: python3 lm_chat.py status")
            return
        
        # Create and start GPT chatbot
        chatbot = GPTChatBot(model_path, tokenizer_path)
        chatbot.interactive_chat()
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please make sure you have trained the GPT model first!")
    except Exception as e:
        print(f"❌ Error starting GPT chatbot: {e}")
        print("Please check your model files and try again.")

if __name__ == "__main__":
    main()