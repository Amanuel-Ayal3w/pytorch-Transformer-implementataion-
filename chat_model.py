import torch
import torch.nn as nn
from main import build_transformer

class ConversationalTransformer(nn.Module):
    """
    Conversational transformer wrapper using the full encoder-decoder architecture.
    Built on top of the transformer implementation from main.py.
    """
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=8, d_ff=1024, 
                 max_len=64, dropout=0.1):
        """
        Initialize conversational transformer.
        
        Args:
            vocab_size (int): Size of vocabulary
            d_model (int): Model dimension
            n_layers (int): Number of encoder/decoder layers
            n_heads (int): Number of attention heads
            d_ff (int): Feed-forward dimension
            max_len (int): Maximum sequence length
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # Build the transformer using your architecture from main.py
        self.transformer = build_transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,  # Same vocab for questions and answers
            src_seq_len=max_len,
            tgt_seq_len=max_len,
            d_model=d_model,
            N=n_layers,
            h=n_heads,
            dropout=dropout,
            d_ff=d_ff
        )
        
        # Store special token indices (will be set by tokenizer)
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.unk_token = 3
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Forward pass for training with teacher forcing.
        
        Args:
            src (torch.Tensor): Source sequences (questions) [batch, src_len]
            tgt (torch.Tensor): Target sequences (answers) [batch, tgt_len]
            src_mask (torch.Tensor): Source attention mask
            tgt_mask (torch.Tensor): Target attention mask
            
        Returns:
            torch.Tensor: Output logits [batch, tgt_len, vocab_size]
        """
        # Encode the source (question)
        encoder_output = self.transformer.encode(src, src_mask)
        
        # Decode with target (answer) using teacher forcing
        decoder_output = self.transformer.decode(encoder_output, src_mask, tgt, tgt_mask)
        
        # Project to vocabulary
        output = self.transformer.project(decoder_output)
        
        return output
    
    def generate_response(self, question_tokens, tokenizer, max_gen_len=50, 
                         temperature=1.0, top_k=None):
        """
        Generate response for a given question using greedy decoding.
        
        Args:
            question_tokens (torch.Tensor): Tokenized question [1, seq_len]
            tokenizer: Tokenizer object with word2idx and idx2word
            max_gen_len (int): Maximum generation length
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling (None for greedy)
            
        Returns:
            list: Generated token indices
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Ensure question is on correct device and has batch dimension
            if question_tokens.dim() == 1:
                question_tokens = question_tokens.unsqueeze(0)
            question_tokens = question_tokens.to(device)
            
            # Create source mask (hide padding tokens)
            src_mask = self.create_src_mask(question_tokens)
            
            # Encode the question
            encoder_output = self.transformer.encode(question_tokens, src_mask)
            
            # Start with <sos> token
            generated = [self.sos_token]
            
            # Generate tokens one by one
            for _ in range(max_gen_len):
                # Current target sequence
                tgt_tokens = torch.tensor([generated], device=device)
                
                # Create target mask (causal mask)
                tgt_mask = self.create_tgt_mask(tgt_tokens)
                
                # Decode next token
                decoder_output = self.transformer.decode(
                    encoder_output, src_mask, tgt_tokens, tgt_mask
                )
                
                # Get logits for last position
                logits = self.transformer.project(decoder_output)
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
                
                # Add to generated sequence
                generated.append(next_token)
                
                # Stop if we generate <eos> token
                if next_token == self.eos_token:
                    break
            
            # Remove <sos> token from output
            return generated[1:]
    
    def chat(self, question, tokenizer, max_gen_len=50):
        """
        High-level chat interface.
        
        Args:
            question (str): Input question
            tokenizer: Tokenizer object
            max_gen_len (int): Maximum response length
            
        Returns:
            str: Generated response
        """
        # Tokenize question
        question_tokens = tokenizer.encode(question)
        question_tensor = torch.tensor([self.sos_token] + question_tokens)
        
        # Generate response
        response_tokens = self.generate_response(
            question_tensor, tokenizer, max_gen_len
        )
        
        # Decode response (remove <eos> if present)
        if response_tokens and response_tokens[-1] == self.eos_token:
            response_tokens = response_tokens[:-1]
        
        response = tokenizer.decode(response_tokens)
        return response.strip()
    
    def create_src_mask(self, src):
        """
        Create source mask to hide padding tokens in encoder.
        
        Args:
            src (torch.Tensor): Source sequences [batch, src_len]
            
        Returns:
            torch.Tensor: Source mask [batch, 1, 1, src_len]
        """
        # Mask padding tokens (assuming pad_token = 0)
        src_mask = (src != self.pad_token).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def create_tgt_mask(self, tgt):
        """
        Create target mask for decoder (causal mask + padding mask).
        
        Args:
            tgt (torch.Tensor): Target sequences [batch, tgt_len]
            
        Returns:
            torch.Tensor: Target mask [batch, 1, tgt_len, tgt_len]
        """
        batch_size, tgt_len = tgt.shape
        device = tgt.device
        
        # Create causal mask (lower triangular) - convert to bool
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
        
        # Create padding mask
        padding_mask = (tgt != self.pad_token).unsqueeze(1).unsqueeze(2)
        
        # Combine masks: causal AND padding
        tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0) & padding_mask
        
        return tgt_mask

# Test the model creation
if __name__ == "__main__":
    # Test model initialization
    vocab_size = 1000
    model = ConversationalTransformer(
        vocab_size=vocab_size,
        d_model=128,  # Smaller for testing
        n_layers=2,
        n_heads=4,
        max_len=32
    )
    
    print(f"Model created successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create simple masks (all ones for testing)
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_mask.expand(batch_size, 1, seq_len, seq_len)
    
    output = model(src, tgt, src_mask, tgt_mask)
    print(f"Output shape: {output.shape}")  # Should be [batch_size, seq_len, vocab_size]

def create_masks(src, tgt, pad_token=0):
    """
    Utility function to create both source and target masks.
    
    Args:
        src (torch.Tensor): Source sequences [batch, src_len]
        tgt (torch.Tensor): Target sequences [batch, tgt_len]
        pad_token (int): Padding token index
        
    Returns:
        tuple: (src_mask, tgt_mask)
    """
    # Source mask: hide padding tokens
    src_mask = (src != pad_token).unsqueeze(1).unsqueeze(2)
    
    # Target mask: causal + padding
    batch_size, tgt_len = tgt.shape
    device = tgt.device
    
    # Causal mask (lower triangular) - convert to bool
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
    
    # Padding mask
    padding_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(2)
    
    # Combine masks
    tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0) & padding_mask
    
    return src_mask, tgt_mask

# Test mask creation
if __name__ == "__main__":
    print("\nTesting mask creation...")
    
    # Test data with padding
    batch_size = 2
    src_len = 8
    tgt_len = 6
    pad_token = 0
    
    # Create test sequences with padding
    src = torch.tensor([
        [1, 2, 3, 4, 0, 0, 0, 0],  # Sequence with padding
        [5, 6, 7, 8, 9, 0, 0, 0]   # Another sequence with padding
    ])
    
    tgt = torch.tensor([
        [1, 2, 3, 0, 0, 0],  # Target with padding
        [4, 5, 6, 7, 0, 0]   # Another target with padding
    ])
    
    # Create masks
    src_mask, tgt_mask = create_masks(src, tgt, pad_token)
    
    print(f"Source mask shape: {src_mask.shape}")
    print(f"Target mask shape: {tgt_mask.shape}")
    print(f"Source mask (first sample):\n{src_mask[0, 0, 0]}")
    print(f"Target mask (first sample):\n{tgt_mask[0, 0]}")