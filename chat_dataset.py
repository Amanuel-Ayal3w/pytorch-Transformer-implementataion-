import torch
from torch.utils.data import Dataset
from tokenizer import SimpleTokenizer

def parse_conversation_data(file_path):
    """
    Parse conversation data from tab-separated file format.
    Each line contains: question\tanswer
    
    Args:
        file_path (str): Path to the conversation data file
        
    Returns:
        list: List of (question, answer) tuples
    """
    conversations = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    question = parts[0].strip()
                    answer = parts[1].strip()
                    
                    # Skip empty questions or answers
                    if question and answer:
                        conversations.append((question, answer))
    
    return conversations

def clean_conversation_pairs(conversations):
    """
    Clean and filter conversation pairs.
    
    Args:
        conversations (list): List of (question, answer) tuples
        
    Returns:
        list: Cleaned conversation pairs
    """
    cleaned = []
    
    for question, answer in conversations:
        # Basic cleaning
        question = question.strip().lower()
        answer = answer.strip().lower()
        
        # Skip very short or very long conversations
        if len(question.split()) < 2 or len(answer.split()) < 1:
            continue
        if len(question.split()) > 50 or len(answer.split()) > 50:
            continue
            
        # Skip duplicate pairs
        if (question, answer) not in cleaned:
            cleaned.append((question, answer))
    
    return cleaned

# Test the parsing function
if __name__ == "__main__":
    # Test conversation parsing
    conversations = parse_conversation_data('english_corpus.txt')
    cleaned_conversations = clean_conversation_pairs(conversations)
    
    print(f"Total conversations: {len(conversations)}")
    print(f"Cleaned conversations: {len(cleaned_conversations)}")
    print("\nFirst 5 conversation pairs:")
    for i, (q, a) in enumerate(cleaned_conversations[:5]):
        print(f"{i+1}. Q: {q}")
        print(f"   A: {a}")
        print()

class ConversationalDataset(Dataset):
    """
    Dataset class for conversational question-answer pairs.
    Processes conversations for encoder-decoder transformer training.
    """
    
    def __init__(self, data_path, tokenizer, max_len=64):
        """
        Initialize conversational dataset.
        
        Args:
            data_path (str): Path to conversation data file
            tokenizer (SimpleTokenizer): Tokenizer for encoding text
            max_len (int): Maximum sequence length for padding/truncation
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Parse and clean conversation data
        conversations = parse_conversation_data(data_path)
        self.conversations = clean_conversation_pairs(conversations)
        
        print(f"Loaded {len(self.conversations)} conversation pairs")
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        """
        Get a conversation pair as tokenized tensors.
        
        Returns:
            tuple: (src_tokens, tgt_input, tgt_output) where:
                - src_tokens: Source sequence (question) with <sos> token
                - tgt_input: Target input sequence (<sos> + answer)
                - tgt_output: Target output sequence (answer + <eos>)
        """
        question, answer = self.conversations[idx]
        
        # Tokenize question and answer
        question_tokens = self.tokenizer.encode(question)
        answer_tokens = self.tokenizer.encode(answer)
        
        # Create source sequence: <sos> + question
        src_tokens = [self.tokenizer.word2idx['<sos>']] + question_tokens
        
        # Create target input: <sos> + answer (for teacher forcing)
        tgt_input = [self.tokenizer.word2idx['<sos>']] + answer_tokens
        
        # Create target output: answer + <eos> (for loss calculation)
        tgt_output = answer_tokens + [self.tokenizer.word2idx['<eos>']]
        
        # Pad or truncate sequences
        src_tokens = self._pad_sequence(src_tokens)
        tgt_input = self._pad_sequence(tgt_input)
        tgt_output = self._pad_sequence(tgt_output)
        
        return (
            torch.tensor(src_tokens, dtype=torch.long),
            torch.tensor(tgt_input, dtype=torch.long),
            torch.tensor(tgt_output, dtype=torch.long)
        )
    
    def _pad_sequence(self, tokens):
        """
        Pad or truncate sequence to max_len.
        
        Args:
            tokens (list): List of token indices
            
        Returns:
            list: Padded/truncated token list
        """
        if len(tokens) > self.max_len:
            # Truncate if too long
            return tokens[:self.max_len]
        else:
            # Pad if too short
            pad_token = self.tokenizer.word2idx['<pad>']
            return tokens + [pad_token] * (self.max_len - len(tokens))

# Example usage and testing
if __name__ == "__main__":
    # Test the complete dataset
    from tokenizer import SimpleTokenizer
    
    # Create and fit tokenizer
    print("Creating tokenizer...")
    tokenizer = SimpleTokenizer()
    
    # Read all text for tokenizer fitting
    with open('english_corpus.txt', 'r', encoding='utf-8') as f:
        all_text = f.read()
    
    tokenizer.fit([all_text])
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size()}")
    
    # Create dataset
    print("\nCreating conversational dataset...")
    dataset = ConversationalDataset('english_corpus.txt', tokenizer, max_len=32)
    
    # Test a few samples
    print(f"\nDataset size: {len(dataset)}")
    print("\nFirst 3 samples:")
    for i in range(min(3, len(dataset))):
        src, tgt_in, tgt_out = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"Source (question): {tokenizer.decode(src.tolist())}")
        print(f"Target input: {tokenizer.decode(tgt_in.tolist())}")
        print(f"Target output: {tokenizer.decode(tgt_out.tolist())}")