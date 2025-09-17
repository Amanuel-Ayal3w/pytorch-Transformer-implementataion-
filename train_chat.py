import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import math

from tokenizer import SimpleTokenizer
from chat_dataset import ConversationalDataset
from chat_model import ConversationalTransformer, create_masks

# Training Configuration
CONFIG = {
    'data_path': 'english_corpus.txt',
    'batch_size': 16,
    'epochs': 20,
    'max_len': 64,
    'd_model': 256,
    'n_layers': 4,
    'n_heads': 8,
    'd_ff': 1024,
    'dropout': 0.1,
    'learning_rate': 3e-4,
    'warmup_steps': 100,
    'max_grad_norm': 1.0,
    'save_every': 5,
    'model_path': 'chat_model.pt',
    'tokenizer_path': 'chat_tokenizer.pt',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def create_learning_rate_scheduler(optimizer, d_model, warmup_steps):
    """
    Create learning rate scheduler with warmup (simplified and more stable).
    
    Args:
        optimizer: PyTorch optimizer
        d_model (int): Model dimension
        warmup_steps (int): Number of warmup steps
        
    Returns:
        function: Learning rate scheduler function
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay after warmup
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (10000 - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, max_grad_norm):
    """
    Train for one epoch using attention-based encoder-decoder.
    
    Args:
        model: ConversationalTransformer model
        dataloader: Training data loader
        optimizer: PyTorch optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler
        device: Training device
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, (src, tgt_input, tgt_output) in enumerate(dataloader):
        # Move to device
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)
        
        # Create attention masks
        src_mask, tgt_mask = create_masks(src, tgt_input, pad_token=0)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        # Forward pass with teacher forcing
        optimizer.zero_grad()
        
        # Model forward pass: encoder processes src, decoder generates from tgt_input
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # Calculate loss against tgt_output
        # Reshape for cross entropy: [batch*seq, vocab_size] and [batch*seq]
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = tgt_output.view(-1)
        
        loss = criterion(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Print progress
        if batch_idx % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
    
    return total_loss / num_batches

def validate_model(model, dataloader, criterion, device):
    """
    Validate the model on validation data.
    
    Args:
        model: ConversationalTransformer model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device for computation
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for src, tgt_input, tgt_output in dataloader:
            # Move to device
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            
            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt_input, pad_token=0)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            
            # Forward pass
            logits = model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = tgt_output.view(-1)
            loss = criterion(logits_flat, targets_flat)
            
            total_loss += loss.item()
    
    return total_loss / num_batches

def test_generation(model, tokenizer, device, test_questions):
    """
    Test the model's generation capability with sample questions.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        device: Device
        test_questions: List of test questions
    """
    model.eval()
    print("\n" + "="*50)
    print("TESTING GENERATION")
    print("="*50)
    
    for question in test_questions:
        print(f"\nQ: {question}")
        try:
            response = model.chat(question, tokenizer, max_gen_len=30)
            print(f"A: {response}")
        except Exception as e:
            print(f"A: [Generation failed: {e}]")
    
    print("="*50)

def train_conversational_model():
    """
    Main training function for conversational transformer.
    """
    print("Starting Conversational Transformer Training")
    print(f"Device: {CONFIG['device']}")
    print(f"Configuration: {CONFIG}")
    
    # 1. Prepare tokenizer and dataset
    print("\n1. Loading and preparing data...")
    
    # Load text for tokenizer
    with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create and fit tokenizer
    tokenizer = SimpleTokenizer(max_vocab_size=8000)  # Larger vocab for conversations
    tokenizer.fit([text])
    vocab_size = tokenizer.vocab_size()
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset
    dataset = ConversationalDataset(CONFIG['data_path'], tokenizer, CONFIG['max_len'])
    
    # Split into train/validation (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 2. Create model
    print("\n2. Creating model...")
    model = ConversationalTransformer(
        vocab_size=vocab_size,
        d_model=CONFIG['d_model'],
        n_layers=CONFIG['n_layers'],
        n_heads=CONFIG['n_heads'],
        d_ff=CONFIG['d_ff'],
        max_len=CONFIG['max_len'],
        dropout=CONFIG['dropout']
    )
    
    # Set tokenizer special tokens
    model.pad_token = tokenizer.word2idx['<pad>']
    model.sos_token = tokenizer.word2idx['<sos>']
    model.eos_token = tokenizer.word2idx['<eos>']
    model.unk_token = tokenizer.word2idx['<unk>']
    
    model = model.to(CONFIG['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 3. Setup training
    print("\n3. Setting up training...")
    
    # Loss function (ignore padding tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<pad>'])
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                          betas=(0.9, 0.98), eps=1e-9)
    
    # Learning rate scheduler
    scheduler = create_learning_rate_scheduler(
        optimizer, CONFIG['d_model'], CONFIG['warmup_steps']
    )
    
    # Test questions for generation testing
    test_questions = [
        "hi, how are you doing?",
        "what school do you go to?",
        "how's the weather today?",
        "what do you think about that?",
        "where are you from?"
    ]
    
    # 4. Training loop
    print(f"\n4. Starting training for {CONFIG['epochs']} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 30)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            scheduler, CONFIG['device'], CONFIG['max_grad_norm']
        )
        
        # Validate
        val_loss = validate_model(model, val_loader, criterion, CONFIG['device'])
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['model_path'])
            torch.save(tokenizer, CONFIG['tokenizer_path'])
            print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Test generation every few epochs
        if (epoch + 1) % CONFIG['save_every'] == 0:
            test_generation(model, tokenizer, CONFIG['device'], test_questions)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CONFIG['model_path']}")
    print(f"Tokenizer saved to: {CONFIG['tokenizer_path']}")

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                   tokenizer, checkpoint_path):
    """
    Save training checkpoint with all necessary information.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        tokenizer: Tokenizer object
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'tokenizer': tokenizer,
        'config': CONFIG
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """
    Load training checkpoint and resume training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        
    Returns:
        tuple: (start_epoch, best_val_loss, tokenizer)
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf'), None
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']
    tokenizer = checkpoint['tokenizer']
    
    print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    return start_epoch, best_val_loss, tokenizer

def log_training_progress(epoch, train_loss, val_loss, epoch_time, best_val_loss, log_file):
    """
    Log training progress to file.
    
    Args:
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        epoch_time: Time taken for epoch
        best_val_loss: Best validation loss so far
        log_file: Path to log file
    """
    log_entry = f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | " \
                f"Time: {epoch_time:6.2f}s | Best: {best_val_loss:.4f}\n"
    
    with open(log_file, 'a') as f:
        f.write(log_entry)

def print_training_stats(model, train_loader, val_loader):
    """
    Print detailed training statistics.
    
    Args:
        model: The model
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Data statistics
    print(f"\nDataset Statistics:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Max sequence length: {CONFIG['max_len']}")
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Warmup steps: {CONFIG['warmup_steps']}")
    print(f"  Max gradient norm: {CONFIG['max_grad_norm']}")
    print(f"  Device: {CONFIG['device']}")
    
    print("="*60)

def estimate_training_time(train_loader, epochs, sample_batches=5):
    """
    Estimate total training time based on sample batches.
    
    Args:
        train_loader: Training data loader
        epochs: Number of epochs
        sample_batches: Number of batches to sample for timing
        
    Returns:
        float: Estimated training time in hours
    """
    print(f"\nEstimating training time using {sample_batches} sample batches...")
    
    # Sample timing (this would need actual model for accurate timing)
    # For now, provide a rough estimate based on typical transformer training
    batches_per_epoch = len(train_loader)
    
    # Rough estimate: ~0.5 seconds per batch for this size model
    estimated_seconds_per_batch = 0.5
    
    total_batches = batches_per_epoch * epochs
    total_seconds = total_batches * estimated_seconds_per_batch
    total_hours = total_seconds / 3600
    
    print(f"Estimated training time: {total_hours:.1f} hours ({total_seconds/60:.0f} minutes)")
    return total_hours

# Enhanced training function with better progress tracking
def train_conversational_model_enhanced():
    """
    Enhanced training function with checkpointing and detailed progress tracking.
    """
    print("Starting Enhanced Conversational Transformer Training")
    print(f"Device: {CONFIG['device']}")
    
    # Create log file
    log_file = 'training_log.txt'
    with open(log_file, 'w') as f:
        f.write("Epoch | Train Loss | Val Loss | Time | Best Val Loss\n")
        f.write("-" * 55 + "\n")
    
    # 1. Prepare data (same as before)
    print("\n1. Loading and preparing data...")
    
    with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = SimpleTokenizer(max_vocab_size=8000)
    tokenizer.fit([text])
    vocab_size = tokenizer.vocab_size()
    
    dataset = ConversationalDataset(CONFIG['data_path'], tokenizer, CONFIG['max_len'])
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 2. Create model (same as before)
    print("\n2. Creating model...")
    model = ConversationalTransformer(
        vocab_size=vocab_size,
        d_model=CONFIG['d_model'],
        n_layers=CONFIG['n_layers'],
        n_heads=CONFIG['n_heads'],
        d_ff=CONFIG['d_ff'],
        max_len=CONFIG['max_len'],
        dropout=CONFIG['dropout']
    )
    
    model.pad_token = tokenizer.word2idx['<pad>']
    model.sos_token = tokenizer.word2idx['<sos>']
    model.eos_token = tokenizer.word2idx['<eos>']
    model.unk_token = tokenizer.word2idx['<unk>']
    
    model = model.to(CONFIG['device'])
    
    # Print detailed statistics
    print_training_stats(model, train_loader, val_loader)
    
    # Estimate training time
    estimate_training_time(train_loader, CONFIG['epochs'])
    
    # 3. Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                          betas=(0.9, 0.98), eps=1e-9)
    scheduler = create_learning_rate_scheduler(optimizer, CONFIG['d_model'], CONFIG['warmup_steps'])
    
    # Try to load checkpoint
    checkpoint_path = 'checkpoint.pt'
    start_epoch, best_val_loss, loaded_tokenizer = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler
    )
    
    if loaded_tokenizer is not None:
        tokenizer = loaded_tokenizer
    
    # Test questions
    test_questions = [
        "hi, how are you doing?",
        "what school do you go to?",
        "how's the weather today?",
        "what do you think about that?",
        "where are you from?"
    ]
    
    # 4. Enhanced training loop
    print(f"\n4. Starting training from epoch {start_epoch+1}...")
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 40)
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                               scheduler, CONFIG['device'], CONFIG['max_grad_norm'])
        val_loss = validate_model(model, val_loader, criterion, CONFIG['device'])
        
        epoch_time = time.time() - start_time
        
        # Log progress
        log_training_progress(epoch+1, train_loss, val_loss, epoch_time, best_val_loss, log_file)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['model_path'])
            torch.save(tokenizer, CONFIG['tokenizer_path'])
            print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                       tokenizer, checkpoint_path)
        
        # Test generation
        if (epoch + 1) % CONFIG['save_every'] == 0:
            test_generation(model, tokenizer, CONFIG['device'], test_questions)
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CONFIG['model_path']}")
    print(f"Training log: {log_file}")

# Add option to use enhanced training
if __name__ == "__main__":
    # Use enhanced training by default
    train_conversational_model_enhanced()