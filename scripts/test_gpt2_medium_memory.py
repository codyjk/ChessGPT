"""Test GPT-2 Medium memory usage during training"""
import sys
sys.path.insert(0, 'src')

from chess_model.model import ModelFactory
from chess_model.model.tokenizer import ChessTokenizer
import torch
import torch.nn as nn

def get_memory_usage():
    """Get current memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    elif torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1e9
    else:
        return 0  # CPU memory is harder to track

def main():
    print("=" * 60)
    print("GPT-2 Medium Memory Usage Test")
    print("=" * 60)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    print(f"\nInitial memory: {get_memory_usage():.2f} GB")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = ChessTokenizer.load("out/chess_tokenizer.json")
    print(f"   Tokenizer loaded: {tokenizer.vocab_size} tokens")
    print(f"   Memory: {get_memory_usage():.2f} GB")

    # Create model
    print("\n2. Creating GPT-2 Medium model...")
    config = {
        "architecture": "gpt2",
        "base_model_name": "gpt2-medium",
        "max_context_length": 100,
        "num_embeddings": 1024,
        "num_layers": 24,
        "num_heads": 16,
    }
    model = ModelFactory.create_from_config(config, tokenizer)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Memory after model: {get_memory_usage():.2f} GB")

    # Create optimizer
    print("\n3. Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"   Memory after optimizer: {get_memory_usage():.2f} GB")

    # Create dummy batch
    print("\n4. Creating dummy training batch...")
    batch_size = 64
    seq_length = 50
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length)).to(device)
    labels = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length)).to(device)
    print(f"   Batch size: {batch_size}, Sequence length: {seq_length}")
    print(f"   Memory after batch: {get_memory_usage():.2f} GB")

    # Forward pass
    print("\n5. Running forward pass...")
    outputs = model(input_ids)
    print(f"   Output shape: {outputs.shape}")
    print(f"   Memory after forward: {get_memory_usage():.2f} GB")

    # Backward pass
    print("\n6. Running backward pass...")
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs.view(-1, tokenizer.vocab_size), labels.view(-1))
    loss.backward()
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Memory after backward: {get_memory_usage():.2f} GB")

    # Optimizer step
    print("\n7. Running optimizer step...")
    optimizer.step()
    optimizer.zero_grad()
    print(f"   Memory after optimizer step: {get_memory_usage():.2f} GB")

    # Summary
    print("\n" + "=" * 60)
    print("Memory Usage Summary")
    print("=" * 60)
    final_memory = get_memory_usage()
    print(f"Final memory usage: {final_memory:.2f} GB")

    if final_memory < 8:
        print("✓ Memory usage is within expected range (<8GB)")
        print("✓ Model should fit comfortably on 16GB unified memory")
    elif final_memory < 12:
        print("⚠ Memory usage is higher than expected but should work")
    else:
        print("✗ Memory usage is too high - may need optimizations")

    print("\nNote: Actual training will use gradient checkpointing")
    print("which reduces memory by ~30-40%")
    print("=" * 60)

if __name__ == "__main__":
    main()
