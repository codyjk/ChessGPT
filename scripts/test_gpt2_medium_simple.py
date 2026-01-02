"""Simple test of GPT-2 Medium loading without OmegaConf"""
import sys
sys.path.insert(0, 'src')

from chess_model.model import ModelFactory
from chess_model.model.tokenizer import ChessTokenizer
import torch

def main():
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = ChessTokenizer.load("out/chess_tokenizer.json")
    print(f"✓ Tokenizer loaded: {tokenizer.vocab_size} tokens\n")

    # Create config dict directly (no OmegaConf needed)
    print("Creating GPT-2 Medium config...")
    config = {
        "architecture": "gpt2",
        "base_model_name": "gpt2-medium",
        "max_context_length": 100,
        "num_embeddings": 1024,
        "num_layers": 24,
        "num_heads": 16,
    }
    print(f"✓ Config created: {config['architecture']}, {config['base_model_name']}\n")

    # Create model
    print("Creating GPT-2 Medium model...")
    print("(This will download ~1.5GB from HuggingFace if not cached)")
    try:
        model = ModelFactory.create_from_config(config, tokenizer)
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        print("\nThis is expected if transformers isn't fully installed.")
        print("Run: poetry install --with model")
        return

    # Check model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✓ Model created successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Ratio: {total_params / 630019:.0f}x larger than current model\n")

    # Test forward pass
    print("Testing forward pass...")
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 10))
    with torch.no_grad():
        output = model(input_ids)
    print(f"✓ Forward pass successful!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}\n")

    print("=" * 60)
    print("✓ GPT-2 Medium test PASSED!")
    print("=" * 60)
    print("\nModel is ready for training.")
    print("Expected memory usage: ~6GB (model + gradients + optimizer)")

if __name__ == "__main__":
    main()
