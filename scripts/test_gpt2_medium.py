"""Test loading GPT-2 Medium with ModelFactory"""
import sys
sys.path.insert(0, 'src')

from omegaconf import OmegaConf
from chess_model.model import ModelFactory
from chess_model.model.tokenizer import ChessTokenizer
import torch

def main():
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = ChessTokenizer.load("trained_models/tokenizer.json")
    print(f"✓ Tokenizer loaded: {tokenizer.vocab_size} tokens")

    # Load GPT-2 Medium config
    print("\nLoading GPT-2 Medium config...")
    config = OmegaConf.load("configs/model/gpt2_medium.yaml")
    print(f"✓ Config loaded: {config.architecture}, {config.base_model_name}")

    # Create model
    print("\nCreating GPT-2 Medium model...")
    print("(This will download ~1.5GB if not cached)")
    model = ModelFactory.create_from_config(config, tokenizer)

    # Check model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✓ Model created successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Architecture: gpt2-medium")

    # Test forward pass
    print("\nTesting forward pass...")
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 10))
    with torch.no_grad():
        output = model(input_ids)
    print(f"✓ Forward pass successful!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")

    print("\n" + "="*60)
    print("GPT-2 Medium test PASSED")
    print("="*60)

if __name__ == "__main__":
    main()
