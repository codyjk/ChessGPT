"""
Simple training test without new dependencies.

Tests the basic training pipeline with existing infrastructure.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from chess_model.model import ChessTransformer
from chess_model.model.tokenizer import ChessTokenizer
from chess_model.data import ChessDataset
from chess_model.training import train_model


def main():
    """Simple training test."""

    print("=" * 60)
    print("Simple Training Test - Pipeline Validation")
    print("=" * 60 + "\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = ChessTokenizer.load("trained_models/tokenizer.json")
    print(f"Vocab size: {tokenizer.vocab_size}\n")

    # Create model (smaller for quick testing)
    print("Creating model...")
    model = ChessTransformer(
        vocab_size=tokenizer.vocab_size,
        n_positions=10,  # Match max moves
        n_embd=128,      # Smaller embeddings
        n_layer=3,       # Fewer layers
        n_head=4,        # Fewer heads
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = ChessDataset(
        csv_file="data/quick_sample/training-data.csv",
        tokenizer=tokenizer,
        max_context_length=10,
    )
    val_dataset = ChessDataset(
        csv_file="data/quick_sample/validation-data.csv",
        tokenizer=tokenizer,
        max_context_length=10,
    )
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}\n")

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}\n")

    # Train
    print("=" * 60)
    print("Starting training (5 epochs)...")
    print("=" * 60 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=5,
        learning_rate=3e-4,
        device=device,
    )

    # Save model
    output_dir = Path("outputs/simple_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), output_dir / "model.pt")
    tokenizer.save(output_dir / "tokenizer.json")
    print(f"\nModel saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("Simple training test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
