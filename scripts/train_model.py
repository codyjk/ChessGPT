import torch
from torch.utils.data import DataLoader

from chess_model import (
    ChessDataset,
    ChessTransformer,
    calculate_random_baseline,
    fit_tokenizer,
    get_device,
    train_model,
)

MAX_LEN = 16


def main():
    # Initialize tokenizer and model
    print("Initializing tokenizer...")
    tokenizer = fit_tokenizer("out/training-data.csv")
    print(f"Tokenizer initialized with vocab_size={tokenizer.vocab_size}")
    model = ChessTransformer(
        vocab_size=tokenizer.vocab_size, n_positions=MAX_LEN, n_embd=64
    )

    # Load and prepare data
    print("Loading training/validation data...")
    train_dataset = ChessDataset("out/training-data.csv", tokenizer, max_length=MAX_LEN)
    val_dataset = ChessDataset("out/validation-data.csv", tokenizer, max_length=MAX_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256)

    # Get the appropriate device
    device = get_device()
    print(f"Using device: {device}")

    # Calculate random baseline loss
    random_baseline_loss = calculate_random_baseline(
        train_dataloader, model.config.vocab_size, device
    )
    print(f"Random Baseline Loss: {random_baseline_loss:.4f}")

    # Train the model
    trained_model = train_model(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs=5,
        learning_rate=1e-3,
        device=device,
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), "chess_transformer_model.pth")
