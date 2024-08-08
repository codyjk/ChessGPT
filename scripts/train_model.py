import argparse

import torch

from chess_model import (
    ChessDataset,
    ChessTransformer,
    calculate_random_baseline,
    fit_tokenizer,
    get_device,
    train_model,
)

DEFAULT_MAX_LENGTH = 50
DEFAULT_NUM_EMBEDDINGS = 64
DEFAULT_NUM_EPOCHS = 5
DEFAULT_OUTPUT_FILE = "out/chess_transformer_model.pth"


def main():
    """
    Usage: poetry run train --training-data out/training-data.csv --val-data out/validation-data.csv
    """

    parser = argparse.ArgumentParser(description="Train the LLM.")
    parser.add_argument(
        "--training-data",
        type=str,
        help="The input training data file, as returned by `poetry run prepare-training-data`",
        required=True,
    )
    parser.add_argument(
        "--val-data",
        type=str,
        help="The input validation data file, as returned by `poetry run prepare-training-data`",
        required=True,
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help=f"Where to save the pickle file for the trained model. Default: {DEFAULT_OUTPUT_FILE}",
        required=False,
        default=DEFAULT_OUTPUT_FILE,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help=f"The maximum context length (number of moves) to train against. Default: {DEFAULT_MAX_LENGTH}",
        required=False,
        default=DEFAULT_MAX_LENGTH,
    )
    parser.add_argument(
        "--num-embeddings",
        type=int,
        help=f"The number of embeddings to use in the model. Default: {DEFAULT_NUM_EMBEDDINGS}",
        required=False,
        default=DEFAULT_NUM_EMBEDDINGS,
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help=f"The number of epochs to train the model for. Default: {DEFAULT_NUM_EPOCHS}",
        required=False,
        default=DEFAULT_NUM_EPOCHS,
    )
    args = parser.parse_args()

    # Initialize tokenizer and model
    print("Initializing tokenizer...")
    tokenizer = fit_tokenizer(args.training_data)
    print(f"Tokenizer initialized with vocab_size={tokenizer.vocab_size}")
    model = ChessTransformer(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.max_length,
        n_embd=args.num_embeddings,
    )

    # Load and prepare data
    print("Loading training/validation data...")
    train_dataset = ChessDataset(
        args.training_data, tokenizer, max_length=args.max_length
    )
    val_dataset = ChessDataset(args.val_data, tokenizer, max_length=args.max_length)

    # Get the appropriate device
    device = get_device()
    print(f"Using device: {device}")

    # Calculate random baseline loss
    random_baseline_loss = calculate_random_baseline(
        train_dataset, model.config.vocab_size, device
    )
    print(f"Random Baseline Loss: {random_baseline_loss:.4f}")

    # Train the model
    trained_model = train_model(
        model,
        train_dataset,
        val_dataset,
        num_epochs=args.num_epochs,
        learning_rate=1e-3,
        device=device,
    )

    # Save the trained model
    output_file = open(args.output_file, "wb")
    torch.save(trained_model.state_dict(), output_file)
