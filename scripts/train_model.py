import argparse

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

DEFAULT_MAX_LENGTH = 10
DEFAULT_NUM_EMBEDDINGS = 256
DEFAULT_NUM_EPOCHS = 10
DEFAULT_OUTPUT_FILE = "out/chess_transformer_model.pth"
DEFAULT_INITIAL_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 4


def main():
    """
    Usage: poetry run train --training-data out/training-data.csv --val-data out/validation-data.csv
    """

    parser = build_arg_parser()
    args = parser.parse_args()
    print_training_header(args)

    # Initialize tokenizer and model
    print("Initializing tokenizer...")
    tokenizer = fit_tokenizer(args.training_data)
    print(f"Tokenizer initialized with vocab_size={tokenizer.vocab_size}")
    model = ChessTransformer(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.max_length,
        n_embd=args.num_embeddings,
        n_layer=args.num_layers,
        n_head=args.num_heads,
    )

    if args.state_dict_file:
        print(f"Initializing model from {args.state_dict_file}...")
        model.load_state_dict(torch.load(args.state_dict_file))

    # Load and prepare data
    print("Loading training/validation data...")
    train_dataset = ChessDataset(
        args.training_data, tokenizer, max_length=args.max_length
    )
    val_dataset = ChessDataset(args.val_data, tokenizer, max_length=args.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

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
        num_epochs=args.num_epochs,
        learning_rate=1e-3,
        device=device,
    )

    # Save the trained model
    output_file = open(args.output_file, "wb")
    torch.save(trained_model.state_dict(), output_file)


def build_arg_parser():
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
    parser.add_argument(
        "--initial-learning-rate",
        type=float,
        help=f"The initial learning rate to use. Default: {DEFAULT_INITIAL_LEARNING_RATE}",
        required=False,
        default=DEFAULT_INITIAL_LEARNING_RATE,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"The batch size to use. Default: {DEFAULT_BATCH_SIZE}",
        required=False,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help=f"The number of layers to use in the model. Default: {DEFAULT_NUM_LAYERS}",
        required=False,
        default=DEFAULT_NUM_LAYERS,
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        help=f"The number of heads to use in the model. Default: {DEFAULT_NUM_HEADS}",
        required=False,
        default=DEFAULT_NUM_HEADS,
    )
    parser.add_argument(
        "--state-dict-file",
        type=str,
        help="The state dict file to load the initial model from. If not provided, the model will be randomly initialized.",
        required=False,
        default=None,
    )

    return parser


def print_training_header(args):
    print(
        "###################################################################################################"
    )
    print("## Training model with args:")
    print(f"Training data:          {args.training_data}")
    print(f"Validation data:        {args.val_data}")
    print(f"Output file:            {args.output_file}")
    print(f"State dict file:        {args.state_dict_file}")
    print(f"Max length:             {args.max_length}")
    print(f"Num embeddings:         {args.num_embeddings}")
    print(f"Num layers:             {args.num_layers}")
    print(f"Num heads:              {args.num_heads}")
    print(f"Num training epochs:    {args.num_epochs}")
    print(f"Initial learning rate:  {args.initial_learning_rate}")
    print(f"Batch size:             {args.batch_size}")
    print(
        "###################################################################################################"
    )
