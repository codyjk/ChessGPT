"""CLI: Train the chess transformer."""

import argparse
import shutil
import tomllib
from pathlib import Path

from torch.utils.data import DataLoader

from chessgpt.data.dataset import ChessDataset
from chessgpt.model.tokenizer import ChessTokenizer
from chessgpt.model.transformer import ChessTransformer, TransformerConfig
from chessgpt.training.trainer import get_device, train


def main():
    parser = argparse.ArgumentParser(description="Train ChessGPT model")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="out", help="Base output directory")
    parser.add_argument(
        "--log-style",
        type=str,
        choices=["tqdm", "line"],
        default="tqdm",
        help="Logging style: tqdm (progress bars) or line (newline-based, better for tmux/logs)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to output for reproducibility
    shutil.copy2(args.config, output_dir / "config.toml")

    # Load tokenizer
    tokenizer = ChessTokenizer.load(config["data"]["tokenizer_file"])
    print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")

    # Create model
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        max_seq_len=config["model"]["max_context_length"],
        dropout=config["model"].get("dropout", 0.0),
    )
    model = ChessTransformer(model_config)
    print(f"Model parameters: ~{model_config.param_count_estimate:,}")

    # Load datasets
    train_dataset = ChessDataset(
        config["data"]["training_file"],
        tokenizer,
        max_context_length=config["model"]["max_context_length"],
        checkmate_weight=config["training"].get("checkmate_weight", 5.0),
    )
    print(f"Training examples: {len(train_dataset)}")

    val_dataset = None
    val_loader = None
    if "validation_file" in config["data"]:
        val_dataset = ChessDataset(
            config["data"]["validation_file"],
            tokenizer,
            max_context_length=config["model"]["max_context_length"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        print(f"Validation examples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # Save tokenizer to output early (so eval works even if training is interrupted)
    tokenizer.save(str(output_dir / "tokenizer.json"))

    # Train
    device = get_device()
    print(f"Device: {device}")

    training_config = {
        "lr": config["training"]["lr"],
        "num_epochs": config["training"]["num_epochs"],
        "grad_clip": config["training"].get("grad_clip", 1.0),
        "alpha": config["training"].get("alpha", 0.5),
        "beta": config["training"].get("beta", 0.5),
        "accumulation_steps": config["training"].get("accumulation_steps", 1),
        "patience": config["training"].get("patience", 0),
    }

    train(
        model,
        train_loader,
        val_loader,
        training_config,
        device,
        str(output_dir),
        log_style=args.log_style,
    )

    print(f"\nExperiment saved to: {output_dir}")
