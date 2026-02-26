"""CLI: Evaluate a trained chess model."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chessgpt.data.dataset import ChessDataset
from chessgpt.evaluation.metrics import (
    evaluate_checkmate_detection,
    evaluate_legal_move_rate,
    evaluate_move_accuracy,
    evaluate_value_accuracy,
)
from chessgpt.evaluation.puzzles import evaluate_mate_in_1
from chessgpt.model.tokenizer import ChessTokenizer
from chessgpt.model.transformer import ChessTransformer, TransformerConfig
from chessgpt.training.trainer import get_device


def load_model(model_path: str, device: torch.device) -> tuple[ChessTransformer, ChessTokenizer]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Load tokenizer from same directory as model
    model_dir = Path(model_path).parent
    tokenizer = ChessTokenizer.load(str(model_dir / "tokenizer.json"))

    model_config = TransformerConfig(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        dropout=config.get("dropout", 0.0),
    )
    model = ChessTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChessGPT model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--validation-csv", type=str, default=None, help="Validation CSV (overrides config)"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument(
        "--log-file", type=str, default="experiments/log.jsonl", help="JSONL log file"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, tokenizer = load_model(args.model, device)
    print(f"Loaded model: d_model={model.config.d_model}, n_layers={model.config.n_layers}")

    metrics = {}

    # Load validation data if available
    model_dir = Path(args.model).parent
    val_csv = args.validation_csv
    if val_csv is None:
        # Try to find validation file from config
        config_path = model_dir / "config.toml"
        if config_path.exists():
            import tomllib

            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            val_csv = config.get("data", {}).get("validation_file")

    if val_csv and Path(val_csv).exists():
        dataset = ChessDataset(val_csv, tokenizer, max_context_length=model.config.max_seq_len)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print("Evaluating move accuracy...")
        metrics.update(evaluate_move_accuracy(model, loader, device))

        print("Evaluating legal move rate...")
        metrics.update(evaluate_legal_move_rate(model, loader, tokenizer, device))

        print("Evaluating value accuracy...")
        metrics.update(evaluate_value_accuracy(model, loader, device))

        print("Evaluating checkmate detection...")
        metrics.update(evaluate_checkmate_detection(model, loader, device))
    else:
        print("No validation CSV found — skipping dataset-based metrics")

    # Mate-in-1 puzzles
    print("Evaluating mate-in-1 puzzles...")
    metrics.update(evaluate_mate_in_1(model, tokenizer, device))

    # Print results
    print("\n--- Results ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Log to JSONL
    log_entry = {
        "model": str(args.model),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "d_model": model.config.d_model,
            "n_layers": model.config.n_layers,
            "n_heads": model.config.n_heads,
            "max_seq_len": model.config.max_seq_len,
        },
        "metrics": metrics,
    }

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\nLogged to {log_path}")
