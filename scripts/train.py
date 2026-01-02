"""
Unified training CLI for ChessGPT.

Replaces the 5-step manual pipeline with a single command:
    poetry run train --config phase1_gpt2_medium --name gpt2-medium-v1

Features:
- Automatic model creation from config
- Dataset loading with proper splits
- Training with HuggingFace Trainer
- Automatic checkpointing and logging
- Model registry integration
- Metadata tracking (hyperparams, metrics, timestamps)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, 'src')

from omegaconf import OmegaConf

from chess_model.model import ModelFactory
from chess_model.model.tokenizer import ChessTokenizer
from chess_model.data import ChessDataset
from chess_model.training import ChessTrainer, create_training_args, get_default_callbacks


def load_config(config_name: str) -> dict:
    """
    Load training configuration from YAML files.

    Args:
        config_name: Name of config (e.g., "phase1_gpt2_medium")

    Returns:
        Dictionary with model_config, training_config, data_config
    """
    configs_dir = Path("configs")

    # Try to load as a complete pipeline config
    pipeline_config_path = configs_dir / "pipeline" / f"{config_name}.yaml"
    if pipeline_config_path.exists():
        print(f"Loading pipeline config: {pipeline_config_path}")
        pipeline_config = OmegaConf.load(pipeline_config_path)

        # Load referenced configs
        model_config = OmegaConf.load(
            configs_dir / "model" / f"{pipeline_config.model}.yaml"
        )
        training_config = OmegaConf.load(
            configs_dir / "training" / f"{pipeline_config.training}.yaml"
        )
        data_config = OmegaConf.load(
            configs_dir / "data" / f"{pipeline_config.data}.yaml"
        )

        return {
            "model": model_config,
            "training": training_config,
            "data": data_config,
            "pipeline": pipeline_config,
        }

    # Otherwise, try loading individual configs
    model_config_path = configs_dir / "model" / f"{config_name}.yaml"
    if model_config_path.exists():
        print(f"Loading individual model config: {model_config_path}")
        model_config = OmegaConf.load(model_config_path)

        # Try to find matching training and data configs
        training_config = OmegaConf.load(configs_dir / "training" / "default.yaml")
        data_config = OmegaConf.load(configs_dir / "data" / "dataset.yaml")

        return {
            "model": model_config,
            "training": training_config,
            "data": data_config,
        }

    raise FileNotFoundError(f"Config not found: {config_name}")


def create_output_dir(name: str, base_dir: str = "models") -> Path:
    """
    Create output directory for model.

    Args:
        name: Model name (e.g., "gpt2-medium-v1")
        base_dir: Base directory for models

    Returns:
        Path to output directory
    """
    output_dir = Path(base_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_metadata(
    output_dir: Path,
    config: dict,
    metrics: dict = None,
    training_time: float = None,
):
    """
    Save training metadata with model.

    Args:
        output_dir: Output directory
        config: Training configuration
        metrics: Final metrics
        training_time: Training time in seconds
    """
    metadata = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "model": OmegaConf.to_container(config["model"], resolve=True),
            "training": OmegaConf.to_container(config["training"], resolve=True),
            "data": OmegaConf.to_container(config["data"], resolve=True),
        },
    }

    if metrics:
        metadata["final_metrics"] = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in metrics.items()
        }

    if training_time:
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        metadata["training_time"] = f"{hours}h {minutes}m"
        metadata["training_time_seconds"] = training_time

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to {metadata_file}")


def register_model(
    name: str,
    output_dir: Path,
    config: dict,
    metrics: dict = None,
    training_time: float = None,
    tags: list = None,
):
    """
    Register model in the model registry.

    Args:
        name: Model name
        output_dir: Model output directory
        config: Training configuration
        metrics: Final metrics
        training_time: Training time in seconds
        tags: Optional tags for the model
    """
    registry_path = Path("models/registry.json")
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing registry or create new one
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    # Create entry
    entry = {
        "path": str(output_dir),
        "architecture": config["model"].get("architecture", "unknown"),
        "created_at": datetime.now().isoformat(),
        "hyperparams": {
            "max_context_length": config["model"].get("max_context_length"),
            "num_embeddings": config["model"].get("num_embeddings"),
            "num_layers": config["model"].get("num_layers"),
            "num_heads": config["model"].get("num_heads"),
            "learning_rate": config["training"].get("learning_rate"),
            "num_epochs": config["training"].get("num_epochs"),
            "batch_size": config["training"].get("batch_size"),
        },
        "tags": tags or [],
        "recommended": False,
    }

    if metrics:
        entry["metrics"] = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in metrics.items()
        }

    if training_time:
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        entry["training_time"] = f"{hours}h {minutes}m"

    # Add to registry
    registry[name] = entry

    # Save registry
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\n✓ Model registered in {registry_path}")
    print(f"  Use: poetry run play --model {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified training CLI for ChessGPT"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config name (e.g., phase1_gpt2_medium)",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Model name (e.g., gpt2-medium-v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Base output directory (default: models)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Tags for model registry",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Don't register model in registry",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ChessGPT Unified Training")
    print("=" * 60)

    # Load configuration
    print(f"\n1. Loading configuration: {args.config}")
    config = load_config(args.config)
    print(f"   Model: {config['model'].architecture}")
    print(f"   Training: {config['training'].num_epochs} epochs")

    # Create output directory
    print(f"\n2. Creating output directory")
    output_dir = create_output_dir(args.name, args.output_dir)
    print(f"   Output: {output_dir}")

    # Load tokenizer
    print(f"\n3. Loading tokenizer")
    tokenizer_path = config["data"].get("tokenizer_path", "out/chess_tokenizer.json")
    tokenizer = ChessTokenizer.load(tokenizer_path)
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Create model
    print(f"\n4. Creating model")
    model = ModelFactory.create_from_config(
        OmegaConf.to_container(config["model"], resolve=True),
        tokenizer
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    # Load datasets
    print(f"\n5. Loading datasets")
    train_dataset = ChessDataset(
        csv_file=config["data"].training_data_path,
        tokenizer=tokenizer,
        max_context_length=config["model"].max_context_length,
    )
    val_dataset = ChessDataset(
        csv_file=config["data"].validation_data_path,
        tokenizer=tokenizer,
        max_context_length=config["model"].max_context_length,
    )
    print(f"   Training: {len(train_dataset):,} examples")
    print(f"   Validation: {len(val_dataset):,} examples")

    # Create training arguments
    print(f"\n6. Setting up training")
    training_args = create_training_args({
        "output_dir": str(output_dir),
        "num_epochs": config["training"].num_epochs,
        "batch_size": config["training"].batch_size,
        "gradient_accumulation_steps": config["training"].gradient_accumulation_steps,
        "learning_rate": config["training"].learning_rate,
        "warmup_steps": config["training"].warmup_steps,
        "lr_scheduler": config["training"].lr_scheduler,
        "weight_decay": config["training"].weight_decay,
        "gradient_clip_norm": config["training"].gradient_clip_norm,
        "mixed_precision": config["training"].get("mixed_precision", "no"),
        "gradient_checkpointing": config["training"].get("gradient_checkpointing", False),
        "logging_steps": config["training"].logging_steps,
        "save_steps": config["training"].save_steps,
        "save_total_limit": config["training"].save_total_limit,
        "eval_steps": config["training"].eval_steps,
        "load_best_model_at_end": config["training"].load_best_model_at_end,
        "metric_for_best_model": config["training"].metric_for_best_model,
        "report_to": config["training"].get("report_to") or None,  # None disables all integrations
    })

    # Create trainer
    print(f"\n7. Creating trainer")
    callbacks = get_default_callbacks(
        config=OmegaConf.to_container(config["training"], resolve=True)
    )
    trainer = ChessTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    import time
    start_time = time.time()

    try:
        if args.resume_from:
            print(f"Resuming from: {args.resume_from}")
            trainer.train(resume_from_checkpoint=args.resume_from)
        else:
            trainer.train()

        training_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)

        # Final evaluation
        print("\n8. Running final evaluation")
        metrics = trainer.evaluate()

        print("\nFinal Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

        # Save model
        print(f"\n9. Saving model")
        trainer.save_model(output_dir / "final_model")
        tokenizer.save(output_dir / "final_model" / "tokenizer.json")
        print(f"   Saved to: {output_dir / 'final_model'}")

        # Save metadata
        save_metadata(output_dir, config, metrics, training_time)

        # Register model
        if not args.no_register:
            register_model(
                args.name,
                output_dir,
                config,
                metrics,
                training_time,
                args.tags,
            )

        print("\n" + "=" * 60)
        print("✓ Training complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Checkpoint saved - use --resume-from to continue")
        sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Training failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
