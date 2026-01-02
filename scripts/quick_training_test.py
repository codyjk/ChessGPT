"""
Quick training test to validate the pipeline.

Uses small model, small dataset, and 2 epochs for rapid testing.
"""

from pathlib import Path
from omegaconf import OmegaConf

from chess_model.model import ModelFactory
from chess_model.model.tokenizer import ChessTokenizer
from chess_model.data import ChessDataset
from chess_model.training import (
    ChessTrainer,
    create_training_args,
    get_default_callbacks,
)


def main():
    """Quick training test."""

    print("=" * 60)
    print("Quick Training Test - Pipeline Validation")
    print("=" * 60 + "\n")

    # Load configs
    print("Loading configuration...")
    model_config = OmegaConf.load("configs/model/gpt2_quick_test.yaml")
    training_config = OmegaConf.load("configs/training/quick_test.yaml")
    data_config = OmegaConf.load("configs/data/dataset.yaml")

    print(f"Model: {model_config.architecture}")
    print(f"Layers: {model_config.num_layers}")
    print(f"Embeddings: {model_config.num_embeddings}")
    print(f"Epochs: {training_config.num_epochs}")
    print(f"Batch size: {training_config.batch_size}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = ChessTokenizer.load(data_config.tokenizer_path)
    print(f"Vocab size: {tokenizer.vocab_size}\n")

    # Create model
    print("Creating model...")
    model = ModelFactory.create_from_config(model_config, tokenizer)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = ChessDataset(
        csv_file=data_config.training_data_path,
        tokenizer=tokenizer,
        max_context_length=data_config.max_context_length,
    )
    val_dataset = ChessDataset(
        csv_file=data_config.validation_data_path,
        tokenizer=tokenizer,
        max_context_length=data_config.max_context_length,
    )
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}\n")

    # Create training arguments
    print("Setting up training...")
    training_args = create_training_args({
        "output_dir": "outputs/quick_test",
        "num_epochs": training_config.num_epochs,
        "batch_size": training_config.batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "learning_rate": training_config.learning_rate,
        "warmup_steps": training_config.warmup_steps,
        "lr_scheduler": training_config.lr_scheduler,
        "weight_decay": training_config.weight_decay,
        "gradient_clip_norm": training_config.gradient_clip_norm,
        "mixed_precision": None,  # Disable for compatibility
        "gradient_checkpointing": training_config.gradient_checkpointing,
        "logging_steps": training_config.logging_steps,
        "save_steps": training_config.save_steps,
        "save_total_limit": training_config.save_total_limit,
        "eval_steps": training_config.eval_steps,
        "load_best_model_at_end": training_config.load_best_model_at_end,
        "metric_for_best_model": training_config.metric_for_best_model,
        "report_to": training_config.report_to,
    })

    # Create trainer
    callbacks = get_default_callbacks(config={})

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

    trainer.train()

    # Evaluate
    print("\n" + "=" * 60)
    print("Final evaluation...")
    print("=" * 60 + "\n")

    metrics = trainer.evaluate()

    print("Final metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Save
    output_dir = Path("outputs/quick_test/final_model")
    trainer.save_model(str(output_dir))
    tokenizer.save(output_dir / "tokenizer.json")

    print(f"\n✓ Model saved to: {output_dir}")
    print("\n" + "=" * 60)
    print("Quick training test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
