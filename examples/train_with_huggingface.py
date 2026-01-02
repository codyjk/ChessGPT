"""
Example: Training ChessGPT with HuggingFace Trainer

This example demonstrates how to use the custom ChessTrainer
with our configuration system and callbacks.
"""

from pathlib import Path
from omegaconf import OmegaConf

# ChessGPT imports
from chess_model.config.schemas import ModelConfig, TrainingConfig, DataConfig
from chess_model.model import ChessTransformer
from chess_model.model.tokenizer import ChessTokenizer
from chess_model.data import ChessDataset
from chess_model.training import (
    ChessTrainer,
    create_training_args,
    get_default_callbacks,
)


def main():
    """Main training function."""

    # ============================================================
    # 1. Load Configuration
    # ============================================================
    print("Loading configuration...")

    # Option A: Load from YAML
    model_config = OmegaConf.load("configs/model/gpt2_baseline.yaml")
    training_config = OmegaConf.load("configs/training/phase1_general.yaml")
    data_config = OmegaConf.load("configs/data/dataset.yaml")

    # Option B: Create programmatically
    # model_config = ModelConfig(
    #     architecture="gpt2",
    #     max_context_length=50,
    #     num_embeddings=512,
    #     num_layers=4,
    #     num_heads=4,
    # )

    # ============================================================
    # 2. Load Tokenizer
    # ============================================================
    print("Loading tokenizer...")
    tokenizer = ChessTokenizer.load(data_config.tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # ============================================================
    # 3. Create Model
    # ============================================================
    print("Creating model...")
    model = ChessTransformer(
        vocab_size=tokenizer.vocab_size,
        n_positions=model_config.max_context_length,
        n_embd=model_config.num_embeddings,
        n_layer=model_config.num_layers,
        n_head=model_config.num_heads,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ============================================================
    # 4. Load Datasets
    # ============================================================
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

    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(val_dataset):,}")

    # ============================================================
    # 5. Create Training Arguments
    # ============================================================
    print("Setting up training arguments...")

    training_args = create_training_args({
        "output_dir": "outputs/gpt2_baseline",
        "num_epochs": training_config.num_epochs,
        "batch_size": training_config.batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "learning_rate": training_config.learning_rate,
        "warmup_steps": training_config.warmup_steps,
        "lr_scheduler": training_config.lr_scheduler,
        "weight_decay": training_config.weight_decay,
        "gradient_clip_norm": training_config.gradient_clip_norm,
        "mixed_precision": training_config.mixed_precision,
        "gradient_checkpointing": training_config.gradient_checkpointing,
        "logging_steps": training_config.logging_steps,
        "save_steps": training_config.save_steps,
        "save_total_limit": training_config.save_total_limit,
        "eval_steps": training_config.eval_steps,
        "load_best_model_at_end": training_config.load_best_model_at_end,
        "metric_for_best_model": training_config.metric_for_best_model,
        "report_to": training_config.report_to,
    })

    # ============================================================
    # 6. Create Trainer with Callbacks
    # ============================================================
    print("Creating trainer...")

    callbacks = get_default_callbacks(
        config={"early_stopping_patience": 3}
    )

    trainer = ChessTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )

    # ============================================================
    # 7. Train!
    # ============================================================
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    trainer.train()

    # ============================================================
    # 8. Save Final Model
    # ============================================================
    print("\n" + "="*60)
    print("Saving final model...")
    print("="*60)

    output_dir = Path("outputs/gpt2_baseline/final_model")
    trainer.save_model(output_dir)
    tokenizer.save(output_dir / "tokenizer.json")

    print(f"\nModel saved to: {output_dir}")

    # ============================================================
    # 9. Final Evaluation
    # ============================================================
    print("\n" + "="*60)
    print("Running final evaluation...")
    print("="*60 + "\n")

    metrics = trainer.evaluate()

    print("Final Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
