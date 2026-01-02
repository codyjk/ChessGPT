"""
Example: Training ChessGPT with Llama 3.2 1B and LoRA

This example demonstrates how to use the new LlamaChessTransformer
with the model factory and HuggingFace Trainer.
"""

from pathlib import Path
from omegaconf import OmegaConf

# ChessGPT imports
from chess_model.model import ModelFactory
from chess_model.model.tokenizer import ChessTokenizer
from chess_model.data import ChessDataset
from chess_model.training import (
    ChessTrainer,
    create_training_args,
    get_default_callbacks,
)


def main():
    """Main training function for Llama model."""

    # ============================================================
    # 1. Load Configuration
    # ============================================================
    print("Loading configuration...")

    # Load Llama config
    model_config = OmegaConf.load("configs/model/llama_1b.yaml")
    training_config = OmegaConf.load("configs/training/phase1_general.yaml")
    data_config = OmegaConf.load("configs/data/dataset.yaml")

    print(f"Model: {model_config.architecture}")
    print(f"Base model: {model_config.base_model_name}")
    print(f"Use LoRA: {model_config.use_lora}")
    if model_config.use_lora:
        print(f"LoRA rank: {model_config.lora_config.r}")

    # ============================================================
    # 2. Load Tokenizer
    # ============================================================
    print("\nLoading tokenizer...")
    tokenizer = ChessTokenizer.load(data_config.tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # ============================================================
    # 3. Create Model Using Factory
    # ============================================================
    print("\nCreating Llama model...")

    # Option A: Use factory with config
    model = ModelFactory.create_from_config(model_config, tokenizer)

    # Option B: Create directly
    # from chess_model.model import create_llama_chess
    # model = create_llama_chess(
    #     tokenizer=tokenizer,
    #     use_lora=True,
    #     lora_config={
    #         "r": 16,
    #         "lora_alpha": 32,
    #         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    #         "lora_dropout": 0.05,
    #     },
    # )

    print(f"\n{model}")

    # ============================================================
    # 4. Load Datasets
    # ============================================================
    print("\nLoading datasets...")

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
    print("\nSetting up training arguments...")

    training_args = create_training_args({
        "output_dir": "outputs/llama_1b",
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
    print("\n" + "=" * 60)
    print("Starting training with Llama 3.2 1B + LoRA")
    print("=" * 60 + "\n")

    trainer.train()

    # ============================================================
    # 8. Save Final Model
    # ============================================================
    print("\n" + "=" * 60)
    print("Saving final model...")
    print("=" * 60)

    output_dir = Path("outputs/llama_1b/final_model")
    model.save_pretrained(str(output_dir))
    tokenizer.save(output_dir / "tokenizer.json")

    print(f"\nModel saved to: {output_dir}")
    print("This includes:")
    print("  - LoRA adapter weights")
    print("  - VocabBridge weights")
    print("  - Chess config")
    print("  - Tokenizer")

    # ============================================================
    # 9. Final Evaluation
    # ============================================================
    print("\n" + "=" * 60)
    print("Running final evaluation...")
    print("=" * 60 + "\n")

    metrics = trainer.evaluate()

    print("Final Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nTraining complete!")
    print("\nNext steps:")
    print("  1. Test the model: poetry run play --model outputs/llama_1b/final_model")
    print("  2. Compare with GPT-2 baseline")
    print("  3. Proceed to checkmate specialization training")


def compare_architectures():
    """
    Compare GPT-2 and Llama architectures.

    This is useful for understanding the model capacity differences.
    """
    print("\n" + "=" * 60)
    print("Comparing Model Architectures")
    print("=" * 60 + "\n")

    # Load configs
    gpt2_config = OmegaConf.load("configs/model/gpt2_baseline.yaml")
    llama_config = OmegaConf.load("configs/model/llama_1b.yaml")

    # Load tokenizer
    tokenizer = ChessTokenizer.load("trained_models/tokenizer.json")

    # Compare
    comparison = ModelFactory.compare_models(
        config1=gpt2_config,
        config2=llama_config,
        tokenizer=tokenizer,
    )

    return comparison


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Just compare architectures
        compare_architectures()
    else:
        # Full training
        main()
