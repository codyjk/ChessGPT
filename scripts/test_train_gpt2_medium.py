"""Quick sanity check: Train GPT-2 Medium for 1 epoch on 1K games"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
from chess_model.model import ModelFactory
from chess_model.model.tokenizer import ChessTokenizer
from chess_model.data import ChessDataset
from chess_model.training import ChessTrainer, create_training_args, get_default_callbacks


def main():
    print("=" * 60)
    print("GPT-2 Medium Sanity Check Training")
    print("1 epoch on 1K games")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = ChessTokenizer.load("out/chess_tokenizer.json")
    print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")

    # Create GPT-2 Medium model
    print("\n2. Creating GPT-2 Medium model...")
    config = {
        "architecture": "gpt2",
        "base_model_name": "gpt2-medium",
        "max_context_length": 100,
        "num_embeddings": 1024,
        "num_layers": 24,
        "num_heads": 16,
    }
    model = ModelFactory.create_from_config(config, tokenizer)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")

    # Load datasets
    print("\n3. Loading datasets...")
    train_dataset = ChessDataset(
        csv_file="data/phase_splits/training_test_1k.csv",
        tokenizer=tokenizer,
        max_context_length=100,
    )
    val_dataset = ChessDataset(
        csv_file="data/phase_splits/validation_test_1k.csv",
        tokenizer=tokenizer,
        max_context_length=100,
    )
    print(f"   Training examples: {len(train_dataset):,}")
    print(f"   Validation examples: {len(val_dataset):,}")

    # Create training arguments
    print("\n4. Setting up training arguments...")
    training_args = create_training_args({
        "output_dir": "outputs/gpt2_medium_test",
        "num_epochs": 1,
        "batch_size": 4,  # Small batch for testing
        "gradient_accumulation_steps": 16,  # Effective batch size: 64
        "learning_rate": 3e-4,
        "warmup_steps": 100,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "mixed_precision": "bf16",  # Use bfloat16 for M1/M2
        "gradient_checkpointing": True,  # Essential for memory
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 2,
        "eval_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "report_to": [],  # Disable WandB for test
    })

    # Create trainer
    print("\n5. Creating trainer...")
    callbacks = get_default_callbacks()
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

    try:
        trainer.train()

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)

        # Final evaluation
        print("\nRunning final evaluation...")
        metrics = trainer.evaluate()

        print("\nFinal Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

        # Save model
        output_dir = Path("outputs/gpt2_medium_test/final_model")
        trainer.save_model(output_dir)
        tokenizer.save(output_dir / "tokenizer.json")
        print(f"\nModel saved to: {output_dir}")

        print("\n" + "=" * 60)
        print("✓ Sanity check PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review metrics and memory usage")
        print("2. Increase dataset size gradually")
        print("3. Build unified training CLI")
        print("4. Run full Phase 1 training")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Training failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("- Check memory usage (should be <8GB)")
        print("- Verify dataset format")
        print("- Try smaller batch size")


if __name__ == "__main__":
    main()
