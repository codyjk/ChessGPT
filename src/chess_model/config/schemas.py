"""Type-safe configuration schemas using Pydantic."""

from pathlib import Path
from typing import Literal, Optional, List

from pydantic import BaseModel, Field, field_validator


class LoRAConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""

    r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(
        default=32, ge=1, description="LoRA scaling factor (typically 2*r)"
    )
    target_modules: List[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Target modules for LoRA",
    )
    lora_dropout: float = Field(
        default=0.05, ge=0.0, le=0.5, description="Dropout for LoRA layers"
    )

    @field_validator("lora_alpha")
    @classmethod
    def validate_alpha(cls, v: int, info) -> int:
        """Validate that lora_alpha is typically 2*r."""
        if "r" in info.data and v < info.data["r"]:
            raise ValueError(f"lora_alpha ({v}) should be >= r ({info.data['r']})")
        return v


class ModelConfig(BaseModel):
    """Configuration for the chess model architecture."""

    architecture: Literal["gpt2", "llama"] = Field(
        default="llama", description="Model architecture to use"
    )
    base_model_name: str = Field(
        default="meta-llama/Llama-3.2-1B",
        description="HuggingFace model name or 'gpt2' for legacy",
    )
    vocab_size: Optional[int] = Field(
        default=None, description="Vocabulary size (set from tokenizer)"
    )
    max_context_length: int = Field(
        default=100, ge=10, le=2048, description="Maximum context length (positions)"
    )

    # GPT-2 specific params (legacy)
    num_embeddings: int = Field(default=512, ge=64, le=2048, description="GPT-2 only")
    num_layers: int = Field(default=4, ge=1, le=24, description="GPT-2 only")
    num_heads: int = Field(default=4, ge=1, le=32, description="GPT-2 only")

    # LoRA configuration
    use_lora: bool = Field(default=True, description="Whether to use LoRA adapters")
    lora_config: Optional[LoRAConfig] = Field(
        default_factory=LoRAConfig, description="LoRA configuration"
    )

    # Training precision
    use_bf16: bool = Field(default=True, description="Use bfloat16 precision")
    use_fp16: bool = Field(default=False, description="Use float16 precision")

    @field_validator("use_fp16")
    @classmethod
    def validate_precision(cls, v: bool, info) -> bool:
        """Ensure only one precision mode is enabled."""
        if v and info.data.get("use_bf16", False):
            raise ValueError("Cannot use both bf16 and fp16")
        return v


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""

    # Basic training params
    num_epochs: int = Field(default=8, ge=1, le=100, description="Number of epochs")
    batch_size: int = Field(default=64, ge=1, le=512, description="Batch size")
    gradient_accumulation_steps: int = Field(
        default=2, ge=1, le=64, description="Gradient accumulation steps"
    )

    # Learning rate
    learning_rate: float = Field(
        default=3e-4, gt=0.0, lt=1.0, description="Initial learning rate"
    )
    warmup_steps: int = Field(
        default=1000, ge=0, description="Number of warmup steps"
    )
    lr_scheduler: Literal["cosine", "linear", "constant", "constant_with_warmup"] = (
        Field(default="cosine", description="Learning rate scheduler")
    )

    # Regularization
    weight_decay: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Weight decay (L2 regularization)"
    )
    gradient_clip_norm: float = Field(
        default=1.0, ge=0.0, description="Max gradient norm (0 = no clipping)"
    )

    # Mixed precision
    mixed_precision: Literal["no", "fp16", "bf16"] = Field(
        default="bf16", description="Mixed precision training"
    )
    gradient_checkpointing: bool = Field(
        default=True, description="Use gradient checkpointing to save memory"
    )

    # Checkpointing
    save_steps: int = Field(
        default=500, ge=0, description="Save checkpoint every N steps (0 = per epoch)"
    )
    save_total_limit: int = Field(
        default=3, ge=1, description="Maximum number of checkpoints to keep"
    )
    load_best_model_at_end: bool = Field(
        default=True, description="Load best checkpoint at end of training"
    )

    # Evaluation
    eval_steps: int = Field(
        default=500, ge=0, description="Evaluate every N steps (0 = per epoch)"
    )
    metric_for_best_model: str = Field(
        default="eval_loss", description="Metric to use for best model selection"
    )

    # Logging
    logging_steps: int = Field(default=10, ge=1, description="Log every N steps")
    report_to: List[str] = Field(
        default=["wandb"], description="Where to report metrics"
    )

    # Checkmate-specific training params
    checkmate_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Ratio of checkmate examples in training data",
    )
    checkmate_weight: float = Field(
        default=1.0, ge=0.0, description="Loss weight for checkmate examples"
    )

    # Phase information
    phase: int = Field(default=1, ge=1, le=2, description="Training phase (1 or 2)")
    resume_from: Optional[str] = Field(
        default=None, description="Path to checkpoint to resume from"
    )


class DataConfig(BaseModel):
    """Configuration for data loading and processing."""

    # Data paths
    training_data_path: Path = Field(
        default=Path("data/general_games_training.csv"),
        description="Path to training data CSV",
    )
    checkmate_data_path: Optional[Path] = Field(
        default=Path("data/checkmate_games_training.csv"),
        description="Path to checkmate training data CSV",
    )
    validation_data_path: Path = Field(
        default=Path("data/validation.csv"), description="Path to validation data CSV"
    )
    tokenizer_path: Path = Field(
        default=Path("data/chess_tokenizer.json"), description="Path to tokenizer file"
    )

    # Data processing
    max_context_length: int = Field(
        default=100, ge=10, le=2048, description="Maximum context length"
    )
    validation_split: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Validation data split ratio"
    )

    # Data mixing (for two-phase training)
    use_mixed_dataset: bool = Field(
        default=True, description="Whether to mix regular and checkmate data"
    )
    checkmate_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Ratio of checkmate examples in mixed dataset",
    )

    @field_validator("training_data_path", "validation_data_path", "tokenizer_path")
    @classmethod
    def validate_paths_exist(cls, v: Path) -> Path:
        """Validate that required data files exist (can be skipped during config creation)."""
        # Note: We don't enforce existence here to allow config creation before data prep
        return v


class PipelineConfig(BaseModel):
    """Configuration for the end-to-end training pipeline."""

    # Pipeline metadata
    name: str = Field(
        default="chessgpt_2026", description="Name of the training run"
    )
    output_dir: Path = Field(
        default=Path("outputs"), description="Output directory for models/logs"
    )

    # Auto-download settings
    auto_download: bool = Field(
        default=False,
        description="Automatically download and process PGN data if missing",
    )
    pgn_sources: List[dict] = Field(
        default_factory=list, description="List of PGN sources to download"
    )

    # Component configs
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training_phase1: TrainingConfig = Field(
        default_factory=lambda: TrainingConfig(phase=1, checkmate_ratio=0.05)
    )
    training_phase2: TrainingConfig = Field(
        default_factory=lambda: TrainingConfig(
            phase=2,
            num_epochs=4,
            learning_rate=1e-5,
            checkmate_ratio=0.70,
            checkmate_weight=2.0,
        )
    )

    # Evaluation settings
    run_full_eval: bool = Field(
        default=True, description="Run full evaluation after training"
    )
    test_set_path: Optional[Path] = Field(
        default=None, description="Path to test set for final evaluation"
    )
