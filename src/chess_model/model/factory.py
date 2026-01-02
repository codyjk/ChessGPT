"""
Model Factory: Unified interface for creating chess models.

Supports:
- GPT-2 baseline (legacy)
- Llama 3.2 1B with LoRA (2026)
- Easy A/B testing between architectures
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path

import torch
import torch.nn as nn

try:
    from omegaconf import DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    DictConfig = None

from .transformer import ChessTransformer  # GPT-2 baseline
from .llama_chess import LlamaChessTransformer
from .tokenizer import ChessTokenizer


class ModelFactory:
    """
    Factory for creating chess models.

    Usage:
        # From config
        model = ModelFactory.create_from_config(config, tokenizer)

        # Programmatically
        model = ModelFactory.create_model(
            architecture="llama",
            vocab_size=287,
            use_lora=True,
        )

        # Load pretrained
        model = ModelFactory.load_pretrained("outputs/checkpoint-1000")
    """

    @staticmethod
    def create_model(
        architecture: str,
        vocab_size: int,
        max_context_length: int = 100,
        **kwargs,
    ) -> nn.Module:
        """
        Create a chess model from scratch.

        Args:
            architecture: Model architecture ("gpt2", "llama")
            vocab_size: Vocabulary size
            max_context_length: Maximum sequence length
            **kwargs: Architecture-specific arguments

        Returns:
            model: Initialized chess model

        Raises:
            ValueError: If architecture is not supported
        """
        architecture = architecture.lower()

        if architecture == "gpt2":
            return ModelFactory._create_gpt2(
                vocab_size=vocab_size,
                max_context_length=max_context_length,
                **kwargs,
            )
        elif architecture == "llama":
            return ModelFactory._create_llama(
                vocab_size=vocab_size,
                max_context_length=max_context_length,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Supported: 'gpt2', 'llama'"
            )

    @staticmethod
    def _create_gpt2(
        vocab_size: int,
        max_context_length: int,
        num_embeddings: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        **kwargs,
    ) -> ChessTransformer:
        """
        Create GPT-2 baseline model.

        Args:
            vocab_size: Chess vocabulary size
            max_context_length: Maximum sequence length
            num_embeddings: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads

        Returns:
            model: GPT-2 chess model
        """
        print(f"Creating GPT-2 model: vocab={vocab_size}, layers={num_layers}")

        model = ChessTransformer(
            vocab_size=vocab_size,
            n_positions=max_context_length,
            n_embd=num_embeddings,
            n_layer=num_layers,
            n_head=num_heads,
        )

        return model

    @staticmethod
    def _create_llama(
        vocab_size: int,
        max_context_length: int,
        base_model_name: str = "meta-llama/Llama-3.2-1B",
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        pad_token_id: Optional[int] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> LlamaChessTransformer:
        """
        Create Llama chess model with LoRA.

        Args:
            vocab_size: Chess vocabulary size
            max_context_length: Maximum sequence length (not used for Llama directly)
            base_model_name: HuggingFace model identifier
            use_lora: Whether to use LoRA adapters
            lora_config: LoRA configuration
            pad_token_id: Padding token ID
            device_map: Device placement strategy
            torch_dtype: Model dtype

        Returns:
            model: Llama chess model
        """
        print(f"Creating Llama model: {base_model_name}, LoRA={use_lora}")

        model = LlamaChessTransformer(
            chess_vocab_size=vocab_size,
            base_model_name=base_model_name,
            use_lora=use_lora,
            lora_config=lora_config,
            pad_token_id=pad_token_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        return model

    @staticmethod
    def create_from_config(
        config: Union[Dict, "DictConfig"],
        tokenizer: ChessTokenizer,
    ) -> nn.Module:
        """
        Create model from configuration.

        Args:
            config: Model configuration (dict or OmegaConf DictConfig)
            tokenizer: Chess tokenizer (for vocab size)

        Returns:
            model: Initialized chess model
        """
        # Convert OmegaConf to dict if needed
        if OMEGACONF_AVAILABLE and isinstance(config, DictConfig):
            from omegaconf import OmegaConf
            config = OmegaConf.to_container(config, resolve=True)

        # Extract architecture
        architecture = config.get("architecture", "gpt2")
        vocab_size = tokenizer.vocab_size

        # Extract common parameters
        max_context_length = config.get("max_context_length", 100)

        # Architecture-specific parameters
        if architecture == "gpt2":
            kwargs = {
                "num_embeddings": config.get("num_embeddings", 512),
                "num_layers": config.get("num_layers", 6),
                "num_heads": config.get("num_heads", 8),
            }
        elif architecture == "llama":
            kwargs = {
                "base_model_name": config.get("base_model_name", "meta-llama/Llama-3.2-1B"),
                "use_lora": config.get("use_lora", True),
                "pad_token_id": tokenizer.pad_token_id,
            }

            # LoRA config
            if "lora_config" in config:
                lora_cfg = config["lora_config"]
                kwargs["lora_config"] = {
                    "r": lora_cfg.get("r", 16),
                    "lora_alpha": lora_cfg.get("lora_alpha", 32),
                    "target_modules": lora_cfg.get(
                        "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
                    ),
                    "lora_dropout": lora_cfg.get("lora_dropout", 0.05),
                }
        else:
            kwargs = {}

        return ModelFactory.create_model(
            architecture=architecture,
            vocab_size=vocab_size,
            max_context_length=max_context_length,
            **kwargs,
        )

    @staticmethod
    def load_pretrained(
        model_path: Union[str, Path],
        architecture: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        """
        Load pretrained model from checkpoint.

        Args:
            model_path: Path to checkpoint directory
            architecture: Model architecture (auto-detected if None)
            device_map: Device placement strategy
            torch_dtype: Model dtype

        Returns:
            model: Loaded model

        Raises:
            ValueError: If architecture cannot be determined
        """
        model_path = Path(model_path)

        # Auto-detect architecture if not provided
        if architecture is None:
            architecture = ModelFactory._detect_architecture(model_path)

        print(f"Loading {architecture} model from {model_path}")

        if architecture == "gpt2":
            # Load GPT-2 checkpoint
            model = torch.load(model_path / "model.pt", map_location="cpu")
            return model

        elif architecture == "llama":
            # Load Llama checkpoint
            model = LlamaChessTransformer.from_pretrained(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
            return model

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    @staticmethod
    def _detect_architecture(model_path: Path) -> str:
        """
        Auto-detect model architecture from checkpoint.

        Args:
            model_path: Path to checkpoint directory

        Returns:
            architecture: Detected architecture ("gpt2" or "llama")
        """
        # Check for chess_config.json (Llama)
        if (model_path / "chess_config.json").exists():
            return "llama"

        # Check for adapter_config.json (LoRA)
        if (model_path / "adapter_config.json").exists():
            return "llama"

        # Check for model.pt (GPT-2)
        if (model_path / "model.pt").exists():
            return "gpt2"

        # Default to GPT-2
        print("Warning: Could not detect architecture, defaulting to gpt2")
        return "gpt2"

    @staticmethod
    def compare_models(
        config1: Dict,
        config2: Dict,
        tokenizer: ChessTokenizer,
    ) -> Dict[str, Any]:
        """
        Compare two model configurations.

        Useful for A/B testing and architecture selection.

        Args:
            config1: First model configuration
            config2: Second model configuration
            tokenizer: Chess tokenizer

        Returns:
            comparison: Dict with parameter counts and architecture details
        """
        print("Creating models for comparison...")

        model1 = ModelFactory.create_from_config(config1, tokenizer)
        model2 = ModelFactory.create_from_config(config2, tokenizer)

        comparison = {
            "model1": {
                "architecture": config1.get("architecture", "gpt2"),
                "total_params": sum(p.numel() for p in model1.parameters()),
                "trainable_params": sum(
                    p.numel() for p in model1.parameters() if p.requires_grad
                ),
            },
            "model2": {
                "architecture": config2.get("architecture", "gpt2"),
                "total_params": sum(p.numel() for p in model2.parameters()),
                "trainable_params": sum(
                    p.numel() for p in model2.parameters() if p.requires_grad
                ),
            },
        }

        # Calculate ratios
        comparison["param_ratio"] = (
            comparison["model2"]["total_params"]
            / comparison["model1"]["total_params"]
        )

        comparison["trainable_ratio"] = (
            comparison["model2"]["trainable_params"]
            / comparison["model1"]["trainable_params"]
        )

        # Print comparison
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)
        print(f"Model 1 ({comparison['model1']['architecture']}):")
        print(f"  Total params: {comparison['model1']['total_params']:,}")
        print(f"  Trainable: {comparison['model1']['trainable_params']:,}")
        print(f"\nModel 2 ({comparison['model2']['architecture']}):")
        print(f"  Total params: {comparison['model2']['total_params']:,}")
        print(f"  Trainable: {comparison['model2']['trainable_params']:,}")
        print(f"\nRatios:")
        print(f"  Total params: {comparison['param_ratio']:.2f}x")
        print(f"  Trainable params: {comparison['trainable_ratio']:.2f}x")
        print("=" * 60 + "\n")

        return comparison


# Convenience functions
def create_gpt2_baseline(
    tokenizer: ChessTokenizer,
    max_context_length: int = 100,
    **kwargs,
) -> ChessTransformer:
    """
    Convenience function to create GPT-2 baseline.

    Args:
        tokenizer: Chess tokenizer
        max_context_length: Maximum sequence length
        **kwargs: Additional GPT-2 parameters

    Returns:
        model: GPT-2 chess model
    """
    return ModelFactory.create_model(
        architecture="gpt2",
        vocab_size=tokenizer.vocab_size,
        max_context_length=max_context_length,
        **kwargs,
    )


def create_llama_chess(
    tokenizer: ChessTokenizer,
    use_lora: bool = True,
    **kwargs,
) -> LlamaChessTransformer:
    """
    Convenience function to create Llama chess model.

    Args:
        tokenizer: Chess tokenizer
        use_lora: Whether to use LoRA adapters
        **kwargs: Additional Llama parameters

    Returns:
        model: Llama chess model
    """
    return ModelFactory.create_model(
        architecture="llama",
        vocab_size=tokenizer.vocab_size,
        use_lora=use_lora,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )
