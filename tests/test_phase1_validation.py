"""Validation tests for Phase 1 components."""

import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Test imports
from chess_model.config.schemas import (
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    DataConfig,
    PipelineConfig,
)
from chess_model.evaluation.metrics import ChessMetrics


class TestConfigSchemas:
    """Test Pydantic configuration schemas."""

    def test_lora_config_validation(self):
        """Test LoRA config validation."""
        # Valid config
        config = LoRAConfig(r=16, lora_alpha=32)
        assert config.r == 16
        assert config.lora_alpha == 32

        # Invalid: lora_alpha < r
        with pytest.raises(ValueError):
            LoRAConfig(r=16, lora_alpha=8)

    def test_model_config_defaults(self):
        """Test model config defaults."""
        config = ModelConfig()
        assert config.architecture == "llama"
        assert config.base_model_name == "meta-llama/Llama-3.2-1B"
        assert config.max_context_length == 100
        assert config.use_lora is True

    def test_model_config_precision_validation(self):
        """Test that both fp16 and bf16 cannot be enabled."""
        with pytest.raises(ValueError):
            ModelConfig(use_bf16=True, use_fp16=True)

    def test_training_config_defaults(self):
        """Test training config defaults."""
        config = TrainingConfig()
        assert config.num_epochs == 8
        assert config.batch_size == 64
        assert config.learning_rate == 3e-4
        assert config.mixed_precision == "bf16"
        assert config.checkmate_ratio == 0.05
        assert config.phase == 1

    def test_data_config_defaults(self):
        """Test data config defaults."""
        config = DataConfig()
        assert config.max_context_length == 100
        assert config.validation_split == 0.1
        assert config.use_mixed_dataset is True

    def test_pipeline_config_composition(self):
        """Test that pipeline config composes other configs."""
        config = PipelineConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training_phase1, TrainingConfig)
        assert isinstance(config.training_phase2, TrainingConfig)

        # Verify phase-specific settings
        assert config.training_phase1.phase == 1
        assert config.training_phase1.checkmate_ratio == 0.05
        assert config.training_phase2.phase == 2
        assert config.training_phase2.checkmate_ratio == 0.70


class TestYAMLConfigs:
    """Test YAML configuration loading."""

    def test_load_llama_config(self):
        """Test loading Llama model config."""
        config_path = Path("configs/model/llama_1b.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        config_dict = OmegaConf.load(config_path)
        assert config_dict.architecture == "llama"
        assert config_dict.use_lora is True
        assert config_dict.lora_config.r == 16

    def test_load_training_phase1_config(self):
        """Test loading Phase 1 training config."""
        config_path = Path("configs/training/phase1_general.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        config_dict = OmegaConf.load(config_path)
        assert config_dict.phase == 1
        assert config_dict.num_epochs == 8
        assert config_dict.checkmate_ratio == 0.05

    def test_load_training_phase2_config(self):
        """Test loading Phase 2 training config."""
        config_path = Path("configs/training/phase2_checkmate.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        config_dict = OmegaConf.load(config_path)
        assert config_dict.phase == 2
        assert config_dict.num_epochs == 4
        assert config_dict.checkmate_ratio == 0.70
        assert config_dict.learning_rate == 1e-5


class TestEvaluationMetrics:
    """Test evaluation metrics."""

    def test_move_accuracy(self):
        """Test move accuracy calculation."""
        # Create dummy data
        predictions = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # [2, 4]
        targets = torch.tensor([[1, 2, 0, 4], [5, 0, 7, 8]])  # [2, 4]
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float)  # [2, 4]

        accuracy = ChessMetrics.move_accuracy(predictions, targets, mask)
        # 6 correct out of 8 total = 0.75
        assert accuracy == 0.75

    def test_move_accuracy_with_masking(self):
        """Test move accuracy with masking."""
        predictions = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        targets = torch.tensor([[1, 2, 0, 4], [5, 0, 7, 8]])
        mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.float)  # Mask out some positions

        accuracy = ChessMetrics.move_accuracy(predictions, targets, mask)
        # With mask: positions [0,0], [0,1], [0,3], [1,0], [1,2], [1,3] = 6 positions
        # Correct: [0,0], [0,1], [0,3], [1,0], [1,2], [1,3] = 5 correct
        assert accuracy == 5.0 / 6.0

    def test_top_k_accuracy(self):
        """Test top-k accuracy calculation."""
        # Create logits where top-5 predictions include the target
        batch_size, seq_len, vocab_size = 2, 3, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)

        targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = torch.ones(batch_size, seq_len, dtype=torch.float)

        # Set targets to have high logits (so they're in top-5)
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, targets[b, s]] = 10.0

        accuracy = ChessMetrics.top_k_accuracy(logits, targets, mask, k=5)
        assert accuracy == 1.0  # All targets should be in top-5

    def test_checkmate_accuracy(self):
        """Test checkmate accuracy calculation."""
        # Perfect predictions
        pred = torch.tensor([1.0, 0.0, 1.0, 0.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        accuracy = ChessMetrics.checkmate_accuracy(pred, target)
        assert accuracy == 1.0

        # All wrong
        pred = torch.tensor([1.0, 1.0, 1.0, 1.0])
        target = torch.tensor([0.0, 0.0, 0.0, 0.0])
        accuracy = ChessMetrics.checkmate_accuracy(pred, target)
        assert accuracy == 0.0

        # Half correct
        pred = torch.tensor([0.8, 0.2, 0.7, 0.3])  # Will round to [1, 0, 1, 0]
        target = torch.tensor([1.0, 1.0, 1.0, 0.0])
        accuracy = ChessMetrics.checkmate_accuracy(pred, target)
        assert accuracy == 0.5

    def test_perplexity(self):
        """Test perplexity calculation."""
        batch_size, seq_len, vocab_size = 2, 4, 10

        # Create logits and targets
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.float)

        perplexity = ChessMetrics.perplexity(logits, targets, mask)

        # Perplexity should be positive
        assert perplexity > 0
        # For random logits, perplexity should be roughly vocab_size
        assert 1 < perplexity < vocab_size * 2

    def test_batch_metrics(self):
        """Test batch metrics calculation."""
        batch_size, seq_len, vocab_size = 2, 3, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.float)

        metrics = ChessMetrics.batch_metrics(logits, targets, mask)

        # Check all metrics are present
        assert "move_accuracy" in metrics
        assert "top5_accuracy" in metrics
        assert "perplexity" in metrics

        # Check metric ranges
        assert 0 <= metrics["move_accuracy"] <= 1
        assert 0 <= metrics["top5_accuracy"] <= 1
        assert metrics["perplexity"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
