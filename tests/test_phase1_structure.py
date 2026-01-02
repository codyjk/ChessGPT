"""Basic structure validation tests for Phase 1 (no external dependencies required)."""

import pytest
from pathlib import Path
import sys


class TestFileStructure:
    """Test that all Phase 1 files have been created."""

    def test_config_schemas_exists(self):
        """Test that config schemas file exists."""
        path = Path("src/chess_model/config/schemas.py")
        assert path.exists(), f"Config schemas not found at {path}"

        # Verify it has content
        content = path.read_text()
        assert "ModelConfig" in content
        assert "LoRAConfig" in content
        assert "TrainingConfig" in content
        assert "DataConfig" in content
        assert "PipelineConfig" in content

    def test_config_yaml_files_exist(self):
        """Test that all YAML config files exist."""
        expected_files = [
            "configs/model/llama_1b.yaml",
            "configs/model/gpt2_baseline.yaml",
            "configs/training/phase1_general.yaml",
            "configs/training/phase2_checkmate.yaml",
            "configs/data/dataset.yaml",
            "configs/logging/wandb.yaml",
            "configs/config.yaml",
            "configs/pipeline/full_training.yaml",
            "configs/pipeline/quick_test.yaml",
        ]

        for file_path in expected_files:
            path = Path(file_path)
            assert path.exists(), f"Config file not found: {file_path}"

    def test_logger_exists(self):
        """Test that WandB logger file exists."""
        path = Path("src/chess_model/training/logger.py")
        assert path.exists(), f"Logger not found at {path}"

        content = path.read_text()
        assert "ExperimentLogger" in content
        assert "wandb" in content

    def test_evaluation_metrics_exists(self):
        """Test that evaluation metrics file exists."""
        path = Path("src/chess_model/evaluation/metrics.py")
        assert path.exists(), f"Metrics not found at {path}"

        content = path.read_text()
        assert "ChessMetrics" in content
        assert "move_accuracy" in content
        assert "top_k_accuracy" in content
        assert "legal_move_rate" in content
        assert "checkmate_accuracy" in content

    def test_pyproject_updated(self):
        """Test that pyproject.toml has new dependencies."""
        path = Path("pyproject.toml")
        content = path.read_text()

        required_deps = [
            "hydra-core",
            "pydantic",
            "omegaconf",
            "wandb",
            "peft",
            "accelerate",
            "click",
            "zstandard",
        ]

        for dep in required_deps:
            assert dep in content, f"Dependency {dep} not found in pyproject.toml"

    def test_config_directory_structure(self):
        """Test that config directories exist."""
        expected_dirs = [
            "configs/model",
            "configs/training",
            "configs/data",
            "configs/logging",
            "configs/pipeline",
            "src/chess_model/config",
            "src/chess_model/training",
            "src/chess_model/evaluation",
        ]

        for dir_path in expected_dirs:
            path = Path(dir_path)
            assert path.exists() and path.is_dir(), f"Directory not found: {dir_path}"


class TestConfigContent:
    """Test configuration file contents."""

    def test_llama_config_content(self):
        """Test Llama config has correct settings."""
        path = Path("configs/model/llama_1b.yaml")
        if not path.exists():
            pytest.skip("Config file not found")

        content = path.read_text()
        assert "architecture: llama" in content
        assert "meta-llama/Llama-3.2-1B" in content
        assert "use_lora: true" in content
        assert "r: 16" in content

    def test_phase1_training_config_content(self):
        """Test Phase 1 training config."""
        path = Path("configs/training/phase1_general.yaml")
        if not path.exists():
            pytest.skip("Config file not found")

        content = path.read_text()
        assert "phase: 1" in content
        assert "num_epochs: 8" in content
        assert "checkmate_ratio: 0.05" in content

    def test_phase2_training_config_content(self):
        """Test Phase 2 training config."""
        path = Path("configs/training/phase2_checkmate.yaml")
        if not path.exists():
            pytest.skip("Config file not found")

        content = path.read_text()
        assert "phase: 2" in content
        assert "num_epochs: 4" in content
        assert "checkmate_ratio: 0.70" in content or "checkmate_ratio: 0.7" in content
        assert "learning_rate: 1e-5" in content or "learning_rate: 0.00001" in content

    def test_pipeline_configs_differ(self):
        """Test that full and quick pipeline configs are different."""
        full_path = Path("configs/pipeline/full_training.yaml")
        quick_path = Path("configs/pipeline/quick_test.yaml")

        if not (full_path.exists() and quick_path.exists()):
            pytest.skip("Pipeline configs not found")

        full_content = full_path.read_text()
        quick_content = quick_path.read_text()

        # Full should have auto_download: true
        assert "auto_download: true" in full_content

        # Quick should be for testing
        assert "quick_test" in quick_content
        assert full_content != quick_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
