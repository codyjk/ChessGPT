"""
Integration tests for model architecture components.

Tests VocabBridge, LlamaChessTransformer, and ModelFactory.
"""

import pytest
import torch
from pathlib import Path


class TestVocabBridge:
    """Test VocabBridge functionality."""

    def test_vocab_bridge_creation(self):
        """Test creating VocabBridge."""
        from chess_model.model.vocab_bridge import VocabBridge

        bridge = VocabBridge(
            chess_vocab_size=287,
            hidden_size=1024,
            pad_token_id=0,
        )

        assert bridge.chess_vocab_size == 287
        assert bridge.hidden_size == 1024
        assert bridge.pad_token_id == 0
        print("✓ VocabBridge creation works")

    def test_vocab_bridge_embed(self):
        """Test chess token embedding."""
        from chess_model.model.vocab_bridge import VocabBridge

        bridge = VocabBridge(chess_vocab_size=287, hidden_size=1024)

        # Create dummy input
        input_ids = torch.randint(0, 287, (2, 10))  # (batch=2, seq=10)

        # Get embeddings
        embeddings = bridge.embed_chess_moves(input_ids)

        assert embeddings.shape == (2, 10, 1024)
        print("✓ VocabBridge embedding works")

    def test_vocab_bridge_project(self):
        """Test projection to chess logits."""
        from chess_model.model.vocab_bridge import VocabBridge

        bridge = VocabBridge(chess_vocab_size=287, hidden_size=1024)

        # Create dummy hidden states
        hidden_states = torch.randn(2, 10, 1024)

        # Project to logits
        logits = bridge.project_to_chess_logits(hidden_states)

        assert logits.shape == (2, 10, 287)
        print("✓ VocabBridge projection works")

    def test_vocab_bridge_save_load(self, tmp_path):
        """Test saving and loading VocabBridge."""
        from chess_model.model.vocab_bridge import VocabBridge

        # Create and save
        bridge1 = VocabBridge(chess_vocab_size=287, hidden_size=1024)
        bridge1.save_pretrained(str(tmp_path))

        # Load
        bridge2 = VocabBridge.from_pretrained(str(tmp_path))

        assert bridge2.chess_vocab_size == 287
        assert bridge2.hidden_size == 1024
        print("✓ VocabBridge save/load works")


class TestModelFactory:
    """Test ModelFactory functionality."""

    def test_create_gpt2_model(self):
        """Test creating GPT-2 model."""
        from chess_model.model.factory import ModelFactory
        from chess_model.model.tokenizer import ChessTokenizer

        # Create dummy tokenizer
        tokenizer = ChessTokenizer()
        tokenizer.vocab_size = 287

        # Create model
        model = ModelFactory.create_model(
            architecture="gpt2",
            vocab_size=tokenizer.vocab_size,
            max_context_length=50,
            num_embeddings=256,
            num_layers=4,
            num_heads=4,
        )

        # Test forward pass
        input_ids = torch.randint(0, 287, (2, 10))
        outputs = model(input_ids)

        assert outputs.shape == (2, 10, 287)
        print("✓ ModelFactory GPT-2 creation works")

    def test_architecture_detection(self, tmp_path):
        """Test auto-detecting model architecture."""
        from chess_model.model.factory import ModelFactory

        # Create fake chess_config.json for Llama
        config_path = tmp_path / "chess_config.json"
        config_path.write_text('{"architecture": "llama"}')

        arch = ModelFactory._detect_architecture(tmp_path)
        assert arch == "llama"
        print("✓ Architecture detection works")


class TestLlamaIntegration:
    """Test Llama integration (requires transformers/peft)."""

    @pytest.mark.skipif(
        True,  # Skip by default since it requires large downloads
        reason="Requires transformers/peft and model downloads",
    )
    def test_llama_creation(self):
        """Test creating LlamaChessTransformer."""
        from chess_model.model.llama_chess import LlamaChessTransformer

        model = LlamaChessTransformer(
            chess_vocab_size=287,
            base_model_name="meta-llama/Llama-3.2-1B",
            use_lora=True,
        )

        assert model.chess_vocab_size == 287
        assert model.use_lora is True
        print("✓ LlamaChessTransformer creation works")

    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Requires transformers/peft and model downloads",
    )
    def test_llama_forward(self):
        """Test Llama forward pass."""
        from chess_model.model.llama_chess import LlamaChessTransformer

        model = LlamaChessTransformer(
            chess_vocab_size=287,
            base_model_name="meta-llama/Llama-3.2-1B",
            use_lora=True,
        )

        # Test forward
        input_ids = torch.randint(0, 287, (1, 5))
        outputs = model(input_ids)

        assert "logits" in outputs
        assert outputs["logits"].shape == (1, 5, 287)
        print("✓ LlamaChessTransformer forward pass works")


def run_all_tests():
    """Run all tests."""
    import sys

    print("\n" + "=" * 60)
    print("Testing Model Architecture Components")
    print("=" * 60 + "\n")

    # Test VocabBridge
    print("VocabBridge Tests:")
    try:
        test_bridge = TestVocabBridge()
        test_bridge.test_vocab_bridge_creation()
        test_bridge.test_vocab_bridge_embed()
        test_bridge.test_vocab_bridge_project()

        # Test save/load with temp directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_bridge.test_vocab_bridge_save_load(Path(tmp_dir))

        print()
    except Exception as e:
        print(f"✗ VocabBridge tests failed: {e}\n")
        import traceback
        traceback.print_exc()

    # Test ModelFactory
    print("ModelFactory Tests:")
    try:
        test_factory = TestModelFactory()
        test_factory.test_create_gpt2_model()

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_factory.test_architecture_detection(Path(tmp_dir))

        print()
    except Exception as e:
        print(f"✗ ModelFactory tests failed: {e}\n")
        import traceback
        traceback.print_exc()

    # Llama tests (skipped by default)
    print("Llama Integration Tests:")
    print("  ⏸️  Skipped (requires transformers/peft installation)")
    print()

    print("=" * 60)
    print("Model Architecture Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
