#!/usr/bin/env python3
"""Simple script to validate Phase 1 imports and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_config_imports():
    """Test importing config schemas."""
    print("Testing config imports...")
    try:
        from src.chess_model.config.schemas import (
            ModelConfig,
            LoRAConfig,
            TrainingConfig,
            DataConfig,
            PipelineConfig,
        )
        print("  ✓ Config schemas imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import config schemas: {e}")
        return False

def test_config_instantiation():
    """Test creating config instances."""
    print("\nTesting config instantiation...")
    try:
        from src.chess_model.config.schemas import ModelConfig, LoRAConfig

        # Test LoRA config
        lora_config = LoRAConfig(r=16, lora_alpha=32)
        assert lora_config.r == 16
        print("  ✓ LoRAConfig instantiated")

        # Test Model config
        model_config = ModelConfig()
        assert model_config.architecture == "llama"
        print("  ✓ ModelConfig instantiated")

        # Test validation
        try:
            invalid_lora = LoRAConfig(r=16, lora_alpha=8)  # Should fail
            print("  ✗ Validation not working (should have failed)")
            return False
        except ValueError:
            print("  ✓ Config validation working")

        return True
    except Exception as e:
        print(f"  ✗ Failed config instantiation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_imports():
    """Test importing evaluation metrics."""
    print("\nTesting evaluation imports...")
    try:
        from src.chess_model.evaluation.metrics import ChessMetrics
        print("  ✓ ChessMetrics imported successfully")

        # Check methods exist
        assert hasattr(ChessMetrics, 'move_accuracy')
        assert hasattr(ChessMetrics, 'top_k_accuracy')
        assert hasattr(ChessMetrics, 'legal_move_rate')
        assert hasattr(ChessMetrics, 'checkmate_accuracy')
        print("  ✓ All metric methods present")

        return True
    except Exception as e:
        print(f"  ✗ Failed to import evaluation metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_functionality():
    """Test basic metrics functionality."""
    print("\nTesting metrics functionality...")
    try:
        import torch
        from src.chess_model.evaluation.metrics import ChessMetrics

        # Test move accuracy
        predictions = torch.tensor([[1, 2, 3], [4, 5, 6]])
        targets = torch.tensor([[1, 2, 0], [4, 5, 6]])
        mask = torch.ones(2, 3, dtype=torch.float)

        accuracy = ChessMetrics.move_accuracy(predictions, targets, mask)
        expected = 5.0 / 6.0  # 5 correct out of 6
        assert abs(accuracy - expected) < 1e-6
        print(f"  ✓ move_accuracy working (got {accuracy:.4f})")

        # Test top-k accuracy
        logits = torch.randn(2, 3, 10)
        accuracy_topk = ChessMetrics.top_k_accuracy(logits, targets, mask, k=5)
        assert 0 <= accuracy_topk <= 1
        print(f"  ✓ top_k_accuracy working (got {accuracy_topk:.4f})")

        # Test checkmate accuracy
        pred = torch.tensor([1.0, 0.0, 1.0, 0.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        cm_accuracy = ChessMetrics.checkmate_accuracy(pred, target)
        assert cm_accuracy == 1.0
        print(f"  ✓ checkmate_accuracy working (got {cm_accuracy:.4f})")

        return True
    except Exception as e:
        print(f"  ✗ Failed metrics functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("="* 60)
    print("Phase 1 Validation Tests")
    print("="* 60)

    results = []

    # Run tests
    results.append(("Config Imports", test_config_imports()))
    results.append(("Config Instantiation", test_config_instantiation()))
    results.append(("Evaluation Imports", test_evaluation_imports()))
    results.append(("Metrics Functionality", test_metrics_functionality()))

    # Summary
    print("\n" + "="* 60)
    print("Summary:")
    print("="* 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} passed")

    if total_passed == total_tests:
        print("\n🎉 All Phase 1 components validated successfully!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
