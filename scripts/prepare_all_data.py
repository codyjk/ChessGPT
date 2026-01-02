"""
Prepare all training datasets from phase splits.

This script converts the game splits (.txt) to CSV format needed for training:
- Phase 1: 95% regular / 5% checkmate
- Phase 2: 70% checkmate / 30% regular
- Test: 1K games for sanity check

Usage:
    poetry run python scripts/prepare_all_data.py
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        return False


def prepare_dataset(
    input_file: str,
    output_training: str,
    output_validation: str,
    description: str,
):
    """Prepare a single dataset."""
    cmd = [
        "poetry", "run", "prepare-training-data",
        "--input-reduced-pgn-file", input_file,
        "--output-training-data-file", output_training,
        "--output-validation-data-file", output_validation,
        "--max-context-length", "100",
        "--validation-split", "0.1"
    ]

    return run_command(cmd, description)


def main():
    print("=" * 60)
    print("ChessGPT Data Preparation")
    print("=" * 60)
    print("\nThis will prepare all training datasets:")
    print("  1. Phase 1 training data (175K games)")
    print("  2. Phase 2 training data (175K games)")
    print("  3. Test data (1K games) - already done")
    print("\nThis may take 10-20 minutes...")

    # Check if input files exist
    phase1_file = Path("data/phase_splits/phase1_general.txt")
    phase2_file = Path("data/phase_splits/phase2_checkmate.txt")

    if not phase1_file.exists():
        print(f"\n✗ Error: {phase1_file} not found")
        print("Run: poetry run python scripts/prepare_checkmate_data.py")
        sys.exit(1)

    if not phase2_file.exists():
        print(f"\n✗ Error: {phase2_file} not found")
        print("Run: poetry run python scripts/prepare_checkmate_data.py")
        sys.exit(1)

    success = True

    # Prepare Phase 1 data
    if not prepare_dataset(
        input_file="data/phase_splits/phase1_general.txt",
        output_training="data/phase_splits/training_phase1.csv",
        output_validation="data/phase_splits/validation_phase1.csv",
        description="Preparing Phase 1 data (95% regular / 5% checkmate)"
    ):
        success = False

    # Prepare Phase 2 data
    if not prepare_dataset(
        input_file="data/phase_splits/phase2_checkmate.txt",
        output_training="data/phase_splits/training_phase2.csv",
        output_validation="data/phase_splits/validation_phase2.csv",
        description="Preparing Phase 2 data (70% checkmate / 30% regular)"
    ):
        success = False

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✓ All datasets prepared successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("\n1. Run sanity check (1K games, 1 epoch):")
        print("   poetry run train --config test_gpt2_medium_1k --name gpt2-medium-test")
        print("\n2. Run Phase 1 training (175K games, 8 epochs, ~70 hours):")
        print("   poetry run train --config phase1_gpt2_medium --name gpt2-medium-v1-phase1")
        print("\n3. Run Phase 2 training (175K games, 4 epochs, ~10 hours):")
        print("   poetry run train --config phase2_gpt2_medium --name gpt2-medium-v1-phase2 \\")
        print("       --resume-from models/gpt2-medium-v1-phase1/checkpoint-best")
    else:
        print("✗ Some datasets failed to prepare")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
