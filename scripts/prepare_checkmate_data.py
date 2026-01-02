"""
Prepare checkmate-focused datasets for Phase 1 and Phase 2 training.

This script:
1. Extracts checkmate games from master.txt (games ending in 1-0 or 0-1)
2. Creates Phase 1 split: 95% regular + 5% checkmate games
3. Creates Phase 2 split: 70% checkmate + 30% regular games
4. Saves splits to separate files for training
"""
import argparse
import random
from pathlib import Path
from typing import List, Tuple


def load_games(file_path: str) -> List[str]:
    """Load all games from a file."""
    print(f"Loading games from {file_path}...")
    with open(file_path, 'r') as f:
        games = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(games):,} games")
    return games


def split_by_outcome(games: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Split games by outcome.

    Returns:
        (checkmate_games, draw_games, all_decisive_games)
        - checkmate_games: Games ending in 1-0 or 0-1 (assumed checkmate/resignation)
        - draw_games: Games ending in 1/2-1/2
        - all_decisive_games: All non-draw games (for simplicity, treating as "regular")
    """
    checkmate_games = []
    draw_games = []

    for game in games:
        if game.endswith('1-0') or game.endswith('0-1'):
            checkmate_games.append(game)
        elif game.endswith('1/2-1/2'):
            draw_games.append(game)
        else:
            # Malformed game, skip
            continue

    print(f"\nGame distribution:")
    print(f"  Decisive games (1-0 or 0-1): {len(checkmate_games):,} ({len(checkmate_games)/len(games)*100:.1f}%)")
    print(f"  Draw games (1/2-1/2): {len(draw_games):,} ({len(draw_games)/len(games)*100:.1f}%)")

    return checkmate_games, draw_games, checkmate_games  # Using all decisive as "regular" for now


def create_phase1_split(
    regular_games: List[str],
    checkmate_games: List[str],
    output_file: str,
    regular_ratio: float = 0.95,
    checkmate_ratio: float = 0.05,
    seed: int = 42
) -> None:
    """
    Create Phase 1 training split: 95% regular games + 5% checkmate games.

    Args:
        regular_games: Pool of regular games
        checkmate_games: Pool of checkmate games
        output_file: Output file path
        regular_ratio: Ratio of regular games (default 0.95)
        checkmate_ratio: Ratio of checkmate games (default 0.05)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Sample games
    n_checkmate = int(len(checkmate_games) * checkmate_ratio)
    n_regular = int(n_checkmate * (regular_ratio / checkmate_ratio))

    # Ensure we don't exceed available games
    n_checkmate = min(n_checkmate, len(checkmate_games))
    n_regular = min(n_regular, len(regular_games))

    sampled_regular = random.sample(regular_games, n_regular)
    sampled_checkmate = random.sample(checkmate_games, n_checkmate)

    # Combine and shuffle
    phase1_games = sampled_regular + sampled_checkmate
    random.shuffle(phase1_games)

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for game in phase1_games:
            f.write(game + '\n')

    print(f"\nPhase 1 split created:")
    print(f"  Regular games: {n_regular:,} ({n_regular/len(phase1_games)*100:.1f}%)")
    print(f"  Checkmate games: {n_checkmate:,} ({n_checkmate/len(phase1_games)*100:.1f}%)")
    print(f"  Total: {len(phase1_games):,} games")
    print(f"  Saved to: {output_path}")


def create_phase2_split(
    regular_games: List[str],
    checkmate_games: List[str],
    output_file: str,
    checkmate_ratio: float = 0.70,
    regular_ratio: float = 0.30,
    seed: int = 42
) -> None:
    """
    Create Phase 2 training split: 70% checkmate games + 30% regular games.

    Args:
        regular_games: Pool of regular games
        checkmate_games: Pool of checkmate games
        output_file: Output file path
        checkmate_ratio: Ratio of checkmate games (default 0.70)
        regular_ratio: Ratio of regular games (default 0.30)
        seed: Random seed for reproducibility
    """
    random.seed(seed + 1)  # Different seed from Phase 1

    # Sample games
    n_checkmate = int(len(checkmate_games) * checkmate_ratio)
    n_regular = int(n_checkmate * (regular_ratio / checkmate_ratio))

    # Ensure we don't exceed available games
    n_checkmate = min(n_checkmate, len(checkmate_games))
    n_regular = min(n_regular, len(regular_games))

    sampled_regular = random.sample(regular_games, n_regular)
    sampled_checkmate = random.sample(checkmate_games, n_checkmate)

    # Combine and shuffle
    phase2_games = sampled_regular + sampled_checkmate
    random.shuffle(phase2_games)

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for game in phase2_games:
            f.write(game + '\n')

    print(f"\nPhase 2 split created:")
    print(f"  Checkmate games: {n_checkmate:,} ({n_checkmate/len(phase2_games)*100:.1f}%)")
    print(f"  Regular games: {n_regular:,} ({n_regular/len(phase2_games)*100:.1f}%)")
    print(f"  Total: {len(phase2_games):,} games")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare checkmate-focused datasets for training"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="out/master.txt",
        help="Input PGN file with all games (default: out/master.txt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/phase_splits",
        help="Output directory for splits (default: data/phase_splits)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Checkmate Data Preparation")
    print("=" * 60)

    # Load all games
    games = load_games(args.input_file)

    # Split by outcome
    checkmate_games, draw_games, regular_games = split_by_outcome(games)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Phase 1 split (95% regular / 5% checkmate)
    print("\n" + "=" * 60)
    print("Creating Phase 1 Split (General Chess)")
    print("=" * 60)
    create_phase1_split(
        regular_games=regular_games,
        checkmate_games=checkmate_games,
        output_file=output_dir / "phase1_general.txt",
        regular_ratio=0.95,
        checkmate_ratio=0.05,
        seed=args.seed
    )

    # Create Phase 2 split (70% checkmate / 30% regular)
    print("\n" + "=" * 60)
    print("Creating Phase 2 Split (Checkmate Specialization)")
    print("=" * 60)
    create_phase2_split(
        regular_games=regular_games,
        checkmate_games=checkmate_games,
        output_file=output_dir / "phase2_checkmate.txt",
        checkmate_ratio=0.70,
        regular_ratio=0.30,
        seed=args.seed
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Input: {args.input_file}")
    print(f"Output directory: {output_dir}")
    print(f"  - phase1_general.txt (95% regular / 5% checkmate)")
    print(f"  - phase2_checkmate.txt (70% checkmate / 30% regular)")
    print("\nNext steps:")
    print("1. Run prepare-training-data on phase1_general.txt")
    print("2. Train Phase 1 model (8 epochs)")
    print("3. Run prepare-training-data on phase2_checkmate.txt")
    print("4. Train Phase 2 model (4 epochs, resume from Phase 1)")
    print("=" * 60)


if __name__ == "__main__":
    main()
