"""
Quick data sampling script for testing.

Extracts a small sample from the Lichess database for rapid iteration.
Processes 10K games with at most 5-6 moves each for quick training tests.
"""

import subprocess
import argparse
from pathlib import Path
import re
import csv
from pgn_utils.pgn_utils import (
    process_raw_games_from_file,
    raw_game_has_moves,
    process_chess_moves,
)


def decompress_sample(pgn_zst_file: str, output_pgn: str, num_games: int = 10000):
    """
    Decompress a sample of games from .pgn.zst file.

    Args:
        pgn_zst_file: Path to .pgn.zst file
        output_pgn: Where to write decompressed sample
        num_games: Number of games to extract
    """
    print(f"Decompressing {num_games} games from {pgn_zst_file}...")
    print("This may take a few minutes...")

    # Use zstd to decompress, then head to get first N games
    # PGN games are separated by blank lines, roughly 12-15 lines per game
    lines_per_game = 15
    target_lines = num_games * lines_per_game

    cmd = f"zstd -dc {pgn_zst_file} | head -n {target_lines} > {output_pgn}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error decompressing: {result.stderr}")
        return False

    print(f"✓ Decompressed to {output_pgn}")
    return True


def extract_moves(pgn_file: str, output_txt: str, max_moves: int = 6, min_elo: int = 1800):
    """
    Extract moves from PGN file, keeping only games with good ELO and limited length.

    Args:
        pgn_file: Decompressed PGN file
        output_txt: Output text file with moves (one game per line)
        max_moves: Maximum number of moves per game (for opening phase focus)
        min_elo: Minimum ELO rating
    """
    print(f"\nExtracting moves (max {max_moves} moves, min ELO {min_elo})...")

    white_elo_pattern = re.compile(r'^\s*\[WhiteElo "(\d+)"\]\s*$')
    black_elo_pattern = re.compile(r'^\s*\[BlackElo "(\d+)"\]\s*$')
    result_pattern = re.compile(r'^\s*\[Result "([^"]+)"\]\s*$')

    games_written = 0
    games_skipped = 0

    with open(output_txt, 'w') as out_file:
        for raw_game in process_raw_games_from_file(pgn_file):
            if not raw_game_has_moves(raw_game):
                games_skipped += 1
                continue

            # Extract ELO ratings
            white_elo = None
            black_elo = None
            result = None

            for metadata_line in raw_game.metadata:
                if white_elo_pattern.search(metadata_line):
                    white_elo = int(white_elo_pattern.search(metadata_line).group(1))
                elif black_elo_pattern.search(metadata_line):
                    black_elo = int(black_elo_pattern.search(metadata_line).group(1))
                elif result_pattern.search(metadata_line):
                    result = result_pattern.search(metadata_line).group(1)

            # Filter by ELO
            if white_elo is None or black_elo is None:
                games_skipped += 1
                continue

            if white_elo < min_elo or black_elo < min_elo:
                games_skipped += 1
                continue

            # Process moves
            moves = process_chess_moves(raw_game.moves)
            move_list = moves.strip().split()

            # Limit to max_moves
            if len(move_list) > max_moves:
                move_list = move_list[:max_moves]

            # Write to file with outcome
            # Format: "moves | outcome"
            limited_moves = " ".join(move_list)
            out_file.write(f"{limited_moves} | {result}\n")
            games_written += 1

    print(f"✓ Extracted {games_written} games (skipped {games_skipped})")
    return games_written


def create_training_data(moves_file: str, train_csv: str, val_csv: str, val_split: float = 0.1):
    """
    Convert moves file to training/validation CSV format.

    Args:
        moves_file: Text file with moves (one game per line)
        train_csv: Output training CSV
        val_csv: Output validation CSV
        val_split: Fraction for validation
    """
    print(f"\nCreating training data (val split: {val_split})...")

    # Read all games
    with open(moves_file, 'r') as f:
        games = f.readlines()

    # Split into train/val
    num_val = int(len(games) * val_split)
    val_games = games[:num_val]
    train_games = games[num_val:]

    print(f"Training games: {len(train_games)}")
    print(f"Validation games: {len(val_games)}")

    # Write training CSV (order: context, is_checkmate, outcome)
    with open(train_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['context', 'is_checkmate', 'outcome'])

        for game in train_games:
            if ' | ' in game:
                moves, outcome = game.strip().split(' | ')
                # Order: context (moves), is_checkmate (0 for now), outcome
                writer.writerow([moves, 0, outcome])

    # Write validation CSV (order: context, is_checkmate, outcome)
    with open(val_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['context', 'is_checkmate', 'outcome'])

        for game in val_games:
            if ' | ' in game:
                moves, outcome = game.strip().split(' | ')
                writer.writerow([moves, 0, outcome])

    print(f"✓ Training data: {train_csv}")
    print(f"✓ Validation data: {val_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a quick data sample for testing"
    )
    parser.add_argument(
        "--pgn-file",
        type=str,
        default="lichess_db_standard_rated_2016-04.pgn.zst",
        help="Input PGN file (.pgn.zst)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/quick_sample",
        help="Output directory"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=12000,  # Extract more to account for filtering
        help="Number of games to initially extract"
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=6,
        help="Maximum moves per game (focus on openings)"
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=1800,
        help="Minimum ELO rating"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    temp_pgn = output_dir / "sample.pgn"
    moves_txt = output_dir / "moves.txt"
    train_csv = output_dir / "training-data.csv"
    val_csv = output_dir / "validation-data.csv"

    print("=" * 60)
    print("Quick Data Sample for Testing")
    print("=" * 60)
    print(f"Source: {args.pgn_file}")
    print(f"Target: {args.output_dir}")
    print(f"Max moves: {args.max_moves}")
    print(f"Min ELO: {args.min_elo}")
    print("=" * 60 + "\n")

    # Step 1: Decompress sample
    if not decompress_sample(args.pgn_file, str(temp_pgn), args.num_games):
        print("Failed to decompress")
        return

    # Step 2: Extract moves
    games_extracted = extract_moves(
        str(temp_pgn),
        str(moves_txt),
        max_moves=args.max_moves,
        min_elo=args.min_elo
    )

    if games_extracted < 1000:
        print(f"\n⚠️  Warning: Only extracted {games_extracted} games. Consider lowering --min-elo")

    # Step 3: Create training data
    create_training_data(str(moves_txt), str(train_csv), str(val_csv))

    # Cleanup temp files
    temp_pgn.unlink()
    moves_txt.unlink()

    print("\n" + "=" * 60)
    print("✓ Data sample ready!")
    print("=" * 60)
    print(f"\nTraining data: {train_csv}")
    print(f"Validation data: {val_csv}")
    print(f"\nNext step: Update configs/data/dataset.yaml with these paths")
    print("Then run: poetry run python examples/train_with_huggingface.py")


if __name__ == "__main__":
    main()
