"""
PGN to enriched training CSV pipeline.

Takes a raw PGN file and produces a CSV with:
    moves,outcome,checkmate_move_idx,ply_count

Key steps:
1. Parse PGN into individual games (using pgn utils)
2. Filter by ELO (both players >= min_elo)
3. Extract move sequences and outcomes
4. Detect checkmate games (final move contains '#') and record the checkmate move index
5. Write to CSV

python-chess is used here during data prep to validate checkmate detection.
It is NOT used at inference time.
"""

import csv
from pathlib import Path

from chessgpt.pgn.utils import (
    OUTCOME_PATTERN,
    get_elo,
    process_chess_moves,
    process_raw_games_from_file,
)


def prepare_training_data(
    input_pgn: str,
    output_csv: str,
    min_elo: int = 1800,
    min_moves: int = 10,
) -> dict[str, int]:
    """
    Process a PGN file into an enriched training CSV.

    Returns counts dict with stats about the processing.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"total": 0, "filtered_elo": 0, "filtered_short": 0, "checkmates": 0, "written": 0}

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["moves", "outcome", "checkmate_move_idx", "ply_count"])

        for raw_game in process_raw_games_from_file(input_pgn):
            counts["total"] += 1

            # ELO filter: both players must meet minimum
            white_elo, black_elo = get_elo(raw_game)
            if white_elo is None or black_elo is None:
                counts["filtered_elo"] += 1
                continue
            if white_elo < min_elo or black_elo < min_elo:
                counts["filtered_elo"] += 1
                continue

            # Process moves
            processed = process_chess_moves(raw_game.moves)
            if not processed:
                counts["filtered_short"] += 1
                continue

            parts = processed.split()
            if len(parts) < 2:
                counts["filtered_short"] += 1
                continue

            outcome = parts[-1]
            if not OUTCOME_PATTERN.match(outcome):
                counts["filtered_short"] += 1
                continue

            moves = parts[:-1]
            if len(moves) < min_moves:
                counts["filtered_short"] += 1
                continue

            # Detect checkmate: last move contains '#'
            checkmate_move_idx = -1
            if moves and "#" in moves[-1]:
                checkmate_move_idx = len(moves) - 1
                counts["checkmates"] += 1

            ply_count = len(moves)
            moves_str = " ".join(moves)

            writer.writerow([moves_str, outcome, checkmate_move_idx, ply_count])
            counts["written"] += 1

    return counts


def print_stats(counts: dict[str, int]) -> None:
    print("\nData preparation complete:")
    print(f"  Total games parsed: {counts['total']}")
    print(f"  Filtered (ELO):     {counts['filtered_elo']}")
    print(f"  Filtered (short):   {counts['filtered_short']}")
    print(f"  Written:            {counts['written']}")
    print(f"  Checkmates:         {counts['checkmates']}")
    if counts["written"] > 0:
        pct = counts["checkmates"] / counts["written"] * 100
        print(f"  Checkmate rate:     {pct:.1f}%")
