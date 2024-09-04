import argparse
import os
import sys
from collections import defaultdict

from chess_model import (
    get_elo_directory,
    process_chess_moves,
    process_raw_games_from_file,
    raw_game_has_moves,
)

INFINITY = float("inf")

DIRECTORY_TO_ELO_RATING_RANGE = {
    "beginner": (0, 1250),
    "intermediate": (1250, 1750),
    "master": (1750, 2500),
    "grandmaster": (2500, INFINITY),
    "unknown": (INFINITY, INFINITY),
}


def main():
    """
    Usage: poetry run reduce-pgn --input-pgn-file <input-pgn-file> --output-dir <output-dir>
    """
    parser = argparse.ArgumentParser(
        description="Reduce a chess games PGN database file to a list of moves. The reduced files will be organized by rating range (beginner, intermediate, master, grandmaster)."
    )
    parser.add_argument(
        "--input-pgn-file", type=str, help="The input PGN file.", required=True
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="The output directory, where one reduced file will be written per ELO rating range. Any of these files can then be used in the `prepare-training-data` step.",
        required=True,
    )
    args = parser.parse_args()
    input_file = args.input_pgn_file
    output_directory = args.output_dir

    file_write_counters = defaultdict(int)
    file_handles = {}
    for directory in DIRECTORY_TO_ELO_RATING_RANGE.keys():
        file = open(os.path.join(output_directory, f"{directory}.txt"), "w")
        file_handles[directory] = file

    print("Processing file...")
    for raw_game in process_raw_games_from_file(input_file):
        processed_moves = process_chess_moves(raw_game.moves)
        if not raw_game_has_moves(raw_game):
            continue

        directory = get_elo_directory(raw_game, DIRECTORY_TO_ELO_RATING_RANGE)
        file_handles[directory].write(processed_moves + "\n")
        file_write_counters[directory] += 1

    for directory, file_handle in file_handles.items():
        print(f"Processed {file_write_counters[directory]} games in {directory}.")
        file_handle.close()
