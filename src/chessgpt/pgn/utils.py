"""
PGN parsing utilities for processing Lichess game databases.

Handles the core text parsing of PGN files: extracting metadata lines (e.g. [WhiteElo "1800"]),
parsing algebraic move notation (e.g. "1. d4 e6 2. Nf3 b6"), and detecting game outcomes.
Works line-by-line to support streaming through multi-GB PGN files.
"""

import re
import subprocess
from typing import Iterator, NamedTuple


class RawGame(NamedTuple):
    """A game as extracted from a PGN file: metadata headers + raw move text."""

    metadata: list[str]
    moves: str


METADATA_PATTERN = re.compile(r"^\s*\[(.*)\]\s*$")
MOVE_PATTERN = re.compile(
    r"(\d+\.)\s*"
    r"([BNRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[BNRQ])?(?:e\.p\.)?[+#]?|O-O(?:-O)?)"
    r"\s*(?:\{[^}]*\})?\s*"
    r"(?:(\d+)\.{3})?\s*"
    r"([BNRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[BNRQ])?(?:e\.p\.)?[+#]?|O-O(?:-O)?)?"
)
OUTCOME_PATTERN = re.compile(r"(1-0|0-1|1/2-1/2)")
WHITE_ELO_PATTERN = re.compile(r'^\s*\[WhiteElo "(\d+)"\]\s*$')
BLACK_ELO_PATTERN = re.compile(r'^\s*\[BlackElo "(\d+)"\]\s*$')


def count_lines_fast(filename: str) -> int:
    """Count lines in a file using wc -l for speed on large files."""
    result = subprocess.run(["wc", "-l", filename], capture_output=True, text=True)
    return int(result.stdout.split()[0])


def is_metadata_line(line: str) -> bool:
    return bool(METADATA_PATTERN.search(line))


def is_moves_line(line: str) -> bool:
    return bool(OUTCOME_PATTERN.search(line))


def process_chess_moves(input_string: str) -> str:
    """Extract moves from a PGN move text line into space-separated format with outcome."""
    moves = MOVE_PATTERN.findall(input_string)
    result = OUTCOME_PATTERN.search(input_string)

    processed_moves = []
    for move in moves:
        if move[1]:  # White's move
            processed_moves.append(move[1])
        if move[3]:  # Black's move
            processed_moves.append(move[3])
    result = result.group(1) if result else ""

    return " ".join(processed_moves + [result]).strip()


def raw_game_has_moves(raw_game: RawGame, min_moves: int = 2) -> bool:
    """Check if a game has the minimum number of moves and a valid outcome."""
    moves_and_outcome = raw_game.moves.split(" ")
    moves, outcome = moves_and_outcome[:-1], moves_and_outcome[-1]
    return len(moves) >= min_moves and bool(OUTCOME_PATTERN.search(outcome))


def process_raw_games_from_file(filename: str) -> Iterator[RawGame]:
    """Stream RawGame objects from a PGN file, yielding one per game."""
    from tqdm import tqdm

    total_lines = count_lines_fast(filename)
    current_metadata: list[str] = []
    with open(filename) as file:
        for line in tqdm(file, total=total_lines, desc="Processing PGN"):
            if line == "":
                continue
            if is_metadata_line(line):
                current_metadata.append(line)
                continue
            if is_moves_line(line):
                yield RawGame(current_metadata, line)
                current_metadata = []
                continue


def get_elo(raw_game: RawGame) -> tuple[int | None, int | None]:
    """Extract white and black ELO ratings from game metadata."""
    white_elo = None
    black_elo = None
    for line in raw_game.metadata:
        m = WHITE_ELO_PATTERN.search(line)
        if m:
            white_elo = int(m.group(1))
        m = BLACK_ELO_PATTERN.search(line)
        if m:
            black_elo = int(m.group(1))
    return white_elo, black_elo
