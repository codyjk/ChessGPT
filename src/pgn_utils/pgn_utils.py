import re
from typing import Iterator, List, NamedTuple


class RawGame(NamedTuple):
    metadata: List[str]
    moves: str


METADATA_PATTERN = re.compile(r"^\s*\[(.*)\]\s*$")
MOVE_PATTERN = re.compile(
    r"(\d+\.)\s*([BNRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[BNRQ])?(?:e\.p\.)?[+#]?|O-O(?:-O)?)\s*(?:\{[^}]*\})?\s*(?:(\d+)\.{3})?\s*([BNRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[BNRQ])?(?:e\.p\.)?[+#]?|O-O(?:-O)?)?"
)
OUTCOME_PATTERN = re.compile(r"(1-0|0-1|1/2-1/2)")


def is_metadata_line(line: str) -> bool:
    """Check if a line contains metadata (starts with '[')."""
    return bool(METADATA_PATTERN.search(line))


def is_moves_line(line: str) -> bool:
    """Check if a line contains moves (contains game outcome)."""
    return bool(OUTCOME_PATTERN.search(line))


def process_chess_moves(input_string: str) -> str:
    """Process chess moves from a string into space-separated format."""
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
    """Check if a game has the minimum number of moves."""
    moves_and_outcome = raw_game.moves.split(" ")
    moves, outcome = moves_and_outcome[:-1], moves_and_outcome[-1]
    return len(moves) >= min_moves and bool(OUTCOME_PATTERN.search(outcome))


def process_raw_games_from_file(content: str) -> Iterator[RawGame]:
    """Process PGN content and yield RawGame objects."""
    current_metadata = []

    for line in content.splitlines():
        if not line.strip():
            continue

        if is_metadata_line(line):
            current_metadata.append(line)
            continue

        if is_moves_line(line):
            yield RawGame(current_metadata, line)
            current_metadata = []
            continue
