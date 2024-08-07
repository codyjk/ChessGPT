import os
import re
import sys
from collections import defaultdict, namedtuple

METADATA_PATTERN = re.compile(r"^\s*\[(.*)\]\s*$")
MOVE_PATTERN = re.compile(
    r"(\d+\.)\s*([BNRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[BNRQ])?(?:e\.p\.)?[+#]?|O-O(?:-O)?)\s*(?:\{[^}]*\})?\s*(?:(\d+)\.{3})?\s*([BNRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[BNRQ])?(?:e\.p\.)?[+#]?|O-O(?:-O)?)?"
)
RESULT_PATTERN = re.compile(r"(1-0|0-1|1/2-1/2)")
WHITE_ELO_PATTERN = re.compile(r'^\s*\[WhiteElo "(\d+)"\]\s*$')
BLACK_ELO_PATTERN = re.compile(r'^\s*\[BlackElo "(\d+)"\]\s*$')

RawGame = namedtuple("RawGame", ["metadata", "moves"])

INFINITY = float("inf")


def is_metadata_line(line):
    return METADATA_PATTERN.search(line)


def is_moves_line(line):
    return RESULT_PATTERN.search(line)


def raw_game_has_moves(raw_game):
    # If it has no moves, it will end in 1-0, 0-1, or 1/2-1/2 - the longest of
    # which is 7 characters long.
    return len(raw_game.moves.strip()) > 7


def process_chess_moves(input_string):
    moves = MOVE_PATTERN.findall(input_string)
    result = RESULT_PATTERN.search(input_string)

    processed_moves = []
    for move in moves:
        if move[1]:  # White's move
            processed_moves.append(move[1])
        if move[3]:  # Black's move
            processed_moves.append(move[3])
    result = result.group(1) if result else ""

    output = " ".join(processed_moves + [result]).strip()
    return output


def process_raw_games_from_file(file):
    current_metadata = []
    for line in file:
        if line == "":
            continue

        if is_metadata_line(line):
            current_metadata.append(line)
            continue

        if is_moves_line(line):
            yield RawGame(current_metadata, line)
            current_metadata = []
            continue


# The metadata contains [WhiteElo "1250"] and [BlackElo "1750"].
# Take the max, then find the elo range that contains it.
def get_elo_directory(raw_game, elo_directory_to_rating_range_dict):
    white_elo = None
    black_elo = None

    for metadata_line in raw_game.metadata:
        if WHITE_ELO_PATTERN.search(metadata_line):
            white_elo = WHITE_ELO_PATTERN.search(metadata_line)
        elif BLACK_ELO_PATTERN.search(metadata_line):
            black_elo = BLACK_ELO_PATTERN.search(metadata_line)

    if not white_elo or not black_elo:
        return "unknown"

    white_elo = int(white_elo.group(1))
    black_elo = int(black_elo.group(1))
    target_elo = max(white_elo, black_elo)

    for directory, (min_elo, max_elo) in elo_directory_to_rating_range_dict.items():
        if min_elo < target_elo <= max_elo:
            return directory

    return "unknown"