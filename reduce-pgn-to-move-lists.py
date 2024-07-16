#!/usr/bin/env python3

import os
import re
import sys

from collections import namedtuple

METADATA_PATTERN = re.compile(r'^\s*\[(.*)\]\s*$')
MOVE_PATTERN = re.compile(r'(\d+\.)\s*([BNRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[BNRQ])?(?:e\.p\.)?[+#]?|O-O(?:-O)?)\s*(?:\{[^}]*\})?\s*(?:(\d+)\.{3})?\s*([BNRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[BNRQ])?(?:e\.p\.)?[+#]?|O-O(?:-O)?)?')
RESULT_PATTERN = re.compile(r'(1-0|0-1|1/2-1/2)')

RawGame = namedtuple('RawGame', ['metadata', 'moves'])


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
        if line == '':
            continue

        if is_metadata_line(line):
            current_metadata.append(line)
            continue

        if is_moves_line(line):
            yield RawGame(current_metadata, line)
            current_metadata = []
            continue


def main():
    try:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    except IndexError:
        print("Usage: python3 reduce-pgn-to-move-lists.py input-file output-file")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        sys.exit(1)

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists.")
        sys.exit(1)

    target = open(output_file, 'w')

    written = 0
    with open(input_file, 'r') as file:
        print("Processing file")
        for raw_game in process_raw_games_from_file(file):
            processed_moves = process_chess_moves(raw_game.moves)
            if raw_game_has_moves(raw_game):
                target.write(processed_moves + '\n')
                written += 1

    print(f"Processed {written} games.")


if __name__ == '__main__':
    main()
