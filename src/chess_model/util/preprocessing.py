import csv
import random

from tqdm import tqdm

from pgn_utils.count_lines import count_lines_fast


def process_game(game, max_context_length):
    game = game.split()
    moves = game[:-1]  # Remove the outcome from the move list
    outcome = game[-1]
    context = " ".join(moves[:max_context_length])
    is_checkmate = "1" if context[-1].endswith("#") else "0"
    yield context, is_checkmate, outcome


def prepare_training_data(
    input_reduced_pgn_file,
    output_training_data_file,
    output_validation_data_file,
    max_context_length,
    validation_split,
):
    with open(output_training_data_file, "w", newline="") as train_outfile, open(
        output_validation_data_file, "w", newline=""
    ) as val_outfile:
        train_writer = csv.writer(train_outfile)
        val_writer = csv.writer(val_outfile)

        headers = ["context", "is_checkmate", "outcome"]
        train_writer.writerow(headers)
        val_writer.writerow(headers)

        # Count total lines for progress bar
        total_lines = count_lines_fast(input_reduced_pgn_file)

        with open(input_reduced_pgn_file, "r") as infile:
            for line in tqdm(infile, total=total_lines, desc="Processing games"):
                game = line.strip()
                for context, is_checkmate, outcome in process_game(
                    game, max_context_length
                ):
                    # Decide whether to write to train or val file
                    row = [context, is_checkmate, outcome]
                    if random.random() < validation_split:
                        val_writer.writerow(row)
                    else:
                        train_writer.writerow(row)
