import argparse

from chess_model import prepare_training_data

# 50 moves max by default
DEFAULT_MAX_LENGTH = 50

# Default validation split is 10%
DEFAULT_VALIDATION_SPLIT = 0.1


def main():
    """
    Usage: poetry run prepare-training-data --input-file grandmaster.txt --output-dir out/ --max-length 50 --validation-split 0.1

    Optional arguments:
    --max-length: The maximum number of moves to include in the context. Default: 50
    --validation-split: The proportion of the data to use for validation. Default: 0.1
    """
    parser = argparse.ArgumentParser(description="Prepare training data for the model.")
    parser.add_argument("--input-file", type=str, help="The input file.", required=True)
    parser.add_argument(
        "--output-dir", type=str, help="The output directory.", required=True
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="The maximum number of moves to include in the context. Default: 50",
        default=DEFAULT_MAX_LENGTH,
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        help="The proportion of the data to use for validation. Default: 0.1",
        default=DEFAULT_VALIDATION_SPLIT,
    )

    args = parser.parse_args()
    input_file = args.input_file
    output_directory = args.output_dir
    max_length = args.max_length
    validation_split = args.validation_split

    prepare_training_data(
        input_file,
        f"{output_directory}/training-data.csv",
        f"{output_directory}/validation-data.csv",
        max_length,
        validation_split,
    )
