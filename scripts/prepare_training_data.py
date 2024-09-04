import argparse

from chess_model.util import prepare_training_data

DEFAULT_MAX_CONTEXT_LENGTH = 10
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_OUTPUT_TRAINING_DATA_FILE = "out/training-data.csv"
DEFAULT_OUTPUT_VALIDATION_DATA_FILE = "out/validation-data.csv"


def main():
    """
    Usage: poetry run prepare-training-data --input-reduced-pgn-file grandmaster.txt --output-dir out/ --max-length 50 --validation-split 0.1

    Optional arguments:
    --max-length: The maximum number of moves to include in the context. Default: 50
    --validation-split: The proportion of the data to use for validation. Default: 0.1
    """
    parser = argparse.ArgumentParser(
        description="Prepares training and validation data sets for the model training step."
    )
    parser.add_argument(
        "--input-reduced-pgn-file",
        type=str,
        help="The input file, as returned by `poetry run reduce-pgn`.",
        required=True,
    )
    parser.add_argument(
        "--output-training-data-file",
        type=str,
        help=f"Where to save the training data. Default: {DEFAULT_OUTPUT_TRAINING_DATA_FILE}",
        default=DEFAULT_OUTPUT_TRAINING_DATA_FILE,
    )
    parser.add_argument(
        "--output-validation-data-file",
        type=str,
        help=f"Where to save the validation data. Default: {DEFAULT_OUTPUT_VALIDATION_DATA_FILE}",
        default=DEFAULT_OUTPUT_VALIDATION_DATA_FILE,
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        help=f"The maximum number of moves to include in the context for the examples written to the training and validation data files. Default: {DEFAULT_MAX_CONTEXT_LENGTH}",
        default=DEFAULT_MAX_CONTEXT_LENGTH,
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        help="The proportion of the data to use for validation. Default: 0.1",
        default=DEFAULT_VALIDATION_SPLIT,
    )

    args = parser.parse_args()
    input_reduced_pgn_file = args.input_reduced_pgn_file
    output_training_data_file = args.output_training_data_file
    output_validation_data_file = args.output_validation_data_file
    max_context_length = args.max_context_length
    validation_split = args.validation_split

    prepare_training_data(
        input_reduced_pgn_file,
        output_training_data_file,
        output_validation_data_file,
        max_context_length,
        validation_split,
    )

    print(f"Training data written to: {output_training_data_file}")
    print(f"Validation data written to: {output_validation_data_file}")
