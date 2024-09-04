import argparse

from chess_model.model import ChessTokenizer

DEFAULT_TOKENIZER_OUTPUT_FILE = "out/chess_tokenizer.json"


def main():
    """
    Usage: poetry run fit-and-save-tokenizer --input-training-data-file out/training-data.csv
    """

    parser = argparse.ArgumentParser(description="Fit and save the tokenizer.")
    parser.add_argument(
        "--input-training-data-file",
        type=str,
        help="The input training data file, as returned by `poetry run prepare-training-data`",
        required=True,
    )
    parser.add_argument(
        "--output-tokenizer-file",
        type=str,
        help=f"Where to save tokenizer state. Default: {DEFAULT_TOKENIZER_OUTPUT_FILE}",
        required=False,
        default=DEFAULT_TOKENIZER_OUTPUT_FILE,
    )

    args = parser.parse_args()
    print("Fitting tokenizer...")
    tokenizer = ChessTokenizer.fit(args.input_training_data_file)
    print(f"Tokenizer initialized with vocab_size={tokenizer.vocab_size}")
    tokenizer.save(args.output_tokenizer_file)
    print(f"Tokenizer saved to: {args.output_tokenizer_file}")
