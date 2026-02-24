"""CLI: Prepare training data from PGN files."""

import argparse

from chessgpt.data.prepare import prepare_training_data, print_stats
from chessgpt.model.tokenizer import ChessTokenizer


def main():
    parser = argparse.ArgumentParser(description="Process PGN to enriched training CSV")
    parser.add_argument("--input-pgn", type=str, required=True, help="Input PGN file")
    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV file")
    parser.add_argument("--min-elo", type=int, default=1800, help="Minimum ELO for both players")
    parser.add_argument("--min-moves", type=int, default=10, help="Minimum moves per game")
    parser.add_argument(
        "--fit-tokenizer",
        type=str,
        default=None,
        help="If set, fit and save tokenizer to this path",
    )
    args = parser.parse_args()

    counts = prepare_training_data(
        input_pgn=args.input_pgn,
        output_csv=args.output_csv,
        min_elo=args.min_elo,
        min_moves=args.min_moves,
    )
    print_stats(counts)

    if args.fit_tokenizer:
        print(f"\nFitting tokenizer from {args.output_csv}...")
        tokenizer = ChessTokenizer.fit(args.output_csv)
        tokenizer.save(args.fit_tokenizer)
        print(f"Tokenizer saved to {args.fit_tokenizer} (vocab_size={tokenizer.vocab_size})")
