"""CLI: Download Lichess PGN database."""

import argparse

from chessgpt.data.download import download_pgn


def main():
    parser = argparse.ArgumentParser(description="Download a Lichess monthly PGN database")
    parser.add_argument("--year", type=int, required=True, help="Year (e.g. 2013)")
    parser.add_argument("--month", type=int, required=True, help="Month (1-12)")
    parser.add_argument("--output-dir", type=str, default="data/", help="Output directory")
    args = parser.parse_args()

    path = download_pgn(args.year, args.month, args.output_dir)
    print(f"PGN available at: {path}")
