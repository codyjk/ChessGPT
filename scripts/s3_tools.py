import argparse
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

from chess_model.util import (
    process_chess_moves,
    process_raw_games_from_file,
    raw_game_has_moves,
)

# Previous tree and list functions remain the same...


def process_games_to_s3(
    s3: boto3.client,
    input_bucket: str,
    output_bucket: str,
    s3_path: str,
    output_prefix: str,
    min_elo: Optional[int],
    checkmate_only: bool,
    chunk_size: int = 10000,
) -> None:
    """
    Process PGN file with streaming to S3 in chunks to minimize disk usage.
    Uses a temporary file that gets deleted after each chunk is uploaded.
    """
    # Create a unique identifier for this file's chunks
    file_id = str(uuid4())
    chunk_number = 1
    games_in_chunk = 0

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    try:
        # Download and process PGN file in streaming fashion
        response = s3.get_object(Bucket=input_bucket, Key=s3_path)
        pgn_content = response["Body"].read().decode("utf-8")

        for raw_game in process_raw_games_from_file(pgn_content):
            if not raw_game_has_moves(raw_game):
                continue

            # Check ELO requirements if specified
            if min_elo is not None:
                white_elo = next(
                    (
                        int(l.split('"')[1])
                        for l in raw_game.metadata
                        if "[WhiteElo" in l
                    ),
                    0,
                )
                black_elo = next(
                    (
                        int(l.split('"')[1])
                        for l in raw_game.metadata
                        if "[BlackElo" in l
                    ),
                    0,
                )
                if white_elo < min_elo or black_elo < min_elo:
                    continue

            # Process moves
            processed_moves = process_chess_moves(raw_game.moves)

            # Check for checkmate if required
            if checkmate_only and not processed_moves.split()[-2].endswith("#"):
                continue

            # Write to temp file
            temp_file.write(f"{processed_moves}\n")
            games_in_chunk += 1

            # If we've hit our chunk size, upload to S3 and start new chunk
            if games_in_chunk >= chunk_size:
                temp_file.flush()

                # Upload chunk to S3
                output_key = f"{output_prefix}/{Path(s3_path).stem}_{file_id}_chunk{chunk_number}.txt"
                temp_file.seek(0)
                s3.upload_file(temp_file.name, output_bucket, output_key)

                # Start new chunk
                temp_file.close()
                os.unlink(temp_file.name)
                temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
                chunk_number += 1
                games_in_chunk = 0

        # Upload final chunk if it has any games
        if games_in_chunk > 0:
            temp_file.flush()
            output_key = f"{output_prefix}/{Path(s3_path).stem}_{file_id}_chunk{chunk_number}.txt"
            temp_file.seek(0)
            s3.upload_file(temp_file.name, output_bucket, output_key)

    finally:
        # Clean up temp file
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def extract_games_command():
    """Implement s3-extract-games-from-pgns command with S3 output."""
    parser = argparse.ArgumentParser(description="Extract games from PGN files")
    parser.add_argument("input_bucket", help="Input S3 bucket name")
    parser.add_argument("output_bucket", help="Output S3 bucket name")
    parser.add_argument(
        "--output-prefix", required=True, help="S3 prefix for output files"
    )
    parser.add_argument("--min-elo", type=int, help="Minimum ELO rating")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Number of games per output file"
    )

    args = parser.parse_args()
    s3 = create_s3_client(args.input_bucket)

    # Read PGN paths from stdin
    pgn_paths = [line.strip() for line in sys.stdin]

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        work_items = [
            (
                s3,
                args.input_bucket,
                args.output_bucket,
                path,
                args.output_prefix,
                args.min_elo,
                False,
                args.chunk_size,
            )
            for path in pgn_paths
        ]
        list(
            tqdm(
                executor.map(lambda x: process_games_to_s3(*x), work_items),
                total=len(work_items),
                desc="Processing PGN files",
            )
        )


def extract_checkmates_command():
    """Implement s3-extract-checkmates-from-pgns command with S3 output."""
    parser = argparse.ArgumentParser(
        description="Extract checkmate games from PGN files"
    )
    parser.add_argument("input_bucket", help="Input S3 bucket name")
    parser.add_argument("output_bucket", help="Output S3 bucket name")
    parser.add_argument(
        "--output-prefix", required=True, help="S3 prefix for output files"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Number of games per output file"
    )

    args = parser.parse_args()
    s3 = create_s3_client(args.input_bucket)

    # Read PGN paths from stdin
    pgn_paths = [line.strip() for line in sys.stdin]

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        work_items = [
            (
                s3,
                args.input_bucket,
                args.output_bucket,
                path,
                args.output_prefix,
                None,
                True,
                args.chunk_size,
            )
            for path in pgn_paths
        ]
        list(
            tqdm(
                executor.map(lambda x: process_games_to_s3(*x), work_items),
                total=len(work_items),
                desc="Processing PGN files",
            )
        )
