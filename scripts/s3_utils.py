import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

from pgn_utils import (
    process_chess_moves,
    process_raw_games_from_file,
    raw_game_has_moves,
)


def create_s3_client(bucket: str):
    """Create and verify S3 client with bucket access."""
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        print(f"Error accessing bucket {bucket}: {e}")
        sys.exit(1)
    return s3


def format_size(size: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size = int(size / 1024)
    raise ValueError


def list_bucket_contents(s3, bucket: str, prefix: str = "") -> Iterator[dict]:
    """List all objects in bucket with given prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            yield from page["Contents"]


def print_tree(s3, bucket: str, prefix: str = "", indent: str = ""):
    """Print tree structure of S3 bucket."""
    objects = list(list_bucket_contents(s3, bucket, prefix))

    for i, obj in enumerate(sorted(objects, key=lambda x: x["Key"])):
        is_last = i == len(objects) - 1
        current_indent = indent + ("└── " if is_last else "├── ")

        # Get just the last part of the path
        name = obj["Key"].split("/")[-1] or obj["Key"]
        size = format_size(obj["Size"])
        print(f"{current_indent}{name} ({size})")

        next_indent = indent + ("    " if is_last else "│   ")
        if obj["Key"].endswith("/"):
            print_tree(s3, bucket, obj["Key"], next_indent)


def download_and_process_pgn(args: tuple) -> None:
    """Download and process a single PGN file."""
    s3, bucket, s3_path, output_dir, min_elo, checkmate_only = args

    local_filename = os.path.join(
        output_dir, f"{s3_path.replace('/', '_')}.processed.txt"
    )

    if os.path.exists(local_filename):
        return

    # Download PGN file
    try:
        response = s3.get_object(Bucket=bucket, Key=s3_path)
        pgn_content = response["Body"].read().decode("utf-8")
    except ClientError as e:
        print(f"Error downloading {s3_path}: {e}")
        return

    # Process games
    with open(local_filename, "w") as f:
        for raw_game in process_raw_games_from_file(pgn_content):
            # Skip games without enough moves
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

            f.write(f"{processed_moves}\n")


def tree_command():
    """Implement s3-tree command."""
    parser = argparse.ArgumentParser(
        description="List S3 bucket contents in tree format"
    )
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("--prefix", help="Optional prefix to start from", default="")

    args = parser.parse_args()
    s3 = create_s3_client(args.bucket)

    print(f"{args.bucket}")
    print_tree(s3, args.bucket, args.prefix)


def list_pgns_command():
    """Implement s3-list-pgns command."""
    parser = argparse.ArgumentParser(description="List PGN files in S3 bucket")
    parser.add_argument("bucket", help="S3 bucket name")

    args = parser.parse_args()
    s3 = create_s3_client(args.bucket)

    for obj in list_bucket_contents(s3, args.bucket):
        if obj["Key"].lower().endswith(".pgn"):
            print(obj["Key"])


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
    """Process games from a PGN file and write to S3 in chunks."""
    file_id = str(uuid4())
    chunk_number = 1
    games_in_chunk = 0
    total_games = 0

    # Create temp file for current chunk
    temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)

    try:
        print(f"\nProcessing {s3_path}...")
        response = s3.get_object(Bucket=input_bucket, Key=s3_path)
        pgn_content = response["Body"].read().decode("utf-8")

        # Count total games for progress bar
        game_count = sum(1 for _ in process_raw_games_from_file(pgn_content))
        print(f"Found {game_count} games")

        # Process games with progress bar
        with tqdm(total=game_count, desc="Processing games", unit="game") as pbar:
            for raw_game in process_raw_games_from_file(pgn_content):
                if not raw_game_has_moves(raw_game):
                    pbar.update(1)
                    continue

                if min_elo is not None:
                    # Check ELO from metadata
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
                        pbar.update(1)
                        continue

                processed_moves = process_chess_moves(raw_game.moves)
                if not processed_moves:
                    pbar.update(1)
                    continue

                if checkmate_only and not processed_moves.split()[-2].endswith("#"):
                    pbar.update(1)
                    continue

                temp_file.write(f"{processed_moves}\n")
                games_in_chunk += 1
                total_games += 1

                if games_in_chunk >= chunk_size:
                    temp_file.flush()
                    output_key = f"{output_prefix}/{Path(s3_path).stem}_{file_id}_chunk{chunk_number}.txt"
                    temp_file.seek(0)
                    s3.upload_file(temp_file.name, output_bucket, output_key)

                    print(
                        f"\nUploaded chunk {chunk_number} ({games_in_chunk} games) to {output_key}"
                    )

                    temp_file.close()
                    os.unlink(temp_file.name)
                    temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
                    chunk_number += 1
                    games_in_chunk = 0

                pbar.update(1)

        if games_in_chunk > 0:
            temp_file.flush()
            output_key = f"{output_prefix}/{Path(s3_path).stem}_{file_id}_chunk{chunk_number}.txt"
            temp_file.seek(0)
            s3.upload_file(temp_file.name, output_bucket, output_key)
            print(
                f"\nUploaded final chunk {chunk_number} ({games_in_chunk} games) to {output_key}"
            )

        print(f"Total games processed and uploaded: {total_games}")

    finally:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def extract_games_command():
    """Implement s3-extract-games-from-pgns command"""
    parser = argparse.ArgumentParser(description="Extract games from PGN files")
    parser.add_argument("input_bucket", help="Input S3 bucket name")
    parser.add_argument(
        "output_bucket",
        help="Output S3 bucket and optional path (e.g., 'my-bucket' or 'my-bucket/path/to/output')",
    )
    parser.add_argument("--min-elo", type=int, help="Minimum ELO rating")
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Number of games per output file"
    )

    args = parser.parse_args()
    s3 = create_s3_client(args.input_bucket)

    # Split bucket and path
    bucket_parts = args.output_bucket.split("/", 1)
    output_bucket = bucket_parts[0]
    output_prefix = bucket_parts[1] if len(bucket_parts) > 1 else ""

    # Process each PGN file sequentially
    for pgn_path in sys.stdin:
        pgn_path = pgn_path.strip()
        process_games_to_s3(
            s3,
            args.input_bucket,
            output_bucket,
            pgn_path,
            output_prefix,
            args.min_elo,
            False,
            args.chunk_size,
        )


def extract_checkmates_command():
    """Implement s3-extract-checkmates-from-pgns command"""
    parser = argparse.ArgumentParser(
        description="Extract checkmate games from PGN files"
    )
    parser.add_argument("input_bucket", help="Input S3 bucket name")
    parser.add_argument(
        "output_bucket",
        help="Output S3 bucket and optional path (e.g., 'my-bucket' or 'my-bucket/path/to/output')",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Number of games per output file"
    )

    args = parser.parse_args()
    s3 = create_s3_client(args.input_bucket)

    # Split bucket and path
    bucket_parts = args.output_bucket.split("/", 1)
    output_bucket = bucket_parts[0]
    output_prefix = bucket_parts[1] if len(bucket_parts) > 1 else ""

    # Process each PGN file sequentially
    for pgn_path in sys.stdin:
        pgn_path = pgn_path.strip()
        process_games_to_s3(
            s3,
            args.input_bucket,
            output_bucket,
            pgn_path,
            output_prefix,
            None,
            True,
            args.chunk_size,
        )
