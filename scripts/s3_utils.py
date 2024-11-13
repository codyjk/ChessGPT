import argparse
import logging
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import _TemporaryFileWrapper
from typing import Dict, Iterator, List, Optional, TextIO, Tuple, Union
from uuid import uuid4

import boto3
import psutil
from botocore.exceptions import ClientError
from tqdm import tqdm

from pgn_utils import RawGame, process_chess_moves, raw_game_has_moves

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("s3_extract.log"), logging.StreamHandler()],
)

TempFile = Union[TextIO, _TemporaryFileWrapper[str]]


@dataclass
class ProcessingStats:
    """Track statistics for the processing job."""

    total_games: int = 0
    games_in_chunk: int = 0
    chunk_number: int = 1
    start_time: datetime = field(default_factory=datetime.now)


def create_s3_client(bucket: str):
    """Create and verify S3 client with bucket access."""
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        print(f"Error accessing bucket {bucket}: {e}")
        sys.exit(1)
    return s3


def get_memory_usage() -> str:
    """Get current memory usage of the process."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return f"{mem_info.rss / (1024*1024):.2f} MB"


def list_s3_objects(bucket: str, prefix: str = "") -> Iterator[str]:
    """List all objects in an S3 bucket with given prefix."""
    s3 = create_s3_client(bucket)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                yield obj["Key"]


def build_tree(paths: List[str]) -> Dict:
    """Build a tree structure from a list of paths."""
    root = {}
    for path in paths:
        current = root
        for part in path.split("/"):
            if part:
                current = current.setdefault(part, {})
    return root


def print_tree(node: Dict, prefix: str = "", is_last: bool = True) -> None:
    """Print a tree structure."""
    if not node:
        return

    items = list(node.items())
    count = len(items)

    for i, (name, subtree) in enumerate(items):
        is_last_item = i == count - 1
        if is_last_item:
            branch = "└── "
            new_prefix = prefix + "    "
        else:
            branch = "├── "
            new_prefix = prefix + "│   "

        print(f"{prefix}{branch}{name}")
        print_tree(subtree, new_prefix, is_last_item)


def is_pgn_file(path: str) -> bool:
    """Check if a file path ends with .pgn"""
    return path.lower().endswith(".pgn")


def upload_chunk(
    s3: boto3.client,
    temp_file: TempFile,
    output_bucket: str,
    output_prefix: str,
    s3_path: str,
    file_id: str,
    chunk_number: int,
    games_in_chunk: int,
) -> None:
    """Upload a chunk of processed games to S3."""
    try:
        temp_file.flush()
        stem = Path(s3_path).stem
        output_key = f"{output_prefix}/{stem}_{file_id}_chunk{chunk_number}.txt"
        temp_file.seek(0)
        s3.upload_file(temp_file.name, output_bucket, output_key)
        logging.info(
            f"\nUploaded chunk {chunk_number} ({games_in_chunk} games) to {output_key}"
        )
    except Exception as e:
        logging.error(f"Error uploading chunk {chunk_number}:")
        logging.error(traceback.format_exc())
        raise
    finally:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def process_single_game(
    raw_game: RawGame, min_elo: Optional[int], checkmate_only: bool
) -> Optional[str]:
    """Process a single game and return processed moves if game should be included."""
    try:
        if not raw_game_has_moves(raw_game):
            return None

        if min_elo is not None:
            white_elo = next(
                (int(l.split('"')[1]) for l in raw_game.metadata if "[WhiteElo" in l),
                0,
            )
            black_elo = next(
                (int(l.split('"')[1]) for l in raw_game.metadata if "[BlackElo" in l),
                0,
            )
            if white_elo < min_elo or black_elo < min_elo:
                return None

        processed_moves = process_chess_moves(raw_game.moves)
        if not processed_moves:
            return None

        if checkmate_only and not processed_moves.split()[-2].endswith("#"):
            return None

        return processed_moves

    except Exception as e:
        logging.error(f"Error processing game: {str(e)}")
        logging.error(traceback.format_exc())
        return None


def stream_pgn_games(content_stream: Iterator[bytes]) -> Iterator[RawGame]:
    """Stream PGN games from a byte stream."""
    buffer = ""
    current_game_metadata = []
    current_game_moves = ""

    for chunk in content_stream:
        buffer += chunk.decode("utf-8")
        lines = buffer.split("\n")

        # Keep the last line in buffer if it's incomplete
        buffer = lines[-1]
        lines = lines[:-1]

        for line in lines:
            line = line.strip()

            if not line:
                if current_game_metadata and current_game_moves:
                    yield RawGame(current_game_metadata, current_game_moves)
                    current_game_metadata = []
                    current_game_moves = ""
                continue

            if line.startswith("["):
                current_game_metadata.append(line)
            else:
                current_game_moves += line + " "

    # Process any remaining content in the buffer
    if buffer:
        current_game_moves += buffer

    # Yield final game if exists
    if current_game_metadata and current_game_moves:
        yield RawGame(current_game_metadata, current_game_moves)


def maybe_upload_chunk(
    stats: ProcessingStats,
    temp_file: TempFile,
    s3: boto3.client,
    output_bucket: str,
    output_prefix: str,
    s3_path: str,
    file_id: str,
    chunk_size: int,
) -> Tuple[Union[TextIO, _TemporaryFileWrapper], ProcessingStats]:
    """Upload chunk if needed and return new temp file and updated stats."""
    if stats.games_in_chunk >= chunk_size:
        upload_chunk(
            s3,
            temp_file,
            output_bucket,
            output_prefix,
            s3_path,
            file_id,
            stats.chunk_number,
            stats.games_in_chunk,
        )
        logging.info(
            f"Memory usage after chunk {stats.chunk_number}: {get_memory_usage()}"
        )

        stats.chunk_number += 1
        stats.games_in_chunk = 0
        return tempfile.NamedTemporaryFile(mode="w+", delete=False), stats

    return temp_file, stats


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
    stats = ProcessingStats()

    # Create temp file for current chunk
    temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)

    try:
        logging.info(f"Starting to process {s3_path}")
        logging.info(f"Memory usage before download: {get_memory_usage()}")

        # Get file size without downloading
        response = s3.head_object(Bucket=input_bucket, Key=s3_path)
        file_size = response["ContentLength"]
        logging.info(f"File size: {file_size / (1024*1024):.2f} MB")

        # Stream and process the file
        logging.info("Streaming file contents...")
        response = s3.get_object(Bucket=input_bucket, Key=s3_path)
        stream = response["Body"].iter_chunks(chunk_size=1024 * 1024)  # 1MB chunks

        for raw_game in stream_pgn_games(stream):
            processed_moves = process_single_game(raw_game, min_elo, checkmate_only)
            if processed_moves:
                temp_file.write(f"{processed_moves}\n")
                stats.games_in_chunk += 1
                stats.total_games += 1

                temp_file, stats = maybe_upload_chunk(
                    stats,
                    temp_file,
                    s3,
                    output_bucket,
                    output_prefix,
                    s3_path,
                    file_id,
                    chunk_size,
                )

        # Upload final chunk if needed
        if stats.games_in_chunk > 0:
            upload_chunk(
                s3,
                temp_file,
                output_bucket,
                output_prefix,
                s3_path,
                file_id,
                stats.chunk_number,
                stats.games_in_chunk,
            )

        duration = (datetime.now() - stats.start_time).total_seconds()
        logging.info(f"Total games processed and uploaded: {stats.total_games}")
        logging.info(f"Processing time: {duration:.2f} seconds")
        logging.info(f"Processing rate: {stats.total_games/duration:.2f} games/second")
        logging.info(f"Final memory usage: {get_memory_usage()}")

    except Exception as e:
        logging.error(f"Fatal error processing {s3_path}:")
        logging.error(traceback.format_exc())
        raise

    finally:
        try:
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception as e:
            logging.error(f"Error cleaning up temp file: {str(e)}")


def tree_command():
    """Implement s3-tree command"""
    parser = argparse.ArgumentParser(description="Display S3 bucket contents as a tree")
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="Optional prefix to start from")

    args = parser.parse_args()

    print(f"{args.bucket}/")
    paths = list(list_s3_objects(args.bucket, args.prefix))
    tree = build_tree(paths)
    print_tree(tree)


def list_pgns_command():
    """Implement s3-list-pgns command"""
    parser = argparse.ArgumentParser(description="List all PGN files in an S3 bucket")
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="Optional prefix to start from")

    args = parser.parse_args()

    for path in list_s3_objects(args.bucket, args.prefix):
        if is_pgn_file(path):
            print(path)


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
