import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

from pgn_utils import (
    process_chess_moves,
    process_raw_games_from_file,
    raw_game_has_moves,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("s3_extract.log"), logging.StreamHandler()],
)


@dataclass
class ProcessingConfig:
    """Configuration for PGN processing"""

    input_bucket: str
    output_bucket: str
    output_prefix: str
    min_elo: int = 0
    checkmate_only: bool = False
    chunk_size: int = 10000


def list_s3_objects(bucket, prefix=""):
    """List all objects in an S3 bucket with given prefix"""
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        print(f"Error accessing bucket {bucket}: {e}")
        sys.exit(1)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                yield obj["Key"]


def build_tree(paths):
    """Build a tree structure from a list of paths"""
    root = {}
    for path in paths:
        current = root
        for part in path.split("/"):
            if part:
                current = current.setdefault(part, {})
    return root


def print_tree(node, prefix="", is_last=True):
    """Print a tree structure"""
    if not node:
        return

    items = list(node.items())
    count = len(items)

    for i, (name, subtree) in enumerate(items):
        is_last_item = i == count - 1
        branch = "└── " if is_last_item else "├── "
        new_prefix = prefix + "    " if is_last_item else prefix + "│   "

        print(f"{prefix}{branch}{name}")
        print_tree(subtree, new_prefix, is_last_item)


def is_pgn_file(path):
    """Check if a file path ends with .pgn"""
    return path.lower().endswith(".pgn")


class S3PGNProcessor:
    """Handles processing of PGN files from S3"""

    def __init__(self, config):
        self.config = config
        self.s3 = self._create_s3_client()

    def _create_s3_client(self):
        """Create and verify S3 client"""
        s3 = boto3.client("s3")
        try:
            s3.head_bucket(Bucket=self.config.input_bucket)
        except ClientError as e:
            logging.error(f"Error accessing bucket {self.config.input_bucket}: {e}")
            sys.exit(1)
        return s3

    def _should_include_game(self, raw_game):
        """Check if a game meets inclusion criteria"""
        if not raw_game_has_moves(raw_game):
            return False

        white_elo = next(
            (int(l.split('"')[1]) for l in raw_game.metadata if "[WhiteElo" in l), 0
        )
        black_elo = next(
            (int(l.split('"')[1]) for l in raw_game.metadata if "[BlackElo" in l), 0
        )
        if white_elo < self.config.min_elo or black_elo < self.config.min_elo:
            return False

        processed_moves = process_chess_moves(raw_game.moves)
        if not processed_moves:
            return False

        if self.config.checkmate_only and not processed_moves.split()[-2].endswith("#"):
            return False

        return True

    def _upload_chunk(self, temp_file, chunk_number, file_id, s3_path):
        """Upload a chunk of processed games to S3"""
        try:
            temp_file.flush()
            stem = Path(s3_path).stem
            output_key = (
                f"{self.config.output_prefix}/{stem}_{file_id}_chunk{chunk_number}.txt"
            )
            temp_file.seek(0)
            self.s3.upload_file(temp_file.name, self.config.output_bucket, output_key)
            logging.info(f"Uploaded chunk {chunk_number} to {output_key}")
        except Exception as e:
            logging.error(f"Error uploading chunk {chunk_number}: {e}")
            raise
        finally:
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def process_file(self, s3_path):
        """Process a single PGN file from S3"""
        file_id = str(uuid4())
        chunk_number = 1
        games_in_chunk = 0
        temp_file = NamedTemporaryFile(mode="w+", delete=False)

        try:
            # Download and process PGN file
            response = self.s3.get_object(Bucket=self.config.input_bucket, Key=s3_path)
            content = response["Body"].read().decode("utf-8")

            for raw_game in process_raw_games_from_file(content):
                if not self._should_include_game(raw_game):
                    continue

                processed_moves = process_chess_moves(raw_game.moves)
                temp_file.write(f"{processed_moves}\n")
                games_in_chunk += 1

                if games_in_chunk >= self.config.chunk_size:
                    self._upload_chunk(temp_file, chunk_number, file_id, s3_path)
                    temp_file = NamedTemporaryFile(mode="w+", delete=False)
                    chunk_number += 1
                    games_in_chunk = 0

            # Upload final chunk if needed
            if games_in_chunk > 0:
                self._upload_chunk(temp_file, chunk_number, file_id, s3_path)

        except Exception as e:
            logging.error(f"Error processing {s3_path}: {e}")
            raise
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


def process_pgn_files(config, pgn_paths):
    """Process multiple PGN files with progress tracking"""
    processor = S3PGNProcessor(config)
    for path in tqdm(pgn_paths, desc="Processing PGN files"):
        processor.process_file(path.strip())


def tree_command():
    """Display S3 bucket contents as a tree"""
    parser = argparse.ArgumentParser(description="Display S3 bucket contents as a tree")
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="Optional prefix to start from")

    args = parser.parse_args()

    print(f"{args.bucket}/")
    paths = list(list_s3_objects(args.bucket, args.prefix))
    tree = build_tree(paths)
    print_tree(tree)


def list_pgns_command():
    """List all PGN files in an S3 bucket"""
    parser = argparse.ArgumentParser(description="List all PGN files in an S3 bucket")
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="Optional prefix to start from")

    args = parser.parse_args()

    for path in list_s3_objects(args.bucket, args.prefix):
        if is_pgn_file(path):
            print(path)


def extract_games_command():
    """Extract games from PGN files"""
    parser = argparse.ArgumentParser(description="Extract games from PGN files")
    parser.add_argument("input_bucket", help="Input S3 bucket name")
    parser.add_argument("output_bucket", help="Output S3 bucket name")
    parser.add_argument(
        "--output-prefix", required=True, help="S3 prefix for output files"
    )
    parser.add_argument("--min-elo", type=int, help="Minimum ELO rating")
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Number of games per output file"
    )

    args = parser.parse_args()

    config = ProcessingConfig(
        input_bucket=args.input_bucket,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        min_elo=args.min_elo,
        chunk_size=args.chunk_size,
    )

    pgn_paths = [line.strip() for line in sys.stdin]
    process_pgn_files(config, pgn_paths)


def extract_checkmates_command():
    """Extract checkmate games from PGN files"""
    parser = argparse.ArgumentParser(
        description="Extract checkmate games from PGN files"
    )
    parser.add_argument("input_bucket", help="Input S3 bucket name")
    parser.add_argument("output_bucket", help="Output S3 bucket name")
    parser.add_argument(
        "--output-prefix", required=True, help="S3 prefix for output files"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Number of games per output file"
    )

    args = parser.parse_args()

    config = ProcessingConfig(
        input_bucket=args.input_bucket,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        checkmate_only=True,
        chunk_size=args.chunk_size,
    )

    pgn_paths = [line.strip() for line in sys.stdin]
    process_pgn_files(config, pgn_paths)


if __name__ == "__main__":
    import argparse
