import argparse
import logging
import os
import sys
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

from pgn_utils import process_chess_moves

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
    buffer_size: int = 8192  # 8KB buffer size for streaming


def list_s3_objects(bucket, prefix=""):
    """List all objects in an S3 bucket with given prefix"""
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        logging.error(f"Error accessing bucket {bucket}: {e}")
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


class StreamingPGNProcessor:
    def __init__(self, config):
        self.config = config
        self.s3 = boto3.client("s3")
        self.current_game_buffer = []

    def stream_pgn_file(self, s3_path):
        """Process a PGN file from S3 using streaming"""
        file_id = str(uuid4())
        chunk_number = 1
        games_in_chunk = 0
        temp_file = NamedTemporaryFile(mode="w+", delete=False)

        try:
            response = self.s3.get_object(Bucket=self.config.input_bucket, Key=s3_path)
            stream = response["Body"]

            buffer = ""
            while True:
                chunk = stream.read(self.config.buffer_size).decode("utf-8")
                if not chunk:
                    # Process any remaining data in buffer
                    if buffer:
                        for game in self._process_buffer(buffer, final=True):
                            if self._should_keep_game(game):
                                temp_file.write(f"{game}\n")
                                games_in_chunk += 1

                                if games_in_chunk >= self.config.chunk_size:
                                    self._upload_chunk(
                                        temp_file, chunk_number, file_id, s3_path
                                    )
                                    temp_file = NamedTemporaryFile(
                                        mode="w+", delete=False
                                    )
                                    chunk_number += 1
                                    games_in_chunk = 0
                    break

                buffer += chunk

                # Process complete games from buffer
                for game in self._process_buffer(buffer):
                    if self._should_keep_game(game):
                        temp_file.write(f"{game}\n")
                        games_in_chunk += 1

                        if games_in_chunk >= self.config.chunk_size:
                            self._upload_chunk(
                                temp_file, chunk_number, file_id, s3_path
                            )
                            temp_file = NamedTemporaryFile(mode="w+", delete=False)
                            chunk_number += 1
                            games_in_chunk = 0

            # Upload final chunk if it contains any games
            if games_in_chunk > 0:
                self._upload_chunk(temp_file, chunk_number, file_id, s3_path)

        except Exception as e:
            logging.error(f"Error processing {s3_path}: {e}")
            raise
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def _process_buffer(self, buffer, final=False):
        """Process the buffer and yield complete games"""
        lines = buffer.splitlines(True)  # Keep the newlines
        complete_game_lines = []
        new_buffer = ""

        for line in lines:
            if line.strip():  # Non-empty line
                complete_game_lines.append(line)
            elif complete_game_lines:  # Empty line after game content
                game = self._process_game_lines(complete_game_lines)
                if game:
                    yield game
                complete_game_lines = []

        # Handle any remaining complete game at the end
        if final and complete_game_lines:
            game = self._process_game_lines(complete_game_lines)
            if game:
                yield game

        return new_buffer

    def _process_game_lines(self, lines):
        """Process a single game's lines and return processed moves if valid"""
        moves_line = None
        metadata = []

        for line in lines:
            line = line.strip()
            if line.startswith("["):
                metadata.append(line)
            elif line and not line.startswith("["):
                moves_line = line
                break

        if not moves_line:
            return None

        try:
            processed_moves = process_chess_moves(moves_line)
            return processed_moves if processed_moves else None
        except Exception as e:
            logging.warning(f"Error processing game moves: {e}")
            return None

    def _should_keep_game(self, processed_moves):
        """Determine if a game should be kept based on configuration"""
        if not processed_moves:
            return False

        moves = processed_moves.split()
        if len(moves) < 2:  # Ensure there are at least a few moves
            return False

        if self.config.checkmate_only:
            # Check if the second-to-last move ends with #
            if len(moves) < 2 or not moves[-2].endswith("#"):
                return False

        return True

    def _upload_chunk(self, temp_file, chunk_number, file_id, s3_path):
        """Upload a chunk of processed games to S3"""
        try:
            temp_file.flush()
            stem = os.path.splitext(os.path.basename(s3_path))[0]
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


def process_pgn_files(config, pgn_paths):
    """Process multiple PGN files with progress tracking"""
    processor = StreamingPGNProcessor(config)
    for path in tqdm(pgn_paths, desc="Processing PGN files"):
        processor.stream_pgn_file(path.strip())


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
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=8192,
        help="Buffer size for streaming (bytes)",
    )

    args = parser.parse_args()

    config = ProcessingConfig(
        input_bucket=args.input_bucket,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        min_elo=args.min_elo,
        chunk_size=args.chunk_size,
        buffer_size=args.buffer_size,
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
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=8192,
        help="Buffer size for streaming (bytes)",
    )

    args = parser.parse_args()

    config = ProcessingConfig(
        input_bucket=args.input_bucket,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        checkmate_only=True,
        chunk_size=args.chunk_size,
        buffer_size=args.buffer_size,
    )

    pgn_paths = [line.strip() for line in sys.stdin]
    process_pgn_files(config, pgn_paths)


if __name__ == "__main__":
    # This is just here to prevent accidental execution
    print("This module should be imported and not run directly.")
    sys.exit(1)
