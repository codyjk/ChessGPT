"""CLI: Prepare training data from PGN files."""

import argparse
import csv
import json
import random
from pathlib import Path

from chessgpt.data.prepare import prepare_training_data, print_stats
from chessgpt.model.tokenizer import ChessTokenizer


def _derive_val_path(output_csv: str) -> str:
    """Derive the validation CSV path from the training CSV path.

    'data/train_large.csv' -> 'data/val_large.csv'
    'data/games.csv' -> 'data/games_val.csv'
    """
    output_path = Path(output_csv)
    train_stem = output_path.stem
    val_stem = train_stem.replace("train", "val") if "train" in train_stem else train_stem + "_val"
    return str(output_path.with_stem(val_stem))


def _invoke_lambda(
    year: int, month: int, bucket: str, min_elo: int, function_name: str, region: str | None
) -> None:
    """Invoke the prepare Lambda function asynchronously."""
    import boto3

    client = boto3.client("lambda", **({"region_name": region} if region else {}))
    payload = {"year": year, "month": month, "bucket": bucket, "min_elo": min_elo}
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=json.dumps(payload),
    )
    status = response["StatusCode"]
    if status == 202:
        print(f"Lambda invoked for {year}-{month:02d} (async, check CloudWatch for progress)")
    else:
        print(f"Lambda invoke returned unexpected status: {status}")


def _merge_from_s3(
    year: int, bucket: str, output_csv: str, fit_tokenizer: str | None, val_split: float
) -> None:
    """Download per-month CSVs from S3, merge, split train/val, optionally fit tokenizer."""
    import boto3

    s3 = boto3.client("s3")

    # List all prepared CSVs for the given year
    prefix = f"prepared/lichess_{year:04d}-"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        print(f"No prepared CSVs found in s3://{bucket}/{prefix}*")
        return

    keys = sorted(obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".csv"))
    print(f"Found {len(keys)} prepared CSVs for {year}")

    # Download and merge
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[list[str]] = []
    header: list[str] | None = None

    for key in keys:
        local_path = Path(f"/tmp/{Path(key).name}")
        print(f"Downloading s3://{bucket}/{key}...")
        s3.download_file(bucket, str(key), str(local_path))

        with open(local_path) as f:
            reader = csv.reader(f)
            file_header = next(reader)
            if header is None:
                header = file_header
            for row in reader:
                all_rows.append(row)

        local_path.unlink()

    if header is None:
        print("No data found in downloaded CSVs")
        return

    print(f"Total games: {len(all_rows)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_rows)

    val_count = int(len(all_rows) * val_split)
    train_rows = all_rows[val_count:]
    val_rows = all_rows[:val_count]

    # Write train CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_rows)
    print(f"Train: {len(train_rows)} games -> {output_csv}")

    # Write val CSV (same directory, derive name from train filename)
    val_csv = _derive_val_path(output_csv)
    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(val_rows)
    print(f"Val:   {len(val_rows)} games -> {val_csv}")

    # Fit tokenizer on train set
    if fit_tokenizer:
        print(f"\nFitting tokenizer from {output_csv}...")
        tokenizer = ChessTokenizer.fit(output_csv)
        tokenizer.save(fit_tokenizer)
        print(f"Tokenizer saved to {fit_tokenizer} (vocab_size={tokenizer.vocab_size})")


def _upload_merged(
    output_csv: str, fit_tokenizer: str | None, bucket: str, region: str | None
) -> None:
    """Upload merged train CSV, val CSV, and tokenizer to s3://bucket/merged/."""
    import boto3

    s3 = boto3.client("s3", **({"region_name": region} if region else {}))

    val_csv = _derive_val_path(output_csv)

    files_to_upload: list[tuple[str, str]] = [
        (output_csv, f"merged/{Path(output_csv).name}"),
        (val_csv, f"merged/{Path(val_csv).name}"),
    ]
    if fit_tokenizer:
        files_to_upload.append((fit_tokenizer, f"merged/{Path(fit_tokenizer).name}"))

    uploaded = 0
    for local_path, s3_key in files_to_upload:
        if not Path(local_path).exists():
            print(f"  Warning: {local_path} not found, skipping upload")
            continue
        print(f"  Uploading {local_path} -> s3://{bucket}/{s3_key}")
        s3.upload_file(local_path, bucket, s3_key)
        uploaded += 1

    print(f"Uploaded {uploaded} files to s3://{bucket}/merged/")


def main():
    parser = argparse.ArgumentParser(description="Process PGN to enriched training CSV")
    parser.add_argument("--input-pgn", type=str, help="Input PGN or .pgn.zst file")
    parser.add_argument("--output-csv", type=str, help="Output CSV file")
    parser.add_argument("--min-elo", type=int, default=1800, help="Minimum ELO for both players")
    parser.add_argument("--min-moves", type=int, default=10, help="Minimum moves per game")
    parser.add_argument(
        "--fit-tokenizer",
        type=str,
        default=None,
        help="If set, fit and save tokenizer to this path",
    )

    # Cloud modes
    parser.add_argument(
        "--cloud", action="store_true", help="Invoke AWS Lambda instead of running locally"
    )
    parser.add_argument("--bucket", type=str, default=None, help="S3 bucket name")
    parser.add_argument(
        "--year", type=int, default=None, help="Year for --cloud or --merge-from-s3"
    )
    parser.add_argument("--month", type=int, default=None, help="Month for --cloud mode")
    parser.add_argument(
        "--function-name",
        type=str,
        default="chessgpt-prepare",
        help="Lambda function name (default: chessgpt-prepare)",
    )
    parser.add_argument(
        "--merge-from-s3",
        action="store_true",
        help="Download per-month CSVs from S3, merge, and split train/val",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.05, help="Validation split ratio (default: 0.05)"
    )
    parser.add_argument(
        "--region", type=str, default=None, help="AWS region (default: from AWS config)"
    )
    parser.add_argument(
        "--upload-merged",
        action="store_true",
        help="After merging, upload train/val CSV and tokenizer to s3://bucket/merged/",
    )
    args = parser.parse_args()

    if args.upload_merged and not args.merge_from_s3:
        parser.error("--upload-merged requires --merge-from-s3")

    if args.cloud:
        if not args.bucket:
            parser.error("--cloud requires --bucket")
        if not args.year or not args.month:
            parser.error("--cloud requires --year and --month")
        _invoke_lambda(
            args.year, args.month, args.bucket, args.min_elo, args.function_name, args.region
        )
        return

    if args.merge_from_s3:
        if not args.bucket:
            parser.error("--merge-from-s3 requires --bucket")
        if not args.year:
            parser.error("--merge-from-s3 requires --year")
        if not args.output_csv:
            parser.error("--merge-from-s3 requires --output-csv")
        _merge_from_s3(args.year, args.bucket, args.output_csv, args.fit_tokenizer, args.val_split)
        if args.upload_merged:
            print("\nUploading merged data to S3...")
            _upload_merged(args.output_csv, args.fit_tokenizer, args.bucket, args.region)
        return

    # Local mode
    if not args.input_pgn:
        parser.error("--input-pgn is required for local mode")
    if not args.output_csv:
        parser.error("--output-csv is required for local mode")

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
