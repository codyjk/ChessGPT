"""AWS Lambda handler for preparing training data from .zst files."""

import os
from pathlib import Path

import boto3

from chessgpt.data.download import zst_filename
from chessgpt.data.prepare import prepare_training_data, print_stats


def handler(event, context):
    """
    Download .zst from S3, run prepare pipeline, upload CSV to S3.

    Event: {"year": int, "month": int, "bucket": str, "min_elo": int (optional)}
    """
    year = event["year"]
    month = event["month"]
    bucket = event["bucket"]
    min_elo = event.get("min_elo", 1800)

    s3 = boto3.client("s3")

    # Download .zst from S3 raw/ to /tmp
    zst_name = zst_filename(year, month)
    s3_key = f"raw/{zst_name}"
    local_zst = f"/tmp/{zst_name}"
    csv_name = f"lichess_{year:04d}-{month:02d}.csv"
    local_csv = f"/tmp/{csv_name}"

    try:
        print(f"Downloading s3://{bucket}/{s3_key} to {local_zst}")
        s3.download_file(bucket, s3_key, local_zst)

        # Run prepare pipeline -- .zst streaming means no decompression needed
        counts = prepare_training_data(
            input_pgn=local_zst,
            output_csv=local_csv,
            min_elo=min_elo,
        )
        print_stats(counts)

        # Upload CSV to S3 prepared/ prefix
        s3_csv_key = f"prepared/{csv_name}"
        print(f"Uploading {local_csv} to s3://{bucket}/{s3_csv_key}")
        s3.upload_file(local_csv, bucket, s3_csv_key)

        return {"status": "ok", "s3_key": s3_csv_key, "counts": counts}
    finally:
        # Cleanup only the files we created
        for path in [local_zst, local_csv]:
            if Path(path).exists():
                os.remove(path)
