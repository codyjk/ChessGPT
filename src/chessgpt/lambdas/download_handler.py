"""AWS Lambda handler for downloading Lichess .zst files to S3."""

import os

import boto3

from chessgpt.data.download import download_zst


def handler(event, context):
    """
    Download a Lichess .pgn.zst and upload to S3.

    Event: {"year": int, "month": int, "bucket": str}
    """
    year = event["year"]
    month = event["month"]
    bucket = event["bucket"]

    zst_path = None
    try:
        # Download .zst to /tmp
        zst_path = download_zst(year, month, "/tmp")

        # Upload to S3 raw/ prefix
        s3 = boto3.client("s3")
        s3_key = f"raw/{zst_path.name}"
        print(f"Uploading {zst_path} to s3://{bucket}/{s3_key}")
        s3.upload_file(str(zst_path), bucket, s3_key)

        return {"status": "ok", "s3_key": s3_key}
    finally:
        # Cleanup only the file we created
        if zst_path and zst_path.exists():
            os.remove(zst_path)
