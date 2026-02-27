"""CLI: Download Lichess PGN database."""

import argparse
import json

from chessgpt.data.download import download_pgn, download_zst


def _invoke_lambda(year: int, month: int, bucket: str, function_name: str) -> None:
    """Invoke the download Lambda function asynchronously."""
    import boto3

    client = boto3.client("lambda")
    payload = {"year": year, "month": month, "bucket": bucket}
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


def main():
    parser = argparse.ArgumentParser(description="Download a Lichess monthly PGN database")
    parser.add_argument("--year", type=int, required=True, help="Year (e.g. 2013)")
    parser.add_argument("--month", type=int, required=True, help="Month (1-12)")
    parser.add_argument("--output-dir", type=str, default="data/", help="Output directory")
    parser.add_argument(
        "--zst-only", action="store_true", help="Download .zst only, skip decompression"
    )
    parser.add_argument(
        "--cloud", action="store_true", help="Invoke AWS Lambda instead of running locally"
    )
    parser.add_argument("--bucket", type=str, default=None, help="S3 bucket for --cloud mode")
    parser.add_argument(
        "--function-name",
        type=str,
        default="chessgpt-download",
        help="Lambda function name (default: chessgpt-download)",
    )
    args = parser.parse_args()

    if args.cloud:
        if not args.bucket:
            parser.error("--cloud requires --bucket")
        _invoke_lambda(args.year, args.month, args.bucket, args.function_name)
        return

    if args.zst_only:
        path = download_zst(args.year, args.month, args.output_dir)
        print(f"ZST available at: {path}")
    else:
        path = download_pgn(args.year, args.month, args.output_dir)
        print(f"PGN available at: {path}")
