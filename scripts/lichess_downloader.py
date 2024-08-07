import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import boto3
import requests
from tqdm import tqdm

MAX_RETRIES = 3
MAX_CONCURRENT_DOWNLOADS = 5
COMPLETED_FILE = "completed_downloads.txt"
S3_BUCKET_NAME = "lichess-database-bucket"

s3_client = boto3.client("s3")


def stream_to_s3(url, progress_bar):
    filename = os.path.basename(urlparse(url).path)
    for attempt in range(MAX_RETRIES):
        try:
            # Start the download stream
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))

                # Prepare the multipart upload
                mpu = s3_client.create_multipart_upload(
                    Bucket=S3_BUCKET_NAME, Key=filename
                )

                parts = []
                part_number = 1
                bytes_transferred = 0

                # Use tqdm for progress tracking
                with tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=filename,
                    leave=False,
                ) as pbar:
                    for chunk in r.iter_content(
                        chunk_size=5 * 1024 * 1024
                    ):  # 5MB chunks
                        part = s3_client.upload_part(
                            Bucket=S3_BUCKET_NAME,
                            Key=filename,
                            PartNumber=part_number,
                            UploadId=mpu["UploadId"],
                            Body=chunk,
                        )
                        parts.append({"PartNumber": part_number, "ETag": part["ETag"]})
                        part_number += 1
                        bytes_transferred += len(chunk)
                        pbar.update(len(chunk))

                # Complete the multipart upload
                s3_client.complete_multipart_upload(
                    Bucket=S3_BUCKET_NAME,
                    Key=filename,
                    UploadId=mpu["UploadId"],
                    MultipartUpload={"Parts": parts},
                )

            progress_bar.update(1)
            return url
        except (
            requests.exceptions.RequestException,
            boto3.exceptions.S3UploadFailedError,
        ) as e:
            print(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            time.sleep(5)  # Wait before retrying
            if attempt == MAX_RETRIES - 1:
                # If this was the last attempt, try to abort the multipart upload
                try:
                    s3_client.abort_multipart_upload(
                        Bucket=S3_BUCKET_NAME, Key=filename, UploadId=mpu["UploadId"]
                    )
                except Exception as abort_error:
                    print(f"Failed to abort multipart upload: {str(abort_error)}")

    print(f"Failed to process after {MAX_RETRIES} attempts: {url}")
    return None


def load_completed_downloads():
    if os.path.exists(COMPLETED_FILE):
        with open(COMPLETED_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()


def save_completed_download(url):
    with open(COMPLETED_FILE, "a") as f:
        f.write(f"{url}\n")


def main():
    completed_downloads = load_completed_downloads()

    with open("lichess-files.txt", "r") as f:
        urls = [line.strip() for line in f if line.strip() not in completed_downloads]

    total_files = len(urls)

    with tqdm(
        total=total_files, desc="Overall Progress", unit="file"
    ) as overall_progress:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
            future_to_url = {
                executor.submit(stream_to_s3, url, overall_progress): url
                for url in urls
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        save_completed_download(url)
                except Exception as e:
                    print(f"Exception occurred while processing {url}: {str(e)}")


if __name__ == "__main__":
    main()
