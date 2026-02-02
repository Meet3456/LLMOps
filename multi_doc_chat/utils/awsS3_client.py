import os
from pathlib import Path

import boto3

from multi_doc_chat.logger import GLOBAL_LOGGER as log


class S3Client:
    def __init__(self):
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION", "eu_north_1")

        try:
            self.s3 = boto3.client("s3", region_name=self.region)
            log.info(
                f"S3 Client initialized | bucket={self.bucket_name} | region={self.region}"
            )
        except Exception as e:
            log.error(f"Failed to initialize S3 Client: {e}")
            raise

    def upload_directory(self, local_path: Path, s3_prefix: str):
        """Recursively upload a directory to S3."""
        if not local_path.exists():
            log.warning(f"Local path does not exist, skipping upload: {local_path}")
            return

        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = Path(root) / file
                print(f"Local file path : {local_file_path}")

                # Calculate relative path to maintain structure
                relative_path = local_file_path.relative_to(local_path)
                print(f"Relative file path : {relative_path}")
                s3_key = f"{s3_prefix}/{relative_path}"

                try:
                    self.s3.upload_file(
                        str(local_file_path), self.bucket_name, str(s3_key)
                    )
                    log.debug(f"Uploaded to S3: {s3_key}")
                except Exception as e:
                    log.error(f"Failed to upload {local_file_path} to S3: {e}")

    def download_directory(self, s3_prefix: str, local_path: Path):
        """Download a directory from S3 to local path."""
        try:
            # List objects to get all files in the 'folder'
            paginator = self.s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)

            downloaded_count = 0

            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    s3_key = obj["Key"]

                    # Handle varying prefix structures carefully
                    relative_path = s3_key.replace(s3_prefix, "").lstrip("/")

                    # Construct local path
                    dest_path = local_path / relative_path
                    print(
                        f"The path to be made the directory for download : {dest_path}"
                    )
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    self.s3.download_file(self.bucket_name, s3_key, str(dest_path))
                    downloaded_count += 1

            if downloaded_count > 0:
                log.info(
                    f"Downloaded directory from S3 | prefix={s3_prefix} | local={local_path} | files={downloaded_count}"
                )
            else:
                log.info(f"No files found in S3 for prefix: {s3_prefix}")
        except Exception as e:
            log.error(f"Failed to download directory from S3: {e}")
