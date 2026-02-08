"""Google Cloud Storage client.

Simple wrapper around the ``google-cloud-storage`` library for
upload, download, and listing operations.

Requires ``roshni[storage-gcs]``.
"""

from __future__ import annotations

from typing import Any


class GoogleStorageClient:
    """Google Cloud Storage wrapper.

    Args:
        auth: A :class:`~roshni.core.auth.ServiceAccountAuth` instance.
            If ``None``, uses Application Default Credentials.
    """

    def __init__(self, auth=None):
        self.auth = auth
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from google.cloud import storage
            except ImportError:
                raise ImportError("Install with: pip install roshni[storage-gcs]")

            if self.auth is not None:
                self._client = storage.Client(credentials=self.auth.credentials)
            else:
                self._client = storage.Client()
        return self._client

    def upload_file(self, bucket_name: str, source_path: str, destination_blob: str) -> None:
        """Upload a local file to a GCS bucket."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(source_path)

    def download_file(self, bucket_name: str, source_blob: str, destination_path: str) -> None:
        """Download a blob from GCS to a local file."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        blob.download_to_filename(destination_path)

    def list_blobs(self, bucket_name: str, prefix: str | None = None) -> list[Any]:
        """List blobs in a bucket, optionally filtered by prefix."""
        bucket = self.client.bucket(bucket_name)
        return list(bucket.list_blobs(prefix=prefix))
