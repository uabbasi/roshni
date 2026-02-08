"""Google Drive client.

Provides upload, download, and listing for Google Drive files via
the Drive v3 API with service-account authentication.

Requires ``roshni[google]``.
"""

from __future__ import annotations

import io
import os
from typing import Any


class GoogleDriveClient:
    """Google Drive v3 API wrapper.

    Args:
        auth: A :class:`~roshni.core.auth.ServiceAccountAuth` instance.
    """

    def __init__(self, auth):
        self.auth = auth
        self._service = None

    @property
    def service(self):
        if self._service is None:
            self._service = self.auth.get_drive_service()
        return self._service

    def upload_file(
        self,
        file_path: str,
        parent_folder_id: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        """Upload a local file to Drive.

        Returns:
            File metadata dict (includes ``id``).
        """
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError:
            raise ImportError("Install with: pip install roshni[google]")

        file_metadata: dict[str, Any] = {"name": os.path.basename(file_path)}
        if parent_folder_id:
            file_metadata["parents"] = [parent_folder_id]

        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
        return self.service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    def download_file(self, file_id: str, output_path: str) -> None:
        """Download a file from Drive by ID."""
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError:
            raise ImportError("Install with: pip install roshni[google]")

        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(output_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()

    def list_files(
        self,
        folder_id: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """List files, optionally filtered by folder or query."""
        if folder_id and not query:
            query = f"'{folder_id}' in parents"

        results = (
            self.service.files()
            .list(q=query, fields="nextPageToken, files(id, name, mimeType, modifiedTime)")
            .execute()
        )
        return results.get("files", [])
