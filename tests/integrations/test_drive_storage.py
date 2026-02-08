"""Tests for roshni.integrations.google_drive and google_storage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from roshni.integrations.google_drive import GoogleDriveClient
from roshni.integrations.google_storage import GoogleStorageClient

# -- GoogleDriveClient ------------------------------------------------------


class TestGoogleDriveClient:
    def test_lazy_service_init(self):
        mock_auth = MagicMock()
        client = GoogleDriveClient(auth=mock_auth)
        assert client._service is None

        _ = client.service
        mock_auth.get_drive_service.assert_called_once()

    def test_list_files_with_folder_id(self):
        mock_auth = MagicMock()
        mock_service = MagicMock()
        mock_auth.get_drive_service.return_value = mock_service
        mock_service.files().list().execute.return_value = {"files": [{"id": "1", "name": "test.txt"}]}

        client = GoogleDriveClient(auth=mock_auth)
        files = client.list_files(folder_id="folder123")
        assert len(files) == 1
        assert files[0]["name"] == "test.txt"

    @patch("googleapiclient.http.MediaFileUpload")
    def test_upload_file(self, mock_upload_cls, tmp_path):
        mock_auth = MagicMock()
        mock_service = MagicMock()
        mock_auth.get_drive_service.return_value = mock_service
        mock_service.files().create().execute.return_value = {"id": "new123"}

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        client = GoogleDriveClient(auth=mock_auth)
        result = client.upload_file(str(test_file))
        assert result["id"] == "new123"


# -- GoogleStorageClient ----------------------------------------------------


class TestGoogleStorageClient:
    @patch("google.cloud.storage.Client")
    def test_init_with_auth(self, mock_client_cls):
        mock_auth = MagicMock()
        client = GoogleStorageClient(auth=mock_auth)
        _ = client.client
        mock_client_cls.assert_called_once_with(credentials=mock_auth.credentials)

    @patch("google.cloud.storage.Client")
    def test_init_without_auth(self, mock_client_cls):
        client = GoogleStorageClient()
        _ = client.client
        mock_client_cls.assert_called_once_with()

    @patch("google.cloud.storage.Client")
    def test_upload_file(self, mock_client_cls):
        mock_storage_client = MagicMock()
        mock_client_cls.return_value = mock_storage_client
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        client = GoogleStorageClient(auth=MagicMock())
        client._client = mock_storage_client  # bypass lazy init
        client.upload_file("my-bucket", "/tmp/test.txt", "dest.txt")
        mock_blob.upload_from_filename.assert_called_once_with("/tmp/test.txt")

    @patch("google.cloud.storage.Client")
    def test_list_blobs(self, mock_client_cls):
        mock_storage_client = MagicMock()
        mock_client_cls.return_value = mock_storage_client
        mock_bucket = MagicMock()
        mock_storage_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = ["blob1", "blob2"]

        client = GoogleStorageClient(auth=MagicMock())
        client._client = mock_storage_client
        result = client.list_blobs("my-bucket", prefix="data/")
        assert len(result) == 2
