"""Tests for roshni.core.auth -- service account and OAuth.

Note: pickle usage in tests mirrors the OAuth token storage pattern
used by Google's own auth libraries (trusted local data only).
"""

from __future__ import annotations

import json
import pickle
from unittest.mock import MagicMock, patch

import pytest

from roshni.core.auth.service_account import DEFAULT_SCOPES, ServiceAccountAuth


class _FakeCredentials:
    """Picklable stand-in for Google OAuth credentials in tests."""

    valid = True
    expired = False
    refresh_token = None


# -- ServiceAccountAuth -----------------------------------------------------


class TestServiceAccountAuth:
    def test_init_expands_home(self, tmp_path):
        auth = ServiceAccountAuth(key_path=str(tmp_path / "key.json"))
        assert auth.key_path == tmp_path / "key.json"

    def test_default_scopes(self, tmp_path):
        auth = ServiceAccountAuth(key_path=str(tmp_path / "key.json"))
        assert auth.scopes == list(DEFAULT_SCOPES)

    def test_custom_scopes(self, tmp_path):
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        auth = ServiceAccountAuth(key_path=str(tmp_path / "key.json"), scopes=scopes)
        assert auth.scopes == scopes

    def test_missing_key_raises(self, tmp_path):
        auth = ServiceAccountAuth(key_path=str(tmp_path / "nonexistent.json"))
        with pytest.raises(FileNotFoundError, match="Service account key not found"):
            _ = auth.credentials

    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_credentials_lazy_load(self, mock_from_file, tmp_path):
        key_file = tmp_path / "key.json"
        key_file.write_text(json.dumps({"type": "service_account"}))

        mock_creds = MagicMock()
        mock_from_file.return_value = mock_creds

        auth = ServiceAccountAuth(key_path=str(key_file))
        assert auth._credentials is None

        result = auth.credentials
        assert result is mock_creds
        mock_from_file.assert_called_once()

        # Second access uses cache
        result2 = auth.credentials
        assert result2 is mock_creds
        assert mock_from_file.call_count == 1

    @patch("gspread.service_account")
    def test_get_gspread_client(self, mock_gspread_sa, tmp_path):
        key_file = tmp_path / "key.json"
        key_file.write_text("{}")

        mock_client = MagicMock()
        mock_gspread_sa.return_value = mock_client

        auth = ServiceAccountAuth(key_path=str(key_file))
        client = auth.get_gspread_client()
        assert client is mock_client

        # Cached on second call
        client2 = auth.get_gspread_client()
        assert client2 is mock_client
        assert mock_gspread_sa.call_count == 1

    @patch("googleapiclient.discovery.build")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_get_sheets_service(self, mock_from_file, mock_build, tmp_path):
        key_file = tmp_path / "key.json"
        key_file.write_text("{}")
        mock_from_file.return_value = MagicMock()

        auth = ServiceAccountAuth(key_path=str(key_file))
        auth.get_sheets_service()
        mock_build.assert_called_once_with("sheets", "v4", credentials=auth.credentials)

    @patch("googleapiclient.discovery.build")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_get_drive_service(self, mock_from_file, mock_build, tmp_path):
        key_file = tmp_path / "key.json"
        key_file.write_text("{}")
        mock_from_file.return_value = MagicMock()

        auth = ServiceAccountAuth(key_path=str(key_file))
        auth.get_drive_service()
        mock_build.assert_called_once_with("drive", "v3", credentials=auth.credentials)


# -- GoogleOAuth ------------------------------------------------------------


class TestGoogleOAuth:
    def test_init_creates_token_dir(self, tmp_path):
        from roshni.core.auth.google_oauth import GoogleOAuth

        token = tmp_path / "subdir" / "token.pkl"
        GoogleOAuth(
            credentials_path=tmp_path / "creds.json",
            token_path=token,
        )
        assert token.parent.exists()

    def test_default_scopes(self, tmp_path):
        from roshni.core.auth.google_oauth import SHEETS_READONLY_SCOPE, GoogleOAuth

        oauth = GoogleOAuth(
            credentials_path=tmp_path / "c.json",
            token_path=tmp_path / "t.pkl",
        )
        assert oauth.scopes == SHEETS_READONLY_SCOPE

    def test_missing_credentials_returns_none(self, tmp_path):
        from roshni.core.auth.google_oauth import GoogleOAuth

        oauth = GoogleOAuth(
            credentials_path=tmp_path / "missing.json",
            token_path=tmp_path / "token.pkl",
        )
        result = oauth.authenticate()
        assert result is None

    @patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file")
    def test_fresh_auth_flow(self, mock_from_secrets, tmp_path):
        from roshni.core.auth.google_oauth import GoogleOAuth

        creds_file = tmp_path / "creds.json"
        creds_file.write_text('{"installed":{}}')
        token_path = tmp_path / "token.pkl"

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds
        mock_from_secrets.return_value = mock_flow

        oauth = GoogleOAuth(credentials_path=creds_file, token_path=token_path)
        result = oauth.authenticate()

        assert result is mock_creds
        assert token_path.exists()

    def test_loads_cached_token(self, tmp_path):
        from roshni.core.auth.google_oauth import GoogleOAuth

        token_path = tmp_path / "token.pkl"
        with open(token_path, "wb") as f:
            pickle.dump(_FakeCredentials(), f)

        oauth = GoogleOAuth(
            credentials_path=tmp_path / "c.json",
            token_path=token_path,
        )
        result = oauth.authenticate()
        assert result is not None
