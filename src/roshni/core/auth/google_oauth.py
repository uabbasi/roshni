"""Google OAuth 2.0 authentication.

Handles the interactive OAuth flow for user-facing Google API access.
Token refresh and persistence are automatic.

Supports two token formats:
  - ``"pickle"`` (default) — standard Google auth library pattern
  - ``"json"`` — portable JSON token files

Requires ``roshni[google]``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from loguru import logger

# Common scope presets
SHEETS_READONLY_SCOPE = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
SHEETS_READWRITE_SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]

GMAIL_MODIFY_SCOPE = ["https://www.googleapis.com/auth/gmail.modify"]

ALL_WORKSPACE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
]


def _find_client_secret(search_dir: str | Path = "~/.secrets") -> Path | None:
    """Search for a ``client_secret*.json`` file in *search_dir*."""
    d = Path(search_dir).expanduser()
    if not d.is_dir():
        return None
    candidates = sorted(d.glob("client_secret*.json"))
    return candidates[0] if candidates else None


class GoogleOAuth:
    """Interactive OAuth 2.0 flow with token persistence.

    Tokens are stored as pickle or JSON files and refreshed automatically.
    If no valid token exists, a browser window opens for user consent.

    Args:
        credentials_path: Path to the OAuth client-secrets JSON
            (downloaded from Google Cloud Console).
        token_path: Path to store/load the cached refresh token.
        scopes: OAuth scopes.  Defaults to Sheets read-only.
        token_format: ``"pickle"`` (default) or ``"json"``.
    """

    def __init__(
        self,
        credentials_path: str | Path,
        token_path: str | Path,
        scopes: list[str] | None = None,
        token_format: str = "pickle",
    ):
        self.credentials_path = Path(credentials_path).expanduser()
        self.token_path = Path(token_path).expanduser()
        self.scopes = scopes or list(SHEETS_READONLY_SCOPE)
        self.token_format = token_format

        # Ensure token directory exists
        self.token_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_token_pickle(self):
        """Load credentials from a pickle file."""
        try:
            with open(self.token_path, "rb") as f:
                creds = pickle.load(f)
            logger.debug(f"Loaded credentials from {self.token_path}")
            return creds
        except Exception as e:
            logger.warning(f"Failed to load pickle token: {e}")
            return None

    def _save_token_pickle(self, creds) -> None:
        """Save credentials to a pickle file."""
        try:
            with open(self.token_path, "wb") as f:
                pickle.dump(creds, f)
            logger.debug(f"Token saved to {self.token_path}")
        except Exception as e:
            logger.warning(f"Failed to save pickle token: {e}")

    def _load_token_json(self):
        """Load credentials from a JSON file."""
        try:
            from google.oauth2.credentials import Credentials

            creds = Credentials.from_authorized_user_file(str(self.token_path), self.scopes)
            logger.debug(f"Loaded credentials from {self.token_path}")
            return creds
        except Exception as e:
            logger.warning(f"Failed to load JSON token: {e}")
            return None

    def _save_token_json(self, creds) -> None:
        """Save credentials to a JSON file."""
        try:
            self.token_path.write_text(creds.to_json())
            logger.debug(f"Token saved to {self.token_path}")
        except Exception as e:
            logger.warning(f"Failed to save JSON token: {e}")

    def authenticate(self):
        """Authenticate and return credentials.

        Loads cached token if available, refreshes if expired,
        or starts an interactive OAuth flow.

        Returns:
            ``google.oauth2.credentials.Credentials`` or ``None``.
        """
        try:
            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            raise ImportError("Install with: pip install roshni[google]")

        creds = None

        # Try loading existing token
        if self.token_path.exists():
            if self.token_format == "json":
                creds = self._load_token_json()
            else:
                creds = self._load_token_pickle()

        # Refresh or run interactive flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired credentials")
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Token refresh failed: {e}")
                    creds = None

            if not creds:
                if not self.credentials_path.exists():
                    logger.error(
                        f"OAuth credentials not found: {self.credentials_path}\n"
                        "Download from Google Cloud Console -> APIs & Services -> Credentials."
                    )
                    return None

                try:
                    flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), self.scopes)
                    logger.info("Starting OAuth flow -- complete authentication in browser")
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logger.error(f"OAuth flow failed: {e}")
                    return None

            # Persist token
            if creds:
                if self.token_format == "json":
                    self._save_token_json(creds)
                else:
                    self._save_token_pickle(creds)

        return creds

    def get_service(self, name: str, version: str):
        """Return an authenticated Google API service, or ``None`` on failure.

        Args:
            name: API name (e.g. ``"gmail"``, ``"sheets"``, ``"calendar"``).
            version: API version (e.g. ``"v1"``, ``"v4"``).
        """
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError("Install with: pip install roshni[google]")

        creds = self.authenticate()
        if not creds:
            return None

        return build(name, version, credentials=creds)

    def get_sheets_service(self):
        """Return a Google Sheets API v4 service, or ``None`` on failure."""
        return self.get_service("sheets", "v4")
