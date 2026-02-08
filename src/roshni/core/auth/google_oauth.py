"""Google OAuth 2.0 authentication.

Handles the interactive OAuth flow for user-facing Google API access.
Token refresh and persistence are automatic.

Note: pickle is used for token storage — this is the standard Google
auth library pattern for OAuth refresh tokens (trusted local data only).

Requires ``roshni[google]``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from loguru import logger

# Common scope presets
SHEETS_READONLY_SCOPE = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
SHEETS_READWRITE_SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]


class GoogleOAuth:
    """Interactive OAuth 2.0 flow with token persistence.

    Tokens are stored as pickle files and refreshed automatically.
    If no valid token exists, a browser window opens for user consent.

    Args:
        credentials_path: Path to the OAuth client-secrets JSON
            (downloaded from Google Cloud Console).
        token_path: Path to store/load the cached refresh token.
        scopes: OAuth scopes.  Defaults to Sheets read-only.
    """

    def __init__(
        self,
        credentials_path: str | Path,
        token_path: str | Path,
        scopes: list[str] | None = None,
    ):
        self.credentials_path = Path(credentials_path).expanduser()
        self.token_path = Path(token_path).expanduser()
        self.scopes = scopes or list(SHEETS_READONLY_SCOPE)

        # Ensure token directory exists
        self.token_path.parent.mkdir(parents=True, exist_ok=True)

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
            try:
                with open(self.token_path, "rb") as f:
                    creds = pickle.load(f)
                logger.debug(f"Loaded credentials from {self.token_path}")
            except Exception as e:
                logger.warning(f"Failed to load token: {e}")

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
                try:
                    with open(self.token_path, "wb") as f:
                        pickle.dump(creds, f)
                    logger.debug(f"Token saved to {self.token_path}")
                except Exception as e:
                    logger.warning(f"Failed to save token: {e}")

        return creds

    def get_sheets_service(self):
        """Return a Google Sheets API v4 service, or ``None`` on failure."""
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError("Install with: pip install roshni[google]")

        creds = self.authenticate()
        if not creds:
            return None

        return build("sheets", "v4", credentials=creds)
