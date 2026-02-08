"""Google Service Account authentication.

Handles service-account-based auth for Google APIs (Sheets, Drive, etc.).
Credential path is injected — no hardcoded secret paths.

Requires ``roshni[google]``.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

# Default scopes cover Sheets + Drive (read/write)
DEFAULT_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


class ServiceAccountAuth:
    """Authenticate via a Google service-account JSON key.

    Credentials are loaded lazily on first access.  All Google service
    factories (gspread, Sheets v4, Drive v3) share the same credential
    instance.

    Args:
        key_path: Path to the service-account JSON key file.
        scopes: OAuth scopes.  Defaults to Sheets + Drive read/write.
    """

    def __init__(
        self,
        key_path: str | Path,
        scopes: list[str] | None = None,
    ):
        self.key_path = Path(key_path).expanduser()
        self.scopes = scopes or list(DEFAULT_SCOPES)
        self._credentials = None
        self._gspread_client = None

    @property
    def credentials(self):
        """Lazy-load and cache service-account credentials."""
        if self._credentials is None:
            try:
                from google.oauth2 import service_account
            except ImportError:
                raise ImportError("Install with: pip install roshni[google]")

            if not self.key_path.exists():
                raise FileNotFoundError(
                    f"Service account key not found: {self.key_path}\n"
                    "Download one from Google Cloud Console → IAM → Service Accounts."
                )

            self._credentials = service_account.Credentials.from_service_account_file(
                str(self.key_path), scopes=self.scopes
            )
            logger.debug(f"Loaded service account from {self.key_path}")

        return self._credentials

    def get_gspread_client(self):
        """Return an authenticated ``gspread.Client``."""
        if self._gspread_client is None:
            try:
                import gspread
            except ImportError:
                raise ImportError("Install with: pip install roshni[google]")

            self._gspread_client = gspread.service_account(filename=str(self.key_path), scopes=self.scopes)
            logger.debug("Created gspread client")

        return self._gspread_client

    def get_sheets_service(self):
        """Return a Google Sheets API v4 service resource."""
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError("Install with: pip install roshni[google]")

        return build("sheets", "v4", credentials=self.credentials)

    def get_drive_service(self):
        """Return a Google Drive API v3 service resource."""
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError("Install with: pip install roshni[google]")

        return build("drive", "v3", credentials=self.credentials)
