"""External service integrations.

Google integrations require optional extras:
  - ``roshni[google]``       -- Sheets, Drive, OAuth, service accounts
  - ``roshni[storage-gcs]``  -- Google Cloud Storage
"""

from .gmail import GmailSender
from .google_drive import GoogleDriveClient
from .google_sheets import CacheEntry, GoogleSheetsBase, SheetsTimeoutError
from .google_storage import GoogleStorageClient
from .notion import NotionClient
from .trello import TrelloClient

__all__ = [
    "CacheEntry",
    "GmailSender",
    "GoogleDriveClient",
    "GoogleSheetsBase",
    "GoogleStorageClient",
    "NotionClient",
    "SheetsTimeoutError",
    "TrelloClient",
]
