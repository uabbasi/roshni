"""Google authentication providers.

Both auth methods are behind ``roshni[google]`` — imports will fail
with a helpful message if the Google libraries aren't installed.
"""

from .google_oauth import (
    ALL_WORKSPACE_SCOPES,
    GMAIL_MODIFY_SCOPE,
    SHEETS_READONLY_SCOPE,
    SHEETS_READWRITE_SCOPE,
    GoogleOAuth,
    _find_client_secret,
)
from .service_account import DEFAULT_SCOPES, ServiceAccountAuth

__all__ = [
    "ALL_WORKSPACE_SCOPES",
    "DEFAULT_SCOPES",
    "GMAIL_MODIFY_SCOPE",
    "SHEETS_READONLY_SCOPE",
    "SHEETS_READWRITE_SCOPE",
    "GoogleOAuth",
    "ServiceAccountAuth",
    "_find_client_secret",
]
