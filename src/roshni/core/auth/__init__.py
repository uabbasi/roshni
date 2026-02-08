"""Google authentication providers.

Both auth methods are behind ``roshni[google]`` — imports will fail
with a helpful message if the Google libraries aren't installed.
"""

from .google_oauth import SHEETS_READONLY_SCOPE, SHEETS_READWRITE_SCOPE, GoogleOAuth
from .service_account import DEFAULT_SCOPES, ServiceAccountAuth

__all__ = [
    "DEFAULT_SCOPES",
    "SHEETS_READONLY_SCOPE",
    "SHEETS_READWRITE_SCOPE",
    "GoogleOAuth",
    "ServiceAccountAuth",
]
