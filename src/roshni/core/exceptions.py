"""
Roshni exception hierarchy.

All roshni exceptions inherit from RoshniError, making it easy for consumers
to catch library-level errors while still distinguishing specific failure modes.
"""


class RoshniError(Exception):
    """Base exception class for all roshni errors."""


class ConfigurationError(RoshniError):
    """Raised for configuration errors (missing keys, invalid values)."""


class APIError(RoshniError):
    """Raised for API communication errors."""


class GoogleAPIError(APIError):
    """Raised for Google API errors."""


class DataProcessingError(RoshniError):
    """Raised for data processing errors."""


class FileIOError(RoshniError):
    """Raised for file I/O errors."""


class CacheError(RoshniError):
    """Raised for caching errors."""


class LLMError(APIError):
    """Raised for LLM API errors."""


class AuthenticationError(RoshniError):
    """Raised for authentication errors."""


class SecretNotFoundError(RoshniError):
    """Raised when a required secret cannot be found in any provider."""
