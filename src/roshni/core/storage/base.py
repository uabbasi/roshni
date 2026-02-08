"""
Abstract base class for storage backends.

Provides a unified async interface for local filesystem, GCS, S3, etc.
with support for compression and metadata.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StorageMetadata:
    """Metadata for stored objects."""

    key: str
    size: int
    created_at: datetime
    modified_at: datetime
    content_type: str
    compression: str | None = None
    custom_metadata: dict[str, Any] = field(default_factory=dict)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    def __init__(self, **config):
        self.config = config

    @abstractmethod
    async def save(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        compress: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> StorageMetadata:
        """Save data to storage."""

    @abstractmethod
    async def load(self, key: str) -> bytes:
        """Load data from storage. Raises StorageKeyError if not found."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an object. Returns True if deleted, False if didn't exist."""

    @abstractmethod
    async def list_keys(self, prefix: str = "", limit: int | None = None) -> AsyncIterator[str]:
        """List keys with optional prefix filter."""

    @abstractmethod
    async def get_metadata(self, key: str) -> StorageMetadata:
        """Get metadata for a stored object. Raises StorageKeyError if not found."""

    @abstractmethod
    async def copy(self, source_key: str, dest_key: str) -> StorageMetadata:
        """Copy an object within storage."""

    async def move(self, source_key: str, dest_key: str) -> StorageMetadata:
        """Move an object (copy + delete)."""
        metadata = await self.copy(source_key, dest_key)
        await self.delete(source_key)
        return metadata

    @abstractmethod
    async def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Get a URL for accessing the object."""


class StorageError(Exception):
    """Base exception for storage errors."""


class StorageKeyError(StorageError, KeyError):
    """Raised when a storage key doesn't exist."""


class StoragePermissionError(StorageError):
    """Raised when storage operation is not permitted."""


class StorageQuotaError(StorageError):
    """Raised when storage quota is exceeded."""
