"""
Storage backends for roshni.

Provides async storage with compression, metadata sidecars, and a
pluggable backend interface (local filesystem by default).
"""

from .base import (
    StorageBackend,
    StorageError,
    StorageKeyError,
    StorageMetadata,
    StoragePermissionError,
    StorageQuotaError,
)
from .compression import (
    CompressionType,
    compress_bytes,
    compress_json,
    decompress_bytes,
    decompress_json,
    estimate_compression_ratio,
    get_compression_for_content_type,
)
from .local import LocalStorage

__all__ = [
    "CompressionType",
    "LocalStorage",
    "StorageBackend",
    "StorageError",
    "StorageKeyError",
    "StorageMetadata",
    "StoragePermissionError",
    "StorageQuotaError",
    "compress_bytes",
    "compress_json",
    "decompress_bytes",
    "decompress_json",
    "estimate_compression_ratio",
    "get_compression_for_content_type",
]
