"""
Compression utilities for storage backends.

Core compression (gzip) requires no extra dependencies.
Numpy compression is lazy-loaded behind roshni[health] or similar.
"""

import gzip
import json
from enum import Enum
from io import BytesIO
from typing import Any


class CompressionType(Enum):
    """Supported compression types."""

    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    NPZ = "npz"


def compress_bytes(data: bytes, compression: CompressionType = CompressionType.GZIP) -> bytes:
    """Compress binary data."""
    if compression == CompressionType.NONE:
        return data
    if compression == CompressionType.GZIP:
        buffer = BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6) as gz:
            gz.write(data)
        return buffer.getvalue()
    raise ValueError(f"Unsupported compression type: {compression}")


def decompress_bytes(data: bytes, compression: CompressionType = CompressionType.GZIP) -> bytes:
    """Decompress binary data."""
    if compression == CompressionType.NONE:
        return data
    if compression == CompressionType.GZIP:
        return gzip.decompress(data)
    raise ValueError(f"Unsupported compression type: {compression}")


def compress_json(obj: Any, indent: int | None = None) -> bytes:
    """JSON-serialize and gzip-compress an object."""
    json_str = json.dumps(obj, indent=indent, default=str)
    return compress_bytes(json_str.encode("utf-8"))


def decompress_json(data: bytes) -> Any:
    """Decompress and parse JSON data."""
    json_str = decompress_bytes(data).decode("utf-8")
    return json.loads(json_str)


def compress_numpy(array) -> bytes:
    """Compress a numpy array. Requires numpy to be installed."""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Install numpy for array compression: pip install numpy")

    buffer = BytesIO()
    np.savez_compressed(buffer, array=array)
    return buffer.getvalue()


def decompress_numpy(data: bytes):
    """Decompress a numpy array. Requires numpy to be installed."""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Install numpy for array decompression: pip install numpy")

    buffer = BytesIO(data)
    npz = np.load(buffer)
    return npz["array"]


def estimate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Calculate compression ratio as a percentage (0-100)."""
    if original_size == 0:
        return 0.0
    return (1 - compressed_size / original_size) * 100


def get_compression_for_content_type(content_type: str) -> CompressionType:
    """Determine best compression type based on MIME type."""
    # Already compressed formats
    if any(ct in content_type for ct in ["image/", "video/", "audio/", "zip", "gzip"]):
        return CompressionType.NONE
    # Text-based formats benefit from compression
    if any(ct in content_type for ct in ["text/", "json", "xml", "html"]):
        return CompressionType.GZIP
    # Default to compression
    return CompressionType.GZIP
