"""
Local filesystem storage backend.

Provides async file operations with optional gzip compression.
"""

import asyncio
import os
import shutil
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
from loguru import logger

from .base import StorageBackend, StorageKeyError, StorageMetadata, StoragePermissionError
from .compression import (
    CompressionType,
    compress_bytes,
    compress_json,
    decompress_bytes,
    decompress_json,
    get_compression_for_content_type,
)


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str = "~/.roshni-data/storage", **config):
        super().__init__(**config)
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_full_path(self, key: str) -> Path:
        """Resolve a storage key to an absolute path under ``base_path``.

        Rejects unsafe keys (absolute paths, traversal, empty keys, and
        backslash-delimited paths) to prevent writes outside ``base_path``.
        """
        raw_key = key.strip()
        if not raw_key:
            raise StoragePermissionError("Storage key cannot be empty.")
        if "\x00" in raw_key:
            raise StoragePermissionError("Storage key cannot contain null bytes.")
        if "\\" in raw_key:
            raise StoragePermissionError("Storage key cannot contain backslashes. Use '/' separators.")

        key_path = Path(raw_key)
        if key_path.is_absolute() or raw_key.startswith("~"):
            raise StoragePermissionError(f"Unsafe storage key '{key}': absolute paths are not allowed.")

        full_path = (self.base_path / key_path).resolve()
        try:
            full_path.relative_to(self.base_path)
        except ValueError as e:
            raise StoragePermissionError(f"Unsafe storage key '{key}': path traversal is not allowed.") from e
        return full_path

    def _get_metadata_path(self, key: str) -> Path:
        return self._get_full_path(key).with_suffix(".meta.json")

    async def save(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        compress: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> StorageMetadata:
        path = self._get_full_path(key)

        compression = CompressionType.NONE
        if compress:
            compression = get_compression_for_content_type(content_type)
            if compression != CompressionType.NONE:
                data = compress_bytes(data, compression)
                path = path.with_suffix(path.suffix + ".gz")

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(path, "wb") as f:
                await f.write(data)
        except PermissionError as e:
            raise StoragePermissionError(f"Cannot write to {path}: {e}")

        stat = await aiofiles.os.stat(path)

        storage_metadata = StorageMetadata(
            key=key,
            size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            content_type=content_type,
            compression=compression.value if compression != CompressionType.NONE else None,
            custom_metadata=metadata or {},
        )

        # Save metadata sidecar
        meta_path = self._get_metadata_path(key)
        meta_data = compress_json(storage_metadata.__dict__)
        async with aiofiles.open(meta_path, "wb") as f:
            await f.write(meta_data)

        return storage_metadata

    async def load(self, key: str) -> bytes:
        path = self._get_full_path(key)

        if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
            path = path.with_suffix(path.suffix + ".gz")
            compressed = True
        else:
            compressed = False

        if not path.exists():
            raise StorageKeyError(f"Key not found: {key}")

        try:
            async with aiofiles.open(path, "rb") as f:
                data = await f.read()
        except PermissionError as e:
            raise StoragePermissionError(f"Cannot read {path}: {e}")

        if compressed:
            data = decompress_bytes(data, CompressionType.GZIP)

        return data

    async def exists(self, key: str) -> bool:
        path = self._get_full_path(key)
        return path.exists() or path.with_suffix(path.suffix + ".gz").exists()

    async def delete(self, key: str) -> bool:
        path = self._get_full_path(key)
        compressed_path = path.with_suffix(path.suffix + ".gz")
        meta_path = self._get_metadata_path(key)
        deleted = False

        for p in (path, compressed_path, meta_path):
            if p.exists():
                await aiofiles.os.remove(p)
                deleted = True

        return deleted

    async def list_keys(self, prefix: str = "", limit: int | None = None) -> AsyncIterator[str]:
        base_len = len(str(self.base_path)) + 1
        count = 0

        for root, _dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".meta.json"):
                    continue
                full_path = Path(root) / file
                key = str(full_path)[base_len:]
                if key.endswith(".gz"):
                    key = key[:-3]
                if prefix and not key.startswith(prefix):
                    continue
                yield key
                count += 1
                if limit and count >= limit:
                    return

    async def get_metadata(self, key: str) -> StorageMetadata:
        meta_path = self._get_metadata_path(key)

        if meta_path.exists():
            try:
                async with aiofiles.open(meta_path, "rb") as f:
                    data = await f.read()
                meta_dict = decompress_json(data)
                meta_dict["created_at"] = datetime.fromisoformat(meta_dict["created_at"])
                meta_dict["modified_at"] = datetime.fromisoformat(meta_dict["modified_at"])
                meta_dict["key"] = key  # always use the requested key
                return StorageMetadata(**meta_dict)
            except Exception as e:
                logger.warning(f"Invalid metadata sidecar for key '{key}': {e}. Falling back to file stats.")

        # Fallback to file stats
        path = self._get_full_path(key)
        if not path.exists():
            path = path.with_suffix(path.suffix + ".gz")
            if not path.exists():
                raise StorageKeyError(f"Key not found: {key}")

        stat = await aiofiles.os.stat(path)
        return StorageMetadata(
            key=key,
            size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            content_type="application/octet-stream",
            compression="gzip" if str(path).endswith(".gz") else None,
        )

    async def copy(self, source_key: str, dest_key: str) -> StorageMetadata:
        source_path = self._get_full_path(source_key)
        if not source_path.exists():
            source_path = source_path.with_suffix(source_path.suffix + ".gz")
            if not source_path.exists():
                raise StorageKeyError(f"Source key not found: {source_key}")

        dest_path = self._get_full_path(dest_key)
        if source_path.suffix == ".gz":
            dest_path = dest_path.with_suffix(dest_path.suffix + ".gz")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.get_event_loop().run_in_executor(None, shutil.copy2, source_path, dest_path)

        # Copy metadata sidecar if it exists
        source_meta = self._get_metadata_path(source_key)
        if source_meta.exists():
            dest_meta = self._get_metadata_path(dest_key)
            await asyncio.get_event_loop().run_in_executor(None, shutil.copy2, source_meta, dest_meta)

        return await self.get_metadata(dest_key)

    async def get_url(self, key: str, expires_in: int = 3600) -> str:
        path = self._get_full_path(key)
        if not path.exists():
            path = path.with_suffix(path.suffix + ".gz")
            if not path.exists():
                raise StorageKeyError(f"Key not found: {key}")
        return f"file://{path}"
