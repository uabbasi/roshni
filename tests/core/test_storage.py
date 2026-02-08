"""Tests for core.storage — LocalStorage + compression utilities."""

import pytest

from roshni.core.storage import (
    CompressionType,
    LocalStorage,
    StorageKeyError,
    compress_bytes,
    compress_json,
    decompress_bytes,
    decompress_json,
    estimate_compression_ratio,
    get_compression_for_content_type,
)

# ── Compression utilities ───────────────────────────────────────────


class TestCompression:
    def test_gzip_roundtrip(self):
        data = b"hello world" * 100
        compressed = compress_bytes(data, CompressionType.GZIP)
        assert compressed != data
        assert decompress_bytes(compressed, CompressionType.GZIP) == data

    def test_none_passthrough(self):
        data = b"untouched"
        assert compress_bytes(data, CompressionType.NONE) is data
        assert decompress_bytes(data, CompressionType.NONE) is data

    def test_json_roundtrip(self):
        obj = {"key": "value", "count": 42, "nested": [1, 2, 3]}
        compressed = compress_json(obj)
        assert isinstance(compressed, bytes)
        result = decompress_json(compressed)
        assert result == obj

    def test_compression_ratio(self):
        assert estimate_compression_ratio(1000, 300) == pytest.approx(70.0)
        assert estimate_compression_ratio(0, 0) == 0.0

    def test_content_type_routing(self):
        assert get_compression_for_content_type("application/json") == CompressionType.GZIP
        assert get_compression_for_content_type("text/plain") == CompressionType.GZIP
        assert get_compression_for_content_type("image/png") == CompressionType.NONE
        assert get_compression_for_content_type("video/mp4") == CompressionType.NONE


# ── LocalStorage ────────────────────────────────────────────────────


class TestLocalStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        return LocalStorage(base_path=str(tmp_path / "store"))

    @pytest.mark.asyncio
    async def test_save_and_load(self, storage):
        data = b"test data content"
        meta = await storage.save("docs/readme.txt", data, content_type="text/plain")
        assert meta.key == "docs/readme.txt"
        assert meta.content_type == "text/plain"

        loaded = await storage.load("docs/readme.txt")
        assert loaded == data

    @pytest.mark.asyncio
    async def test_save_compressed(self, storage):
        data = b"repeated " * 500
        meta = await storage.save("big.txt", data, content_type="text/plain", compress=True)
        assert meta.compression == "gzip"
        assert meta.size < len(data)

        loaded = await storage.load("big.txt")
        assert loaded == data

    @pytest.mark.asyncio
    async def test_save_no_compress(self, storage):
        data = b"small"
        meta = await storage.save("raw.bin", data, compress=False)
        assert meta.compression is None

    @pytest.mark.asyncio
    async def test_exists(self, storage):
        assert not await storage.exists("missing")
        await storage.save("present.txt", b"hi", content_type="text/plain")
        assert await storage.exists("present.txt")

    @pytest.mark.asyncio
    async def test_delete(self, storage):
        await storage.save("to_delete.txt", b"bye", content_type="text/plain")
        assert await storage.delete("to_delete.txt")
        assert not await storage.exists("to_delete.txt")
        assert not await storage.delete("to_delete.txt")  # already gone

    @pytest.mark.asyncio
    async def test_list_keys(self, storage):
        await storage.save("a/1.txt", b"a1", content_type="text/plain")
        await storage.save("a/2.txt", b"a2", content_type="text/plain")
        await storage.save("b/1.txt", b"b1", content_type="text/plain")

        all_keys = [k async for k in storage.list_keys()]
        assert len(all_keys) == 3

        a_keys = [k async for k in storage.list_keys(prefix="a/")]
        assert len(a_keys) == 2

    @pytest.mark.asyncio
    async def test_list_keys_limit(self, storage):
        for i in range(5):
            await storage.save(f"item_{i}.txt", b"x", content_type="text/plain")
        keys = [k async for k in storage.list_keys(limit=2)]
        assert len(keys) == 2

    @pytest.mark.asyncio
    async def test_load_missing_raises(self, storage):
        with pytest.raises(StorageKeyError, match="not found"):
            await storage.load("no_such_key")

    @pytest.mark.asyncio
    async def test_get_metadata(self, storage):
        await storage.save("meta_test.json", b'{"a":1}', content_type="application/json")
        meta = await storage.get_metadata("meta_test.json")
        assert meta.key == "meta_test.json"
        assert meta.size > 0

    @pytest.mark.asyncio
    async def test_copy(self, storage):
        await storage.save("src.txt", b"original", content_type="text/plain")
        meta = await storage.copy("src.txt", "dst.txt")
        assert meta.key == "dst.txt"
        assert await storage.load("dst.txt") == b"original"
        assert await storage.exists("src.txt")  # source still exists

    @pytest.mark.asyncio
    async def test_move(self, storage):
        await storage.save("old.txt", b"moving", content_type="text/plain")
        await storage.move("old.txt", "new.txt")
        assert await storage.load("new.txt") == b"moving"
        assert not await storage.exists("old.txt")

    @pytest.mark.asyncio
    async def test_get_url(self, storage):
        await storage.save("url_test.txt", b"hi", content_type="text/plain")
        url = await storage.get_url("url_test.txt")
        assert url.startswith("file://")
