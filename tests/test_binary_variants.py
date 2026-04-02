"""Tests for multi-variant binary management."""

from pathlib import Path

from app.services.binary_manager import BinaryManager, BinaryVariant


class TestBinaryVariantPaths:
    def test_primary_variant_path(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        path = bm.binary_path_for(BinaryVariant.PRIMARY)
        assert path == tmp_path / "llama-server.exe"

    def test_custom_variant_path(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        path = bm.binary_path_for(BinaryVariant.CUSTOM)
        assert path == tmp_path / "custom" / "llama-server.exe"


class TestBinaryVariantAvailability:
    def test_primary_available_when_exists(self, tmp_path):
        (tmp_path / "llama-server.exe").write_bytes(b"fake")
        bm = BinaryManager(bin_dir=tmp_path)
        assert bm.is_variant_available(BinaryVariant.PRIMARY) is True

    def test_custom_not_available_when_missing(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        assert bm.is_variant_available(BinaryVariant.CUSTOM) is False

    def test_custom_available_when_exists(self, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "llama-server.exe").write_bytes(b"fake")
        bm = BinaryManager(bin_dir=tmp_path)
        assert bm.is_variant_available(BinaryVariant.CUSTOM) is True


class TestRegisterBinary:
    def test_register_copies_file(self, tmp_path):
        src = tmp_path / "source" / "llama-server.exe"
        src.parent.mkdir()
        src.write_bytes(b"CUSTOM_EXE")

        bm = BinaryManager(bin_dir=tmp_path)
        result = bm.register_binary(BinaryVariant.CUSTOM, src)
        assert result is True
        assert bm.is_variant_available(BinaryVariant.CUSTOM) is True
        assert src.exists()  # original not moved

    def test_register_copies_dlls_too(self, tmp_path):
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "llama-server.exe").write_bytes(b"EXE")
        (src_dir / "ggml-cuda.dll").write_bytes(b"DLL1")
        (src_dir / "llama.dll").write_bytes(b"DLL2")

        bm = BinaryManager(bin_dir=tmp_path)
        result = bm.register_binary(BinaryVariant.CUSTOM, src_dir / "llama-server.exe")
        assert result is True
        custom_dir = tmp_path / "custom"
        assert (custom_dir / "ggml-cuda.dll").exists()
        assert (custom_dir / "llama.dll").exists()

    def test_register_rejects_nonexistent(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        result = bm.register_binary(BinaryVariant.CUSTOM, tmp_path / "nope.exe")
        assert result is False


class TestGetStatusIncludesVariants:
    def test_status_has_variants_dict(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        status = bm.get_status()
        assert "variants" in status
        assert "primary" in status["variants"]
        assert "custom" in status["variants"]
        assert status["variants"]["primary"]["available"] is False
        assert status["variants"]["custom"]["available"] is False
