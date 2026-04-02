"""Tests for the llama-server binary manager service and endpoints."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.services.binary_manager import BinaryManager
from app.main import app


client = TestClient(app)


# -- Fake GitHub release payload used across tests ---------------------------

FAKE_RELEASE_JSON = {
    "assets": [
        {
            "name": "llama-b5000-bin-win-cuda-cu12.4-x64.zip",
            "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b5000/llama-b5000-bin-win-cuda-cu12.4-x64.zip",
        },
        {
            "name": "llama-b5000-bin-win-vulkan-x64.zip",
            "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b5000/llama-b5000-bin-win-vulkan-x64.zip",
        },
        {
            "name": "llama-b5000-bin-ubuntu-x64.zip",
            "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b5000/llama-b5000-bin-ubuntu-x64.zip",
        },
    ]
}


class TestBinaryManagerInstantiation:
    def test_can_instantiate(self):
        bm = BinaryManager()
        assert bm is not None

    def test_is_available_false_when_no_binary(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        assert bm.is_available is False

    def test_is_available_true_when_binary_exists(self, tmp_path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"fake")
        bm = BinaryManager(bin_dir=tmp_path)
        assert bm.is_available is True


class TestGetStatus:
    def test_returns_expected_dict_structure(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        status = bm.get_status()
        assert isinstance(status, dict)
        assert "available" in status
        assert "path" in status
        assert "status" in status
        assert "progress" in status

    def test_status_idle_by_default(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        status = bm.get_status()
        assert status["status"] == "idle"
        assert status["available"] is False
        assert status["progress"] == 0.0


class TestGetLatestReleaseUrl:
    @pytest.mark.asyncio
    async def test_returns_cuda_url(self):
        bm = BinaryManager()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=FAKE_RELEASE_JSON)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            url = await bm.get_latest_release_url()

        assert url is not None
        assert "cuda" in url
        assert "vulkan" not in url

    @pytest.mark.asyncio
    async def test_returns_none_on_api_failure(self):
        bm = BinaryManager()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            url = await bm.get_latest_release_url()

        assert url is None


class TestDownloadAndInstall:
    @pytest.mark.asyncio
    async def test_download_flow_mocked(self, tmp_path):
        """Full download flow with mocked HTTP — no real download."""
        bm = BinaryManager(bin_dir=tmp_path)

        # Build a minimal valid zip containing a fake llama-server.exe
        import zipfile
        import io

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("build/bin/llama-server.exe", b"FAKE_EXE")
            zf.writestr("build/bin/cuda.dll", b"FAKE_DLL")
        zip_bytes = zip_buf.getvalue()

        # Mock get_latest_release_url and _get_cudart_url
        bm.get_latest_release_url = AsyncMock(
            return_value="https://example.com/fake.zip"
        )
        bm._get_cudart_url = AsyncMock(return_value=None)

        # Mock aiohttp download — deliver the zip in one chunk
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Length": str(len(zip_bytes))}

        async def fake_content_iter(chunk_size=8192):
            yield zip_bytes

        mock_response.content = MagicMock()
        mock_response.content.iter_chunked = fake_content_iter
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await bm.download_and_install()

        assert result is True
        assert bm.download_status == "ready"
        assert bm.is_available is True
        assert (tmp_path / "llama-server.exe").exists()
        assert (tmp_path / "cuda.dll").exists()

    @pytest.mark.asyncio
    async def test_download_sets_error_on_no_url(self, tmp_path):
        bm = BinaryManager(bin_dir=tmp_path)
        bm.get_latest_release_url = AsyncMock(return_value=None)
        result = await bm.download_and_install()
        assert result is False
        assert bm.download_status == "error"


class TestBinaryEndpoints:
    def test_binary_status_returns_200(self):
        response = client.get("/api/system/binary/status")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data
        assert "status" in data

    def test_binary_download_returns_200(self):
        with patch.object(
            BinaryManager, "download_and_install", new_callable=AsyncMock
        ) as mock_dl:
            mock_dl.return_value = True
            response = client.post("/api/system/binary/download")
            assert response.status_code == 200
