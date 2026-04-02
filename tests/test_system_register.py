"""Tests for binary registration endpoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.binary_manager import BinaryManager, BinaryVariant

client = TestClient(app)


class TestBinaryRegisterEndpoint:
    def test_register_returns_200_on_success(self, tmp_path):
        src = tmp_path / "llama-server.exe"
        src.write_bytes(b"FAKE")

        with patch.object(BinaryManager, "instance") as mock_inst:
            mock_bm = MagicMock()
            mock_bm.register_binary.return_value = True
            mock_bm.get_status.return_value = {
                "available": True, "path": "x", "status": "idle",
                "progress": 0.0, "variants": {},
            }
            mock_inst.return_value = mock_bm

            response = client.post(
                "/api/system/binary/register",
                json={"variant": "custom", "source_path": str(src)},
            )
            assert response.status_code == 200
            mock_bm.register_binary.assert_called_once()

    def test_register_returns_400_for_missing_file(self):
        response = client.post(
            "/api/system/binary/register",
            json={"variant": "custom", "source_path": "C:/nonexistent/llama-server.exe"},
        )
        assert response.status_code == 400

    def test_register_returns_400_for_invalid_variant(self):
        response = client.post(
            "/api/system/binary/register",
            json={"variant": "nope", "source_path": "C:/fake.exe"},
        )
        assert response.status_code == 400
