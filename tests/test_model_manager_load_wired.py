"""Test that load_model uses variant selection and optimized command."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import subprocess

import pytest

from app.services.binary_manager import BinaryManager, BinaryVariant
from app.services.model_manager import ModelManager, ModelConfig


class TestLoadModelVariantWiring:
    @pytest.mark.asyncio
    async def test_load_onebit_fails_without_custom_binary(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)

        # Register a 1-bit model in the registry
        model = mm.add_model("Bonsai-8B", str(tmp_path / "Bonsai-8B-Q1_0_g128.gguf"))

        # Mock BinaryManager with no custom binary
        mock_bm = MagicMock()
        mock_bm.is_available = True
        mock_bm.is_variant_available.return_value = False
        mock_bm.binary_path_for.return_value = tmp_path / "custom" / "llama-server.exe"

        with patch.object(BinaryManager, "instance", return_value=mock_bm):
            result = await mm.load_model(model.id)

        assert result is False
        status = mm.get_status()
        assert status["error"] is not None
        assert "custom" in status["error"].lower() or "1-bit" in status["error"].lower()

    @pytest.mark.asyncio
    async def test_load_passes_env_vars_to_popen(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = mm.add_model("TestModel", str(tmp_path / "test-Q4_K_M.gguf"))

        # Create fake binary
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "llama-server.exe").write_bytes(b"fake")

        mock_bm = MagicMock()
        mock_bm.is_available = True
        mock_bm.is_variant_available.return_value = True
        mock_bm.binary_path_for.return_value = bin_dir / "llama-server.exe"

        with patch.object(BinaryManager, "instance", return_value=mock_bm), \
             patch.object(mm, "estimate_resources", return_value={"feasible": True, "vram_needed_mb": 100, "vram_available_mb": 8000, "ram_needed_mb": 100, "ram_available_mb": 30000}), \
             patch.object(mm, "recommend_gpu_layers", return_value=99), \
             patch("subprocess.Popen") as mock_popen:

            mock_proc = MagicMock()
            mock_proc.poll.return_value = 1  # process exits immediately
            mock_proc.stderr.read.return_value = b"test"
            mock_popen.return_value = mock_proc

            await mm.load_model(model.id)

            # Verify env was passed to Popen
            call_kwargs = mock_popen.call_args
            env = call_kwargs.kwargs.get("env") or (call_kwargs[1].get("env") if len(call_kwargs) > 1 else None)
            assert env is not None
            assert env.get("GGML_CUDA_GRAPH_OPT") == "1"
