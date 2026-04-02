"""Integration: 1-bit detection → variant selection → optimized command."""

from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.binary_manager import BinaryManager, BinaryVariant
from app.services.model_manager import ModelManager, ModelConfig
from app.services.model_scanner import (
    infer_quant_from_filename,
    is_onebit_quant,
    infer_capabilities,
)


class TestOnebitFullFlow:
    def test_bonsai_detection_to_variant(self, tmp_path):
        filename = "Bonsai-8B-Q1_0_g128.gguf"

        # Quant detected
        quant = infer_quant_from_filename(filename)
        assert quant == "Q1_0_g128"
        assert is_onebit_quant(quant) is True

        # Capability includes 1bit
        caps = infer_capabilities(filename)
        assert "1bit" in caps

        # Variant selection — 1-bit always goes to CUSTOM regardless of binary availability
        mm = ModelManager(app_dir=tmp_path)
        variant = mm._select_binary_variant(str(tmp_path / filename))
        assert variant == BinaryVariant.CUSTOM

        # Binary path resolves to custom dir
        bm = BinaryManager(bin_dir=tmp_path)
        path = bm.binary_path_for(variant)
        assert "custom" in str(path)

    def test_regular_model_uses_primary(self, tmp_path):
        filename = "Qwen3.5-9B-Q6_K.gguf"

        quant = infer_quant_from_filename(filename)
        assert quant == "Q6_K"
        assert is_onebit_quant(quant) is False

        # Use a fresh BinaryManager with no custom binary present
        bm = BinaryManager(bin_dir=tmp_path)
        mm = ModelManager(app_dir=tmp_path)
        with patch.object(BinaryManager, "instance", return_value=bm):
            variant = mm._select_binary_variant(str(tmp_path / filename))
        assert variant == BinaryVariant.PRIMARY


class TestOptimizedCommandFlow:
    def test_q8_kv_cache_in_command(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(
            id="x", name="test", path="/m.gguf",
            cache_type_k="q8_0", cache_type_v="q8_0",
            speculative="ngram",
        )
        cmd = mm._build_launch_cmd(model, Path("/bin/llama-server.exe"), 99, 8192)

        # KV cache
        assert "--cache-type-k" in cmd
        assert "q8_0" in cmd

        # Speculative
        assert "--spec-type" in cmd
        assert "ngram-mod" in cmd

        # Threads = 1 (fully offloaded)
        t_idx = cmd.index("-t")
        assert cmd[t_idx + 1] == "1"

    def test_env_vars_present(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        env = mm._build_launch_env()
        assert env["GGML_CUDA_GRAPH_OPT"] == "1"
        assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "4x"
