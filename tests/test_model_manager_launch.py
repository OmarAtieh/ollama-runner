"""Tests for optimized model launch: variant selection, command building, env vars."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.services.binary_manager import BinaryManager, BinaryVariant
from app.services.model_manager import ModelManager, ModelConfig


class TestBinaryVariantSelection:
    def test_selects_custom_for_q1_0_g128(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        variant = mm._select_binary_variant(str(tmp_path / "Bonsai-8B-Q1_0_g128.gguf"))
        assert variant == BinaryVariant.CUSTOM

    def test_selects_primary_for_q4_k_m(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        variant = mm._select_binary_variant(str(tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"))
        assert variant == BinaryVariant.PRIMARY

    def test_selects_custom_when_available_and_not_onebit(self, tmp_path):
        """If a custom binary is registered, prefer it for all models."""
        mm = ModelManager(app_dir=tmp_path)
        bin_dir = tmp_path / "bin"
        custom_dir = bin_dir / "custom"
        custom_dir.mkdir(parents=True)
        (custom_dir / "llama-server.exe").write_bytes(b"fake")

        bm = BinaryManager(bin_dir=bin_dir)
        with patch.object(BinaryManager, "instance", return_value=bm):
            variant = mm._select_binary_variant(str(tmp_path / "regular-Q6_K.gguf"))
            assert variant == BinaryVariant.CUSTOM


class TestBuildLaunchCommand:
    def test_includes_kv_cache_flags(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(
            id="x", name="test", path="/m.gguf",
            cache_type_k="q8_0", cache_type_v="q8_0",
        )
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=99,
            context_length=4096,
        )
        assert "--cache-type-k" in cmd
        idx = cmd.index("--cache-type-k")
        assert cmd[idx + 1] == "q8_0"
        assert "--cache-type-v" in cmd
        idx = cmd.index("--cache-type-v")
        assert cmd[idx + 1] == "q8_0"

    def test_skips_kv_cache_flags_when_f16(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(id="x", name="test", path="/m.gguf")
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=99,
            context_length=4096,
        )
        assert "--cache-type-k" not in cmd

    def test_includes_ngram_spec_flags(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(
            id="x", name="test", path="/m.gguf",
            speculative="ngram",
        )
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=99,
            context_length=4096,
        )
        assert "--spec-type" in cmd
        idx = cmd.index("--spec-type")
        assert cmd[idx + 1] == "ngram-mod"
        assert "--draft-max" in cmd

    def test_skips_spec_flags_when_none(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(id="x", name="test", path="/m.gguf")
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=99,
            context_length=4096,
        )
        assert "--spec-type" not in cmd

    def test_threads_1_when_fully_offloaded(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(id="x", name="test", path="/m.gguf")
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=99,
            context_length=4096,
        )
        idx = cmd.index("-t")
        assert cmd[idx + 1] == "1"

    def test_threads_multi_when_partial_offload(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        model = ModelConfig(id="x", name="test", path="/m.gguf")
        cmd = mm._build_launch_cmd(
            model=model,
            binary_path=Path("/bin/llama-server.exe"),
            gpu_layers=10,
            context_length=4096,
        )
        idx = cmd.index("-t")
        assert int(cmd[idx + 1]) > 1


class TestLaunchEnvironment:
    def test_cuda_env_vars_set(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        env = mm._build_launch_env()
        assert env.get("GGML_CUDA_GRAPH_OPT") == "1"
        assert env.get("CUDA_SCALE_LAUNCH_QUEUES") == "4x"

    def test_inherits_system_env(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        env = mm._build_launch_env()
        # Should contain system PATH
        assert "PATH" in env or "Path" in env
