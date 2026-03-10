"""Tests for GGUF metadata reader and model directory scanner."""

import pytest
from pathlib import Path

from app.services.model_scanner import (
    GGUFMetadata,
    read_gguf_metadata,
    scan_models_directory,
    infer_quant_from_filename,
)

# Real model files for integration tests
REAL_MODEL = Path(
    r"C:\Users\omar-\.lmstudio\models"
    r"\Jackrong\Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
    r"\Qwen3.5-4B.Q6_K.gguf"
)
MODELS_DIR = Path(r"C:\Users\omar-\.lmstudio\models")


class TestInferQuantFromFilename:
    def test_q4_k_m(self):
        assert infer_quant_from_filename("model-Q4_K_M.gguf") == "Q4_K_M"

    def test_q6_k(self):
        assert infer_quant_from_filename("Qwen3.5-4B.Q6_K.gguf") == "Q6_K"

    def test_iq2_xxs(self):
        assert infer_quant_from_filename("Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf") == "IQ2_XXS"

    def test_unknown_returns_unknown(self):
        assert infer_quant_from_filename("model.gguf") == "unknown"

    def test_case_insensitive(self):
        assert infer_quant_from_filename("model-q4_k_m.gguf") == "Q4_K_M"


class TestReadGGUFMetadata:
    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Real model file not available")
    def test_reads_real_model(self):
        meta = read_gguf_metadata(REAL_MODEL)
        assert meta is not None
        assert isinstance(meta, GGUFMetadata)

    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Real model file not available")
    def test_extracts_file_size(self):
        meta = read_gguf_metadata(REAL_MODEL)
        assert meta is not None
        assert meta.file_size_mb > 0

    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Real model file not available")
    def test_extracts_architecture(self):
        meta = read_gguf_metadata(REAL_MODEL)
        assert meta is not None
        assert meta.architecture is not None
        assert len(meta.architecture) > 0

    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Real model file not available")
    def test_extracts_quantization(self):
        meta = read_gguf_metadata(REAL_MODEL)
        assert meta is not None
        # Should be Q6_K from metadata or filename
        assert "Q6_K" in meta.quantization.upper()

    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Real model file not available")
    def test_extracts_num_layers(self):
        meta = read_gguf_metadata(REAL_MODEL)
        assert meta is not None
        assert meta.num_layers is not None
        assert meta.num_layers > 0

    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Real model file not available")
    def test_extracts_context_length(self):
        meta = read_gguf_metadata(REAL_MODEL)
        assert meta is not None
        assert meta.context_length is not None
        assert meta.context_length > 0

    def test_nonexistent_file_returns_none(self):
        meta = read_gguf_metadata(Path("C:/nonexistent/model.gguf"))
        assert meta is None


class TestScanModelsDirectory:
    @pytest.mark.skipif(not MODELS_DIR.exists(), reason="Models directory not available")
    def test_finds_models(self):
        models = scan_models_directory(MODELS_DIR)
        assert len(models) > 0

    @pytest.mark.skipif(not MODELS_DIR.exists(), reason="Models directory not available")
    def test_returns_gguf_metadata_list(self):
        models = scan_models_directory(MODELS_DIR)
        for m in models:
            assert isinstance(m, GGUFMetadata)

    @pytest.mark.skipif(not MODELS_DIR.exists(), reason="Models directory not available")
    def test_skips_mmproj_files(self):
        models = scan_models_directory(MODELS_DIR)
        for m in models:
            assert "mmproj" not in m.file_path.lower()

    @pytest.mark.skipif(not MODELS_DIR.exists(), reason="Models directory not available")
    def test_all_have_file_path(self):
        models = scan_models_directory(MODELS_DIR)
        for m in models:
            assert m.file_path
            assert m.file_path.endswith(".gguf")

    def test_nonexistent_directory_returns_empty(self):
        models = scan_models_directory(Path("C:/nonexistent_dir"))
        assert models == []
