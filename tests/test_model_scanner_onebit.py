"""Tests for 1-bit quantization detection in model scanner."""

from app.services.model_scanner import (
    infer_quant_from_filename,
    is_onebit_quant,
    infer_capabilities,
)


class TestOnebitQuantDetection:
    def test_infer_q1_0_g128_from_filename(self):
        assert infer_quant_from_filename("Bonsai-8B-Q1_0_g128.gguf") == "Q1_0_g128"

    def test_infer_q1_0_from_filename(self):
        assert infer_quant_from_filename("model-Q1_0.gguf") == "Q1_0"

    def test_is_onebit_q1_0_g128(self):
        assert is_onebit_quant("Q1_0_g128") is True

    def test_is_onebit_q1_0(self):
        assert is_onebit_quant("Q1_0") is True

    def test_is_not_onebit_q4_k_m(self):
        assert is_onebit_quant("Q4_K_M") is False

    def test_is_not_onebit_unknown(self):
        assert is_onebit_quant("unknown") is False

    def test_is_onebit_case_insensitive(self):
        assert is_onebit_quant("q1_0_g128") is True


class TestOnebitCapabilityInference:
    def test_bonsai_detected_as_onebit(self):
        caps = infer_capabilities("Bonsai-8B-Q1_0_g128.gguf")
        assert "1bit" in caps

    def test_q1_0_in_name_detected(self):
        caps = infer_capabilities("model-Q1_0.gguf")
        assert "1bit" in caps

    def test_regular_model_not_onebit(self):
        caps = infer_capabilities("Qwen3.5-4B-Q4_K_M.gguf")
        assert "1bit" not in caps
