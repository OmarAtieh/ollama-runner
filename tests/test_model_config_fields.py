"""Tests for new ModelConfig fields."""

from app.services.model_manager import ModelConfig


class TestModelConfigNewFields:
    def test_defaults(self):
        m = ModelConfig(id="x", name="test", path="/m.gguf")
        assert m.cache_type_k == "f16"
        assert m.cache_type_v == "f16"
        assert m.speculative == "none"

    def test_custom_values(self):
        m = ModelConfig(
            id="x", name="test", path="/m.gguf",
            cache_type_k="q8_0", cache_type_v="q8_0",
            speculative="ngram",
        )
        assert m.cache_type_k == "q8_0"
        assert m.speculative == "ngram"
