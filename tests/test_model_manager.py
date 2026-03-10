"""Tests for model manager: registry, resource estimation, load/unload, and API endpoints."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient

from app.services.model_manager import ModelConfig, ModelManager
from app.main import app

client = TestClient(app)

REAL_MODEL = Path(
    r"C:\Users\omar-\.lmstudio\models"
    r"\Jackrong\Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
    r"\Qwen3.5-4B.Q6_K.gguf"
)


class TestModelManagerInstantiation:
    def test_can_instantiate(self, tmp_path):
        mm = ModelManager(app_dir=tmp_path)
        assert mm is not None

    def test_singleton(self):
        a = ModelManager.instance()
        b = ModelManager.instance()
        assert a is b


class TestRegistry:
    def test_load_empty_registry(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        models = mm.load_registry()
        assert models == []

    def test_add_model(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        model = mm.add_model("Test Model", "C:/fake/model.gguf")
        assert isinstance(model, ModelConfig)
        assert model.name == "Test Model"
        assert model.path == "C:/fake/model.gguf"

    def test_add_model_generates_id(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        model = mm.add_model("Test Model", "C:/fake/model.gguf")
        assert model.id is not None
        assert len(model.id) > 0

    def test_add_model_persists(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        mm.add_model("Test Model", "C:/fake/model.gguf")
        models = mm.load_registry()
        assert len(models) == 1

    def test_add_model_upserts(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        m1 = mm.add_model("Model A", "C:/fake/model.gguf")
        m2 = mm.add_model("Model B", "C:/fake/model.gguf")
        # Same path -> same ID -> upsert
        assert m1.id == m2.id
        models = mm.load_registry()
        assert len(models) == 1
        assert models[0].name == "Model B"

    def test_remove_model(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        model = mm.add_model("Test Model", "C:/fake/model.gguf")
        mm.remove_model(model.id)
        models = mm.load_registry()
        assert len(models) == 0

    def test_update_model(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        model = mm.add_model("Test Model", "C:/fake/model.gguf")
        updated = mm.update_model(model.id, name="New Name", gpu_layers=20)
        assert updated.name == "New Name"
        assert updated.gpu_layers == 20

    def test_update_nonexistent_model_returns_none(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        result = mm.update_model("nonexistent", name="X")
        assert result is None

    def test_remove_nonexistent_model_no_error(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        mm.remove_model("nonexistent")  # Should not raise


class TestResourceEstimation:
    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Real model file not available")
    def test_estimate_resources_structure(self):
        mm = ModelManager.instance()
        result = mm.estimate_resources(str(REAL_MODEL), gpu_layers=10, context_length=4096)
        assert "vram_needed_mb" in result
        assert "ram_needed_mb" in result
        assert "vram_available_mb" in result
        assert "ram_available_mb" in result
        assert "gpu_layers" in result
        assert "total_layers" in result
        assert "context_memory_mb" in result
        assert "fits_vram" in result
        assert "fits_ram" in result
        assert "feasible" in result

    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Real model file not available")
    def test_estimate_resources_values_sensible(self):
        mm = ModelManager.instance()
        result = mm.estimate_resources(str(REAL_MODEL), gpu_layers=10, context_length=4096)
        assert result["vram_needed_mb"] >= 0
        assert result["ram_needed_mb"] >= 0
        assert result["total_layers"] > 0
        assert result["gpu_layers"] == 10


class TestRecommendGpuLayers:
    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Real model file not available")
    def test_returns_int(self):
        mm = ModelManager.instance()
        layers = mm.recommend_gpu_layers(str(REAL_MODEL))
        assert isinstance(layers, int)

    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Real model file not available")
    def test_returns_non_negative(self):
        mm = ModelManager.instance()
        layers = mm.recommend_gpu_layers(str(REAL_MODEL))
        assert layers >= 0


class TestLoadUnload:
    @pytest.mark.asyncio
    async def test_load_model_not_in_registry(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        result = await mm.load_model("nonexistent_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_model_no_binary(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        model = mm.add_model("Test", "C:/fake/model.gguf")
        with patch("app.services.model_manager.BinaryManager") as mock_bm:
            mock_instance = MagicMock()
            mock_instance.is_available = False
            mock_bm.instance.return_value = mock_instance
            result = await mm.load_model(model.id)
            assert result is False

    @pytest.mark.asyncio
    async def test_unload_when_nothing_loaded(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        await mm.unload_model()  # Should not raise

    def test_get_status_idle(self, tmp_path):
        (tmp_path / "models.json").write_text("[]", encoding="utf-8")
        mm = ModelManager(app_dir=tmp_path)
        status = mm.get_status()
        assert status["loaded"] is False
        assert status["loading"] is False
        assert status["current_model"] is None


class TestAPIEndpoints:
    def test_scan_endpoint(self):
        response = client.get("/api/models/scan")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_registry_get(self):
        response = client.get("/api/models/registry")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_status_endpoint(self):
        response = client.get("/api/models/status")
        assert response.status_code == 200
        data = response.json()
        assert "loaded" in data
        assert "loading" in data

    def test_unload_endpoint(self):
        response = client.post("/api/models/unload")
        assert response.status_code == 200

    def test_load_nonexistent_model(self):
        response = client.post("/api/models/load/nonexistent_id")
        assert response.status_code in (404, 200)

    def test_registry_add_model(self):
        response = client.post(
            "/api/models/registry",
            json={"name": "Test Model", "path": "C:/fake/model.gguf"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data

    def test_registry_delete_nonexistent(self):
        response = client.delete("/api/models/registry/nonexistent_id")
        assert response.status_code == 200
