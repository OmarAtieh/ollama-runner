"""Tests for app startup/shutdown lifecycle (lifespan)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

import app.main as main_module


@pytest.mark.asyncio
async def test_startup_initializes_session_db():
    """Startup should call session_manager.init_db(), creating the sessions table."""
    mock_sm = AsyncMock()
    mock_mm = AsyncMock()

    with patch.object(main_module, "session_manager", mock_sm), \
         patch.object(main_module, "model_manager", mock_mm), \
         patch.object(main_module, "load_config", return_value={}):

        async with main_module.lifespan(main_module.app):
            mock_sm.init_db.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_calls_unload_model():
    """Shutdown should call model_manager.unload_model()."""
    mock_sm = AsyncMock()
    mock_mm = AsyncMock()

    with patch.object(main_module, "session_manager", mock_sm), \
         patch.object(main_module, "model_manager", mock_mm), \
         patch.object(main_module, "load_config", return_value={}):

        async with main_module.lifespan(main_module.app):
            pass

        mock_mm.unload_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_endpoint_works_after_startup():
    """The /api/health endpoint should work normally with the lifespan."""
    transport = ASGITransport(app=main_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_auto_load_triggered_when_both_flags_set():
    """Auto-load should fire when load_model_on_start=True AND default_model_id is set."""
    config = {"load_model_on_start": True, "default_model_id": "abc123"}
    mock_sm = AsyncMock()
    mock_mm = AsyncMock()
    mock_mm.load_model = AsyncMock(return_value=True)
    mock_bm = MagicMock()
    mock_bm.is_available = True

    with patch.object(main_module, "session_manager", mock_sm), \
         patch.object(main_module, "model_manager", mock_mm), \
         patch.object(main_module, "binary_manager", mock_bm), \
         patch.object(main_module, "load_config", return_value=config):

        async with main_module.lifespan(main_module.app):
            # Give the background task a chance to run
            await asyncio.sleep(0.1)
            mock_mm.load_model.assert_awaited_once_with("abc123")


@pytest.mark.asyncio
async def test_auto_load_not_triggered_without_model_id():
    """Auto-load should NOT fire when default_model_id is missing."""
    config = {"load_model_on_start": True}
    mock_sm = AsyncMock()
    mock_mm = AsyncMock()
    mock_bm = MagicMock()
    mock_bm.is_available = True

    with patch.object(main_module, "session_manager", mock_sm), \
         patch.object(main_module, "model_manager", mock_mm), \
         patch.object(main_module, "binary_manager", mock_bm), \
         patch.object(main_module, "load_config", return_value=config):

        async with main_module.lifespan(main_module.app):
            pass

        mock_mm.load_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_load_not_triggered_without_flag():
    """Auto-load should NOT fire when load_model_on_start is False."""
    config = {"load_model_on_start": False, "default_model_id": "abc123"}
    mock_sm = AsyncMock()
    mock_mm = AsyncMock()
    mock_bm = MagicMock()
    mock_bm.is_available = True

    with patch.object(main_module, "session_manager", mock_sm), \
         patch.object(main_module, "model_manager", mock_mm), \
         patch.object(main_module, "binary_manager", mock_bm), \
         patch.object(main_module, "load_config", return_value=config):

        async with main_module.lifespan(main_module.app):
            pass

        mock_mm.load_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_load_skipped_when_binary_not_available():
    """Auto-load should be skipped if llama-server binary is not available."""
    config = {"load_model_on_start": True, "default_model_id": "abc123"}
    mock_sm = AsyncMock()
    mock_mm = AsyncMock()
    mock_bm = MagicMock()
    mock_bm.is_available = False

    with patch.object(main_module, "session_manager", mock_sm), \
         patch.object(main_module, "model_manager", mock_mm), \
         patch.object(main_module, "binary_manager", mock_bm), \
         patch.object(main_module, "load_config", return_value=config):

        async with main_module.lifespan(main_module.app):
            pass

        mock_mm.load_model.assert_not_awaited()
