"""Tests for chat WebSocket endpoint."""

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient

from app.services.session_manager import SessionManager


@pytest_asyncio.fixture
async def session_manager(tmp_path):
    """Create a SessionManager backed by a temp directory."""
    with patch("app.services.session_manager.SESSIONS_DIR", tmp_path):
        manager = SessionManager()
        await manager.init_db()
        yield manager


@pytest.fixture
def mock_model_manager_not_loaded():
    """ModelManager that reports no model loaded."""
    mm = MagicMock()
    mm.get_status.return_value = {
        "loaded": False,
        "loading": False,
        "load_progress": 0.0,
        "current_model": None,
        "server_url": None,
        "error": None,
    }
    return mm


@pytest.fixture
def mock_model_manager_loaded():
    """ModelManager that reports a model loaded."""
    mm = MagicMock()
    mm.get_status.return_value = {
        "loaded": True,
        "loading": False,
        "load_progress": 1.0,
        "current_model": {"id": "test123", "name": "TestModel"},
        "server_url": "http://127.0.0.1:8081",
        "error": None,
    }
    return mm


class TestChatWebSocket:
    def test_websocket_connects(self, session_manager, mock_model_manager_loaded):
        """WebSocket endpoint accepts connections."""
        from app.main import app
        from app.routers.chat import get_session_manager as chat_get_sm
        from app.routers.chat import get_model_manager as chat_get_mm

        app.dependency_overrides[chat_get_sm] = lambda: session_manager
        app.dependency_overrides[chat_get_mm] = lambda: mock_model_manager_loaded

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/test-session") as ws:
            # Connection accepted - just close it
            pass

        app.dependency_overrides.clear()

    def test_error_when_no_model_loaded(self, session_manager, mock_model_manager_not_loaded):
        """Sends error message when no model is loaded."""
        from app.main import app
        from app.routers.chat import get_session_manager as chat_get_sm
        from app.routers.chat import get_model_manager as chat_get_mm

        app.dependency_overrides[chat_get_sm] = lambda: session_manager
        app.dependency_overrides[chat_get_mm] = lambda: mock_model_manager_not_loaded

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/test-session") as ws:
            ws.send_json({"content": "Hello!"})
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "No model loaded" in response["content"]

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_prompt_composer_is_used(self, tmp_path):
        """Verify that prompt composer builds system prompt from files."""
        from app.services.prompt_composer import PromptComposer

        (tmp_path / "system-prompt.md").write_text("Be helpful.", encoding="utf-8")
        (tmp_path / "identity.md").write_text("", encoding="utf-8")
        (tmp_path / "user.md").write_text("", encoding="utf-8")
        (tmp_path / "memory.md").write_text("", encoding="utf-8")

        composer = PromptComposer(app_dir=tmp_path)
        prompt = composer.compose_system_prompt()
        assert prompt == "Be helpful."
