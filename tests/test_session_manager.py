"""Tests for session manager service and sessions API."""

import pytest
import pytest_asyncio
import time
from pathlib import Path
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport

from app.services.session_manager import SessionManager


@pytest_asyncio.fixture
async def sm(tmp_path):
    """Create a SessionManager backed by a temp directory."""
    with patch("app.services.session_manager.SESSIONS_DIR", tmp_path):
        manager = SessionManager()
        await manager.init_db()
        yield manager


@pytest.mark.asyncio
class TestInitDb:
    async def test_creates_tables(self, sm):
        """init_db should create sessions and messages tables."""
        import aiosqlite

        async with aiosqlite.connect(str(sm._db_path)) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in await cursor.fetchall()]
        assert "sessions" in tables
        assert "messages" in tables

    async def test_wal_mode_enabled(self, sm):
        """init_db should enable WAL journal mode."""
        import aiosqlite

        async with aiosqlite.connect(str(sm._db_path)) as db:
            cursor = await db.execute("PRAGMA journal_mode")
            mode = (await cursor.fetchone())[0]
        assert mode == "wal"


@pytest.mark.asyncio
class TestCreateSession:
    async def test_returns_valid_session(self, sm):
        session = await sm.create_session("Test Chat", "qwen3.5")
        assert "id" in session
        assert len(session["id"]) == 8
        assert session["title"] == "Test Chat"
        assert session["model_id"] == "qwen3.5"
        assert "created_at" in session
        assert "updated_at" in session

    async def test_unique_ids(self, sm):
        s1 = await sm.create_session("Chat 1", "model-a")
        s2 = await sm.create_session("Chat 2", "model-b")
        assert s1["id"] != s2["id"]


@pytest.mark.asyncio
class TestListSessions:
    async def test_returns_sessions_ordered_by_updated_at_desc(self, sm):
        s1 = await sm.create_session("First", "m1")
        s2 = await sm.create_session("Second", "m2")
        # Update s1 so it becomes the most recently updated
        await sm.update_session(s1["id"], title="First Updated")
        sessions = await sm.list_sessions()
        assert len(sessions) == 2
        assert sessions[0]["id"] == s1["id"]
        assert sessions[1]["id"] == s2["id"]

    async def test_empty_list(self, sm):
        sessions = await sm.list_sessions()
        assert sessions == []


@pytest.mark.asyncio
class TestGetSession:
    async def test_returns_session(self, sm):
        created = await sm.create_session("My Chat", "m1")
        fetched = await sm.get_session(created["id"])
        assert fetched is not None
        assert fetched["title"] == "My Chat"

    async def test_returns_none_for_missing(self, sm):
        result = await sm.get_session("nonexist")
        assert result is None


@pytest.mark.asyncio
class TestUpdateSession:
    async def test_changes_title(self, sm):
        session = await sm.create_session("Old Title", "m1")
        await sm.update_session(session["id"], title="New Title")
        updated = await sm.get_session(session["id"])
        assert updated["title"] == "New Title"

    async def test_changes_model_id(self, sm):
        session = await sm.create_session("Chat", "old-model")
        await sm.update_session(session["id"], model_id="new-model")
        updated = await sm.get_session(session["id"])
        assert updated["model_id"] == "new-model"


@pytest.mark.asyncio
class TestDeleteSession:
    async def test_removes_session(self, sm):
        session = await sm.create_session("To Delete", "m1")
        await sm.add_message(session["id"], "user", "hello")
        await sm.delete_session(session["id"])
        assert await sm.get_session(session["id"]) is None

    async def test_cascade_deletes_messages(self, sm):
        session = await sm.create_session("To Delete", "m1")
        await sm.add_message(session["id"], "user", "hello")
        await sm.delete_session(session["id"])
        messages = await sm.get_messages(session["id"])
        assert messages == []


@pytest.mark.asyncio
class TestAddMessage:
    async def test_stores_message_with_all_fields(self, sm):
        session = await sm.create_session("Chat", "m1")
        msg = await sm.add_message(
            session["id"],
            "assistant",
            "Hello!",
            token_count=10,
            tokens_per_second=42.5,
            time_to_first_token_ms=120.3,
        )
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert msg["token_count"] == 10
        assert msg["tokens_per_second"] == 42.5
        assert msg["time_to_first_token_ms"] == 120.3
        assert "id" in msg
        assert "timestamp" in msg

    async def test_default_performance_fields(self, sm):
        session = await sm.create_session("Chat", "m1")
        msg = await sm.add_message(session["id"], "user", "hi")
        assert msg["token_count"] == 0
        assert msg["tokens_per_second"] == 0
        assert msg["time_to_first_token_ms"] == 0


@pytest.mark.asyncio
class TestGetMessages:
    async def test_returns_messages_in_order(self, sm):
        session = await sm.create_session("Chat", "m1")
        await sm.add_message(session["id"], "user", "first")
        await sm.add_message(session["id"], "assistant", "second")
        await sm.add_message(session["id"], "user", "third")
        messages = await sm.get_messages(session["id"])
        assert len(messages) == 3
        assert messages[0]["content"] == "first"
        assert messages[1]["content"] == "second"
        assert messages[2]["content"] == "third"


@pytest.mark.asyncio
class TestGetSessionTokenCount:
    async def test_sums_correctly(self, sm):
        session = await sm.create_session("Chat", "m1")
        await sm.add_message(session["id"], "user", "hi", token_count=5)
        await sm.add_message(
            session["id"], "assistant", "hello", token_count=12
        )
        total = await sm.get_session_token_count(session["id"])
        assert total == 17

    async def test_returns_zero_for_empty(self, sm):
        session = await sm.create_session("Chat", "m1")
        total = await sm.get_session_token_count(session["id"])
        assert total == 0


# --- API Endpoint Tests ---


@pytest_asyncio.fixture
async def test_client(tmp_path):
    """Create an AsyncClient with a patched SessionManager."""
    with patch("app.services.session_manager.SESSIONS_DIR", tmp_path):
        from app.main import app
        from app.routers.sessions import get_session_manager

        manager = SessionManager()
        await manager.init_db()

        app.dependency_overrides[get_session_manager] = lambda: manager

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestSessionsAPI:
    async def test_create_session(self, test_client):
        resp = await test_client.post(
            "/api/sessions/", json={"title": "New Chat", "model_id": "m1"}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "New Chat"
        assert len(data["id"]) == 8

    async def test_list_sessions(self, test_client):
        await test_client.post(
            "/api/sessions/", json={"title": "Chat 1", "model_id": "m1"}
        )
        resp = await test_client.get("/api/sessions/")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    async def test_get_session(self, test_client):
        create_resp = await test_client.post(
            "/api/sessions/", json={"title": "Chat", "model_id": "m1"}
        )
        sid = create_resp.json()["id"]
        resp = await test_client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Chat"

    async def test_get_session_not_found(self, test_client):
        resp = await test_client.get("/api/sessions/nonexist")
        assert resp.status_code == 404

    async def test_update_session(self, test_client):
        create_resp = await test_client.post(
            "/api/sessions/", json={"title": "Old", "model_id": "m1"}
        )
        sid = create_resp.json()["id"]
        resp = await test_client.put(
            f"/api/sessions/{sid}", json={"title": "New"}
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "New"

    async def test_delete_session(self, test_client):
        create_resp = await test_client.post(
            "/api/sessions/", json={"title": "Delete Me", "model_id": "m1"}
        )
        sid = create_resp.json()["id"]
        resp = await test_client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 204

    async def test_get_messages(self, test_client):
        create_resp = await test_client.post(
            "/api/sessions/", json={"title": "Chat", "model_id": "m1"}
        )
        sid = create_resp.json()["id"]
        resp = await test_client.get(f"/api/sessions/{sid}/messages")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_get_token_count(self, test_client):
        create_resp = await test_client.post(
            "/api/sessions/", json={"title": "Chat", "model_id": "m1"}
        )
        sid = create_resp.json()["id"]
        resp = await test_client.get(f"/api/sessions/{sid}/tokens")
        assert resp.status_code == 200
        assert resp.json()["token_count"] == 0
