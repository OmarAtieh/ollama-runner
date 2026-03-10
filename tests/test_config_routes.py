import json
import pytest
from httpx import AsyncClient, ASGITransport
from pathlib import Path


@pytest.fixture()
def fake_app_dir(tmp_path, monkeypatch):
    """Redirect APP_DIR to a temp directory and seed default files."""
    # Patch APP_DIR in all modules that import it
    monkeypatch.setattr("app.config.APP_DIR", tmp_path)
    monkeypatch.setattr("app.routers.config.APP_DIR", tmp_path)
    monkeypatch.setattr("app.services.prompt_composer.APP_DIR", tmp_path)

    # Seed a default config
    default_config = {
        "models_directory": str(tmp_path / "models"),
        "host": "127.0.0.1",
        "port": 8080,
        "default_model_id": None,
        "load_model_on_start": False,
        "vram_limit_percent": 95,
        "ram_limit_percent": 85,
        "theme": "dark",
    }
    (tmp_path / "config.json").write_text(json.dumps(default_config, indent=2), encoding="utf-8")

    # Seed prompt files
    for fname in ("system-prompt.md", "identity.md", "user.md", "memory.md"):
        (tmp_path / fname).write_text("", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def client(fake_app_dir):
    """Yield an async test client with patched APP_DIR."""
    from app.main import app
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_get_config_returns_default_structure(client):
    async with client:
        response = await client.get("/api/config/")
    assert response.status_code == 200
    data = response.json()
    for key in ("models_directory", "host", "port", "default_model_id",
                "load_model_on_start", "vram_limit_percent", "ram_limit_percent", "theme"):
        assert key in data


@pytest.mark.asyncio
async def test_put_config_updates_fields_without_overwriting(client, fake_app_dir):
    async with client:
        # Update only theme
        response = await client.put("/api/config/", json={"theme": "light"})
        assert response.status_code == 200
        data = response.json()
        assert data["theme"] == "light"
        # Other fields should remain unchanged
        assert data["vram_limit_percent"] == 95
        assert data["ram_limit_percent"] == 85

        # Verify persisted
        response2 = await client.get("/api/config/")
        assert response2.json()["theme"] == "light"
        assert response2.json()["vram_limit_percent"] == 95


@pytest.mark.asyncio
async def test_get_prompt_file(client, fake_app_dir):
    # Write some content to system-prompt.md
    (fake_app_dir / "system-prompt.md").write_text("Hello world", encoding="utf-8")
    async with client:
        response = await client.get("/api/config/prompt/system-prompt.md")
    assert response.status_code == 200
    assert response.json()["content"] == "Hello world"


@pytest.mark.asyncio
async def test_put_prompt_file_and_read_back(client, fake_app_dir):
    async with client:
        response = await client.put(
            "/api/config/prompt/identity.md",
            json={"content": "I am a helpful assistant."},
        )
        assert response.status_code == 200

        response2 = await client.get("/api/config/prompt/identity.md")
        assert response2.json()["content"] == "I am a helpful assistant."


@pytest.mark.asyncio
async def test_invalid_prompt_filename_returns_400(client):
    async with client:
        response = await client.get("/api/config/prompt/evil.txt")
        assert response.status_code == 400

        response2 = await client.put(
            "/api/config/prompt/evil.txt",
            json={"content": "bad"},
        )
        assert response2.status_code == 400
