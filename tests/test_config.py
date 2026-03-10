import json
import pytest
from pathlib import Path
from unittest.mock import patch

from app.config import (
    DEFAULT_CONFIG,
    ensure_dirs,
    load_config,
    save_config,
)


@pytest.fixture
def tmp_app_dir(tmp_path):
    """Patch APP_DIR and derived paths to use a temporary directory."""
    app_dir = tmp_path / ".ollamarunner"
    bin_dir = app_dir / "bin"
    sessions_dir = app_dir / "sessions"
    with (
        patch("app.config.APP_DIR", app_dir),
        patch("app.config.BIN_DIR", bin_dir),
        patch("app.config.SESSIONS_DIR", sessions_dir),
    ):
        yield app_dir


class TestEnsureDirs:
    def test_creates_all_directories(self, tmp_app_dir):
        ensure_dirs()
        assert tmp_app_dir.is_dir()
        assert (tmp_app_dir / "bin").is_dir()
        assert (tmp_app_dir / "sessions").is_dir()

    def test_creates_config_json(self, tmp_app_dir):
        ensure_dirs()
        config_path = tmp_app_dir / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data == DEFAULT_CONFIG

    def test_creates_default_files(self, tmp_app_dir):
        ensure_dirs()
        for filename in [
            "system-prompt.md",
            "identity.md",
            "user.md",
            "memory.md",
            "models.json",
        ]:
            assert (tmp_app_dir / filename).exists()

    def test_does_not_overwrite_existing_config(self, tmp_app_dir):
        tmp_app_dir.mkdir(parents=True, exist_ok=True)
        custom = {"custom_key": "custom_value"}
        (tmp_app_dir / "config.json").write_text(
            json.dumps(custom), encoding="utf-8"
        )
        ensure_dirs()
        data = json.loads(
            (tmp_app_dir / "config.json").read_text(encoding="utf-8")
        )
        assert data == custom


class TestLoadConfig:
    def test_returns_defaults_when_no_file(self, tmp_app_dir):
        config = load_config()
        assert config == DEFAULT_CONFIG

    def test_merges_stored_values(self, tmp_app_dir):
        ensure_dirs()
        stored = {"theme": "light", "port": 9090}
        (tmp_app_dir / "config.json").write_text(
            json.dumps(stored), encoding="utf-8"
        )
        config = load_config()
        assert config["theme"] == "light"
        assert config["port"] == 9090
        # defaults still present
        assert config["host"] == "127.0.0.1"


class TestSaveConfig:
    def test_persists_config(self, tmp_app_dir):
        tmp_app_dir.mkdir(parents=True, exist_ok=True)
        config = dict(DEFAULT_CONFIG)
        config["theme"] = "light"
        save_config(config)
        data = json.loads(
            (tmp_app_dir / "config.json").read_text(encoding="utf-8")
        )
        assert data["theme"] == "light"
