import json
import os
from pathlib import Path

APP_DIR = Path.home() / ".ollamarunner"
MODELS_DIR = Path.home() / ".lmstudio" / "models"
BIN_DIR = APP_DIR / "bin"
SESSIONS_DIR = APP_DIR / "sessions"

DEFAULT_CONFIG = {
    "models_directory": str(MODELS_DIR),
    "host": "127.0.0.1",
    "port": 8080,
    "default_model_id": None,
    "load_model_on_start": False,
    "vram_limit_percent": 95,
    "ram_limit_percent": 85,
    "theme": "dark",
}

_DEFAULT_FILES = {
    "system-prompt.md": "",
    "identity.md": "",
    "user.md": "",
    "memory.md": "",
    "models.json": "[]",
}


def ensure_dirs():
    """Create all required directories and default files."""
    for d in [APP_DIR, BIN_DIR, SESSIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    config_path = APP_DIR / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")

    for filename, default_content in _DEFAULT_FILES.items():
        filepath = APP_DIR / filename
        if not filepath.exists():
            filepath.write_text(default_content, encoding="utf-8")


def load_config() -> dict:
    """Read config.json and merge with defaults."""
    config_path = APP_DIR / "config.json"
    merged = dict(DEFAULT_CONFIG)
    if config_path.exists():
        try:
            stored = json.loads(config_path.read_text(encoding="utf-8"))
            merged.update(stored)
        except (json.JSONDecodeError, OSError):
            pass
    return merged


def save_config(config: dict) -> None:
    """Write config to config.json."""
    config_path = APP_DIR / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
