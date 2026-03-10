"""Prompt composer: reads and concatenates prompt files for system prompt."""

from pathlib import Path

from app.config import APP_DIR

ALLOWED_FILES = {"system-prompt.md", "identity.md", "user.md", "memory.md"}
PROMPT_FILE_ORDER = ["system-prompt.md", "identity.md", "user.md", "memory.md"]


class PromptComposer:
    """Reads prompt files from APP_DIR and composes system prompts."""

    def __init__(self, app_dir: Path | None = None):
        self._app_dir = app_dir or APP_DIR

    def compose_system_prompt(self) -> str:
        """Read and concatenate prompt files, joined by separator, skipping empty ones."""
        parts = []
        for filename in PROMPT_FILE_ORDER:
            content = self.get_file(filename)
            if content.strip():
                parts.append(content)
        return "\n\n---\n\n".join(parts)

    def get_file(self, filename: str) -> str:
        """Read a prompt file. Returns empty string if file doesn't exist."""
        filepath = self._app_dir / filename
        if not filepath.exists():
            return ""
        return filepath.read_text(encoding="utf-8")

    def save_file(self, filename: str, content: str) -> None:
        """Write a prompt file. Only allowed filenames are accepted."""
        if filename not in ALLOWED_FILES:
            raise ValueError(f"Filename '{filename}' is not allowed. Allowed: {ALLOWED_FILES}")
        filepath = self._app_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
