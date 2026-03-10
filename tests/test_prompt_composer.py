"""Tests for PromptComposer service."""

import pytest
from pathlib import Path

from app.services.prompt_composer import PromptComposer


class TestComposeSystemPrompt:
    def test_all_files_present(self, tmp_path):
        """compose_system_prompt concatenates all files with separator."""
        (tmp_path / "system-prompt.md").write_text("You are helpful.", encoding="utf-8")
        (tmp_path / "identity.md").write_text("Name: Bot", encoding="utf-8")
        (tmp_path / "user.md").write_text("User prefers short answers.", encoding="utf-8")
        (tmp_path / "memory.md").write_text("User likes Python.", encoding="utf-8")

        composer = PromptComposer(app_dir=tmp_path)
        result = composer.compose_system_prompt()

        assert "You are helpful." in result
        assert "Name: Bot" in result
        assert "User prefers short answers." in result
        assert "User likes Python." in result
        assert "\n\n---\n\n" in result

        parts = result.split("\n\n---\n\n")
        assert len(parts) == 4

    def test_skips_empty_files(self, tmp_path):
        """compose_system_prompt skips files that are empty."""
        (tmp_path / "system-prompt.md").write_text("System prompt here.", encoding="utf-8")
        (tmp_path / "identity.md").write_text("", encoding="utf-8")
        (tmp_path / "user.md").write_text("User info.", encoding="utf-8")
        (tmp_path / "memory.md").write_text("", encoding="utf-8")

        composer = PromptComposer(app_dir=tmp_path)
        result = composer.compose_system_prompt()

        parts = result.split("\n\n---\n\n")
        assert len(parts) == 2
        assert "System prompt here." in result
        assert "User info." in result

    def test_skips_missing_files(self, tmp_path):
        """compose_system_prompt skips files that don't exist."""
        (tmp_path / "system-prompt.md").write_text("Hello.", encoding="utf-8")

        composer = PromptComposer(app_dir=tmp_path)
        result = composer.compose_system_prompt()
        assert result == "Hello."

    def test_all_empty_returns_empty(self, tmp_path):
        """compose_system_prompt returns empty string when all files are empty."""
        (tmp_path / "system-prompt.md").write_text("", encoding="utf-8")
        (tmp_path / "identity.md").write_text("", encoding="utf-8")
        (tmp_path / "user.md").write_text("", encoding="utf-8")
        (tmp_path / "memory.md").write_text("", encoding="utf-8")

        composer = PromptComposer(app_dir=tmp_path)
        result = composer.compose_system_prompt()
        assert result == ""


class TestGetFile:
    def test_returns_content(self, tmp_path):
        """get_file returns the content of the file."""
        (tmp_path / "system-prompt.md").write_text("My prompt.", encoding="utf-8")

        composer = PromptComposer(app_dir=tmp_path)
        result = composer.get_file("system-prompt.md")
        assert result == "My prompt."

    def test_returns_empty_for_missing(self, tmp_path):
        """get_file returns empty string for missing files."""
        composer = PromptComposer(app_dir=tmp_path)
        result = composer.get_file("system-prompt.md")
        assert result == ""


class TestSaveFile:
    def test_saves_allowed_file(self, tmp_path):
        """save_file writes content for allowed filenames."""
        composer = PromptComposer(app_dir=tmp_path)

        for filename in ["system-prompt.md", "identity.md", "user.md", "memory.md"]:
            composer.save_file(filename, f"Content for {filename}")
            assert (tmp_path / filename).read_text(encoding="utf-8") == f"Content for {filename}"

    def test_rejects_disallowed_file(self, tmp_path):
        """save_file raises ValueError for disallowed filenames."""
        composer = PromptComposer(app_dir=tmp_path)

        with pytest.raises(ValueError, match="not allowed"):
            composer.save_file("evil.md", "hacked")

        with pytest.raises(ValueError, match="not allowed"):
            composer.save_file("config.json", "bad")

        with pytest.raises(ValueError, match="not allowed"):
            composer.save_file("../etc/passwd", "bad")
