"""Config and prompt file management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import APP_DIR, load_config, save_config

router = APIRouter(prefix="/api/config", tags=["config"])

ALLOWED_PROMPT_FILES = {"system-prompt.md", "identity.md", "user.md", "memory.md"}


class ConfigUpdateRequest(BaseModel):
    models_directory: str | None = None
    default_model_id: str | None = None
    load_model_on_start: bool | None = None
    vram_limit_percent: int | None = None
    ram_limit_percent: int | None = None
    theme: str | None = None


class PromptFileRequest(BaseModel):
    content: str


@router.get("/")
async def get_config():
    """Return the current application config."""
    return load_config()


@router.put("/")
async def update_config(req: ConfigUpdateRequest):
    """Update config with non-None fields and save."""
    config = load_config()
    updates = req.model_dump(exclude_none=True)
    config.update(updates)
    save_config(config)
    return config


@router.get("/prompt/{filename}")
async def get_prompt_file(filename: str):
    """Read a prompt file by name."""
    if filename not in ALLOWED_PROMPT_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid prompt filename: {filename}")
    filepath = APP_DIR / filename
    content = ""
    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
    return {"filename": filename, "content": content}


@router.put("/prompt/{filename}")
async def save_prompt_file(filename: str, req: PromptFileRequest):
    """Save content to a prompt file."""
    if filename not in ALLOWED_PROMPT_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid prompt filename: {filename}")
    filepath = APP_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(req.content, encoding="utf-8")
    return {"filename": filename, "content": req.content}
