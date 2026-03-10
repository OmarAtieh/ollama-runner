"""Model scanning, registry management, and loading endpoints."""

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import load_config
from app.services.model_scanner import scan_models_directory
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/api/models", tags=["models"])


# -- Pydantic request/response models ---------------------------------------

class AddModelRequest(BaseModel):
    name: str
    path: str
    gpu_layers: int = -1
    context_default: int = 4096
    context_recommended: int = 8192
    context_max: int = 32768
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1


class UpdateModelRequest(BaseModel):
    name: Optional[str] = None
    gpu_layers: Optional[int] = None
    context_default: Optional[int] = None
    context_recommended: Optional[int] = None
    context_max: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None


class LoadModelRequest(BaseModel):
    context_length: Optional[int] = None


# -- Endpoints ---------------------------------------------------------------

@router.get("/scan")
async def scan_models():
    """Scan models directory and return list of discovered GGUF files."""
    config = load_config()
    models_dir = Path(config.get("models_directory", ""))
    models = scan_models_directory(models_dir)
    return [asdict(m) for m in models]


@router.get("/registry")
async def list_registry():
    """List all registered models."""
    mm = ModelManager.instance()
    models = mm.load_registry()
    return [asdict(m) for m in models]


@router.post("/registry")
async def add_to_registry(req: AddModelRequest):
    """Add a model to the registry."""
    mm = ModelManager.instance()
    kwargs = req.model_dump(exclude={"name", "path"})
    model = mm.add_model(req.name, req.path, **kwargs)
    return asdict(model)


@router.put("/registry/{model_id}")
async def update_in_registry(model_id: str, req: UpdateModelRequest):
    """Update a registered model's configuration."""
    mm = ModelManager.instance()
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = mm.update_model(model_id, **updates)
    if result is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return asdict(result)


@router.delete("/registry/{model_id}")
async def remove_from_registry(model_id: str):
    """Remove a model from the registry."""
    mm = ModelManager.instance()
    mm.remove_model(model_id)
    return {"status": "ok"}


@router.post("/load/{model_id}")
async def load_model(model_id: str, req: LoadModelRequest = LoadModelRequest()):
    """Load a model (start llama-server)."""
    mm = ModelManager.instance()
    model = mm.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found in registry")
    success = await mm.load_model(model_id, context_length=req.context_length)
    status = mm.get_status()
    if not success:
        status["error"] = status.get("error") or "Failed to load model"
    return status


@router.post("/unload")
async def unload_model():
    """Unload the current model (stop llama-server)."""
    mm = ModelManager.instance()
    await mm.unload_model()
    return mm.get_status()


@router.get("/status")
async def get_status():
    """Get current model loading/running status."""
    mm = ModelManager.instance()
    return mm.get_status()


@router.get("/estimate/{model_id}")
async def estimate_resources(model_id: str, gpu_layers: int = -1, context_length: int = 4096):
    """Estimate resource requirements for a model configuration."""
    mm = ModelManager.instance()
    model = mm.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found in registry")
    layers = gpu_layers if gpu_layers >= 0 else (model.gpu_layers if model.gpu_layers >= 0 else 0)
    return mm.estimate_resources(model.path, layers, context_length)


@router.get("/recommend-layers")
async def recommend_layers(model_path: str):
    """Recommend GPU layers for a model based on available VRAM."""
    mm = ModelManager.instance()
    layers = mm.recommend_gpu_layers(model_path)
    return {"model_path": model_path, "recommended_gpu_layers": layers}
