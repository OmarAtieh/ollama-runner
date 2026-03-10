from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.config import ensure_dirs
from app.routers.system import router as system_router
from app.routers.models import router as models_router

ensure_dirs()

app = FastAPI(title="OllamaRunner", version="0.1.0")
app.include_router(system_router)
app.include_router(models_router)

static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/")
async def root():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "OllamaRunner - UI not yet built"}
