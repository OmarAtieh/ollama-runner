import mimetypes
# Fix MIME types FIRST — must happen before StaticFiles is instantiated.
# Windows Python defaults .js to text/plain, which blocks ES module loading.
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pathlib import Path
from app.config import ensure_dirs, load_config
from app.routers.system import router as system_router
from app.routers.models import router as models_router
from app.routers.sessions import router as sessions_router, projects_router
from app.routers.chat import router as chat_router
from app.routers.config import router as config_router
from app.services.session_manager import SessionManager
from app.services.model_manager import ModelManager
from app.services.binary_manager import BinaryManager

log = logging.getLogger(__name__)

ensure_dirs()

session_manager = SessionManager()
model_manager = ModelManager.instance()
binary_manager = BinaryManager.instance()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    await session_manager.init_db()
    log.info("Session database initialized")

    config = load_config()
    if config.get("load_model_on_start") and config.get("default_model_id"):
        if binary_manager.is_available:
            log.info(
                "Auto-loading default model: %s", config["default_model_id"]
            )
            asyncio.create_task(
                model_manager.load_model(config["default_model_id"])
            )
        else:
            log.warning(
                "Auto-load requested but llama-server binary not available"
            )

    yield

    # --- Shutdown ---
    await model_manager.unload_model()
    log.info("Model unloaded during shutdown")

    # Close the shared aiohttp session used by chat.py
    from app.routers import chat as _chat_module
    if _chat_module._http_session is not None and not _chat_module._http_session.closed:
        await _chat_module._http_session.close()
        log.info("Closed chat HTTP session")


app = FastAPI(title="OllamaRunner", version="0.1.0", lifespan=lifespan)
app.include_router(system_router)
app.include_router(models_router)
app.include_router(sessions_router)
app.include_router(projects_router)
app.include_router(chat_router)
app.include_router(config_router)

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


@app.get("/favicon.ico")
async def favicon():
    # Return empty response to suppress 404
    return Response(content=b"", media_type="image/x-icon")
