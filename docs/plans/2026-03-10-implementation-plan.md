# OllamaRunner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-contained local LLM runner for Windows with web UI, auto-managed llama.cpp CUDA backend, and full model/session/resource management.

**Architecture:** FastAPI async backend orchestrating llama-server processes, vanilla HTML/CSS/JS frontend communicating via WebSocket for streaming and REST for config. SQLite for sessions, JSON for config, Markdown for prompt files. Modular service layer with clean interfaces.

**Tech Stack:** Python 3.13, FastAPI, uvicorn, aiohttp (SSE proxy), pynvml, psutil, SQLite (aiosqlite), vanilla HTML/CSS/JS

**Hardware:** RTX 2070 Max-Q (8GB VRAM), 32GB RAM, iGPU for display

**Model Directory:** `C:\Users\omar-\.lmstudio\models` (recursive GGUF scan)

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `run.py`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/config.py`

**Step 1: Create requirements.txt**

```
fastapi==0.115.12
uvicorn[standard]==0.34.2
aiohttp==3.11.18
aiosqlite==0.21.0
psutil==7.0.0
pynvml==12.0.0
python-multipart==0.0.20
```

**Step 2: Create run.py (entry point)**

```python
import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)
```

**Step 3: Create app directory structure**

```python
# app/__init__.py — empty

# app/config.py
import os
import json
from pathlib import Path

APP_DIR = Path(os.environ.get("OLLAMARUNNER_DIR", Path.home() / ".ollamarunner"))
MODELS_DIR = Path(os.environ.get("OLLAMARUNNER_MODELS", Path.home() / ".lmstudio" / "models"))
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
    "theme": "dark"
}

def ensure_dirs():
    """Create all required directories and default files."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)

    config_path = APP_DIR / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))

    for fname, default_content in [
        ("system-prompt.md", "You are a helpful assistant."),
        ("identity.md", ""),
        ("user.md", ""),
        ("memory.md", ""),
    ]:
        fpath = APP_DIR / fname
        if not fpath.exists():
            fpath.write_text(default_content)

    models_path = APP_DIR / "models.json"
    if not models_path.exists():
        models_path.write_text(json.dumps({"models": []}, indent=2))

def load_config() -> dict:
    ensure_dirs()
    config_path = APP_DIR / "config.json"
    with open(config_path) as f:
        stored = json.load(f)
    merged = {**DEFAULT_CONFIG, **stored}
    return merged

def save_config(config: dict):
    config_path = APP_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
```

**Step 4: Create app/main.py (minimal FastAPI app)**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.config import ensure_dirs

ensure_dirs()

app = FastAPI(title="OllamaRunner", version="0.1.0")

# Mount static files for the web UI
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
```

**Step 5: Install dependencies and verify**

Run: `pip install -r requirements.txt`
Run: `python run.py` — verify http://127.0.0.1:8080/api/health returns ok

**Step 6: Commit**
```
git add -A && git commit -m "feat: project scaffolding with FastAPI, config, and directory structure"
```

---

## Task 2: System Resource Monitor

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/system_monitor.py`
- Create: `app/routers/__init__.py`
- Create: `app/routers/system.py`
- Modify: `app/main.py` (add router)

**Step 1: Create system monitor service**

```python
# app/services/__init__.py — empty

# app/services/system_monitor.py
import psutil
from dataclasses import dataclass

@dataclass
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    gpu_utilization: int
    temperature: int
    is_display_attached: bool

@dataclass
class SystemResources:
    cpu_percent: float
    cpu_count: int
    ram_total_mb: int
    ram_used_mb: int
    ram_free_mb: int
    gpu: GpuInfo | None

class SystemMonitor:
    def __init__(self):
        self._nvml_initialized = False

    def _init_nvml(self):
        if self._nvml_initialized:
            return True
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            return True
        except Exception:
            return False

    def get_gpu_info(self) -> GpuInfo | None:
        if not self._init_nvml():
            return None
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            # Check if display is attached
            is_display = False
            try:
                mode = pynvml.nvmlDeviceGetDisplayMode(handle)
                is_display = mode == pynvml.NVML_FEATURE_ENABLED
            except Exception:
                pass

            return GpuInfo(
                name=name,
                vram_total_mb=info.total // (1024 * 1024),
                vram_used_mb=info.used // (1024 * 1024),
                vram_free_mb=info.free // (1024 * 1024),
                gpu_utilization=util.gpu,
                temperature=temp,
                is_display_attached=is_display,
            )
        except Exception:
            return None

    def get_resources(self) -> SystemResources:
        mem = psutil.virtual_memory()
        return SystemResources(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            cpu_count=psutil.cpu_count(),
            ram_total_mb=mem.total // (1024 * 1024),
            ram_used_mb=mem.used // (1024 * 1024),
            ram_free_mb=mem.available // (1024 * 1024),
            gpu=self.get_gpu_info(),
        )

    def get_vram_limit_mb(self, limit_percent: int = 95) -> int:
        gpu = self.get_gpu_info()
        if not gpu:
            return 0
        return int(gpu.vram_total_mb * limit_percent / 100)

    def get_ram_limit_mb(self, limit_percent: int = 85) -> int:
        mem = psutil.virtual_memory()
        return int(mem.total // (1024 * 1024) * limit_percent / 100)

# Singleton
system_monitor = SystemMonitor()
```

**Step 2: Create system router**

```python
# app/routers/__init__.py — empty

# app/routers/system.py
from fastapi import APIRouter
from app.services.system_monitor import system_monitor
import dataclasses

router = APIRouter(prefix="/api/system", tags=["system"])

@router.get("/resources")
async def get_resources():
    resources = system_monitor.get_resources()
    data = dataclasses.asdict(resources)
    return data
```

**Step 3: Register router in main.py**

Add to app/main.py:
```python
from app.routers import system
app.include_router(system.router)
```

**Step 4: Verify**

Run: `python run.py` and hit `/api/system/resources` — should show CPU, RAM, GPU stats

**Step 5: Commit**
```
git add -A && git commit -m "feat: system resource monitor with GPU/RAM/CPU detection"
```

---

## Task 3: llama-server Binary Manager

**Files:**
- Create: `app/services/binary_manager.py`

**Step 1: Create binary manager**

Downloads the CUDA-enabled llama-server from GitHub releases. Detects platform, downloads correct binary, extracts to BIN_DIR.

```python
# app/services/binary_manager.py
import os
import sys
import zipfile
import asyncio
import aiohttp
from pathlib import Path
from app.config import BIN_DIR

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
BINARY_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"

class BinaryManager:
    def __init__(self):
        self.binary_path = BIN_DIR / BINARY_NAME
        self._download_progress = 0.0
        self._download_status = "idle"  # idle, downloading, extracting, ready, error

    @property
    def is_available(self) -> bool:
        return self.binary_path.exists()

    @property
    def download_progress(self) -> float:
        return self._download_progress

    @property
    def download_status(self) -> str:
        return self._download_status

    async def get_latest_release_url(self) -> str | None:
        """Find the CUDA-enabled Windows release asset URL."""
        url = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

        # Look for Windows CUDA binary
        for asset in data.get("assets", []):
            name = asset["name"].lower()
            if "win" in name and "cuda" in name and name.endswith(".zip"):
                # Prefer the one without "vulkan"
                if "vulkan" not in name:
                    return asset["browser_download_url"]

        # Fallback: any Windows CUDA zip
        for asset in data.get("assets", []):
            name = asset["name"].lower()
            if "win" in name and "cuda" in name and name.endswith(".zip"):
                return asset["browser_download_url"]

        return None

    async def download_and_install(self) -> bool:
        """Download and extract llama-server binary."""
        try:
            self._download_status = "downloading"
            self._download_progress = 0.0

            url = await self.get_latest_release_url()
            if not url:
                self._download_status = "error"
                return False

            zip_path = BIN_DIR / "llama-server-cuda.zip"
            BIN_DIR.mkdir(parents=True, exist_ok=True)

            # Download with progress
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    total = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    with open(zip_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                self._download_progress = downloaded / total

            # Extract
            self._download_status = "extracting"
            self._download_progress = 0.0

            with zipfile.ZipFile(zip_path, "r") as zf:
                # Find llama-server executable inside zip
                members = zf.namelist()
                server_files = [m for m in members if m.endswith(BINARY_NAME)]

                if not server_files:
                    # Extract everything, server might be at top level
                    zf.extractall(BIN_DIR)
                else:
                    # Extract server and related DLLs
                    for member in members:
                        basename = os.path.basename(member)
                        if basename and (basename.endswith(".exe") or basename.endswith(".dll")):
                            # Extract flat into BIN_DIR
                            data = zf.read(member)
                            (BIN_DIR / basename).write_bytes(data)

            # Clean up zip
            zip_path.unlink(missing_ok=True)

            # Verify binary exists
            if not self.binary_path.exists():
                # Check subdirectories
                for f in BIN_DIR.rglob(BINARY_NAME):
                    self.binary_path = f
                    break

            self._download_status = "ready" if self.is_available else "error"
            self._download_progress = 1.0
            return self.is_available

        except Exception as e:
            self._download_status = "error"
            print(f"Binary download error: {e}")
            return False

    def get_status(self) -> dict:
        return {
            "available": self.is_available,
            "path": str(self.binary_path),
            "status": self._download_status,
            "progress": self._download_progress,
        }

# Singleton
binary_manager = BinaryManager()
```

**Step 2: Add binary routes to system router**

Add to `app/routers/system.py`:
```python
from app.services.binary_manager import binary_manager

@router.get("/binary/status")
async def binary_status():
    return binary_manager.get_status()

@router.post("/binary/download")
async def download_binary():
    success = await binary_manager.download_and_install()
    return {"success": success, **binary_manager.get_status()}
```

**Step 3: Commit**
```
git add -A && git commit -m "feat: auto-download llama-server CUDA binary from GitHub releases"
```

---

## Task 4: Model Scanner & Registry

**Files:**
- Create: `app/services/model_scanner.py`
- Create: `app/services/model_manager.py`
- Create: `app/routers/models.py`
- Modify: `app/main.py`

**Step 1: Create GGUF metadata reader and model scanner**

```python
# app/services/model_scanner.py
import struct
from pathlib import Path
from dataclasses import dataclass, field

GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian

@dataclass
class GGUFMetadata:
    file_path: str
    file_size_mb: int
    model_name: str = ""
    architecture: str = ""
    parameter_count: int = 0
    quantization: str = ""
    context_length: int = 0
    embedding_length: int = 0
    num_layers: int = 0
    vocab_size: int = 0

def read_gguf_string(f) -> str:
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")

def read_gguf_value(f, vtype: int):
    """Read a GGUF metadata value by type."""
    if vtype == 0:  # uint8
        return struct.unpack("<B", f.read(1))[0]
    elif vtype == 1:  # int8
        return struct.unpack("<b", f.read(1))[0]
    elif vtype == 2:  # uint16
        return struct.unpack("<H", f.read(2))[0]
    elif vtype == 3:  # int16
        return struct.unpack("<h", f.read(2))[0]
    elif vtype == 4:  # uint32
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == 5:  # int32
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == 6:  # float32
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == 7:  # bool
        return struct.unpack("<?", f.read(1))[0]
    elif vtype == 8:  # string
        return read_gguf_string(f)
    elif vtype == 9:  # array
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        return [read_gguf_value(f, atype) for _ in range(min(alen, 100))]
    elif vtype == 10:  # uint64
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == 11:  # int64
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == 12:  # float64
        return struct.unpack("<d", f.read(8))[0]
    return None

def read_gguf_metadata(file_path: str) -> GGUFMetadata | None:
    """Read metadata from a GGUF file header."""
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".gguf":
        return None

    meta = GGUFMetadata(
        file_path=str(path),
        file_size_mb=path.stat().st_size // (1024 * 1024),
    )

    try:
        with open(path, "rb") as f:
            # Read header
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                return None

            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            # Read metadata key-value pairs
            for _ in range(metadata_kv_count):
                key = read_gguf_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                value = read_gguf_value(f, vtype)

                if key == "general.name":
                    meta.model_name = value
                elif key == "general.architecture":
                    meta.architecture = value
                elif key == "general.quantization_version":
                    meta.quantization = str(value)
                elif key.endswith(".context_length"):
                    meta.context_length = value
                elif key.endswith(".embedding_length"):
                    meta.embedding_length = value
                elif key.endswith(".block_count"):
                    meta.num_layers = value
                elif key == "general.file_type":
                    # Map file type int to quant name
                    quant_map = {
                        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
                        7: "Q8_0", 8: "Q8_1", 10: "Q2_K", 11: "Q3_K_S",
                        12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
                        15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
                        18: "Q6_K", 26: "IQ2_XXS", 27: "IQ2_XS",
                    }
                    meta.quantization = quant_map.get(value, f"type_{value}")

    except Exception as e:
        print(f"Error reading GGUF: {e}")

    # Infer quant from filename if not found in metadata
    if not meta.quantization:
        name_lower = path.name.lower()
        for q in ["q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s",
                   "q3_k_m", "q3_k_l", "q3_k_s", "q2_k", "iq2_xxs", "iq2_xs"]:
            if q in name_lower:
                meta.quantization = q.upper()
                break

    if not meta.model_name:
        meta.model_name = path.stem

    return meta

def scan_models_directory(models_dir: str) -> list[GGUFMetadata]:
    """Recursively scan directory for GGUF files and read metadata."""
    results = []
    base = Path(models_dir)
    if not base.exists():
        return results

    for gguf_path in sorted(base.rglob("*.gguf")):
        # Skip mmproj files (vision projectors, not chat models)
        if "mmproj" in gguf_path.name.lower():
            continue
        meta = read_gguf_metadata(str(gguf_path))
        if meta:
            results.append(meta)

    return results
```

**Step 2: Create model manager**

```python
# app/services/model_manager.py
import asyncio
import json
import hashlib
import signal
from pathlib import Path
from dataclasses import dataclass, asdict
from app.config import APP_DIR, load_config
from app.services.system_monitor import system_monitor
from app.services.binary_manager import binary_manager

@dataclass
class ModelConfig:
    id: str
    name: str
    path: str
    gpu_layers: int = -1  # -1 = auto
    context_default: int = 4096
    context_recommended: int = 8192
    context_max: int = 32768
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1

class ModelManager:
    def __init__(self):
        self._process: asyncio.subprocess.Process | None = None
        self._current_model: ModelConfig | None = None
        self._loading = False
        self._load_progress = ""
        self._server_port = 8081

    @property
    def current_model(self) -> ModelConfig | None:
        return self._current_model

    @property
    def is_loaded(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_progress(self) -> str:
        return self._load_progress

    @property
    def server_url(self) -> str:
        return f"http://127.0.0.1:{self._server_port}"

    def _models_registry_path(self) -> Path:
        return APP_DIR / "models.json"

    def load_registry(self) -> list[ModelConfig]:
        path = self._models_registry_path()
        if not path.exists():
            return []
        with open(path) as f:
            data = json.load(f)
        return [ModelConfig(**m) for m in data.get("models", [])]

    def save_registry(self, models: list[ModelConfig]):
        path = self._models_registry_path()
        data = {"models": [asdict(m) for m in models]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def add_model(self, name: str, path: str, **kwargs) -> ModelConfig:
        models = self.load_registry()
        model_id = hashlib.md5(path.encode()).hexdigest()[:12]

        # Check if already registered
        for m in models:
            if m.path == path:
                # Update existing
                m.name = name
                for k, v in kwargs.items():
                    if hasattr(m, k):
                        setattr(m, k, v)
                self.save_registry(models)
                return m

        config = ModelConfig(id=model_id, name=name, path=path, **kwargs)
        models.append(config)
        self.save_registry(models)
        return config

    def remove_model(self, model_id: str):
        models = self.load_registry()
        models = [m for m in models if m.id != model_id]
        self.save_registry(models)

    def update_model(self, model_id: str, **kwargs) -> ModelConfig | None:
        models = self.load_registry()
        for m in models:
            if m.id == model_id:
                for k, v in kwargs.items():
                    if hasattr(m, k):
                        setattr(m, k, v)
                self.save_registry(models)
                return m
        return None

    def estimate_resources(self, model_path: str, gpu_layers: int, context_length: int) -> dict:
        """Estimate VRAM and RAM usage for a model configuration."""
        from app.services.model_scanner import read_gguf_metadata
        meta = read_gguf_metadata(model_path)
        if not meta:
            return {"error": "Cannot read model metadata"}

        file_size_mb = meta.file_size_mb
        num_layers = meta.num_layers or 32  # fallback

        if gpu_layers < 0:
            gpu_layers = num_layers + 1  # all layers + output

        layer_size_mb = file_size_mb / (num_layers + 1)
        effective_gpu_layers = min(gpu_layers, num_layers + 1)

        # Context memory estimate: ~2 bytes per element, embedding * context * 2 (kv cache)
        embedding = meta.embedding_length or 4096
        ctx_memory_mb = (embedding * context_length * 2 * 2) / (1024 * 1024)

        vram_needed = (effective_gpu_layers * layer_size_mb) + (ctx_memory_mb if effective_gpu_layers > 0 else 0)
        ram_needed = ((num_layers + 1 - effective_gpu_layers) * layer_size_mb) + (ctx_memory_mb if effective_gpu_layers == 0 else 0)

        resources = system_monitor.get_resources()
        config = load_config()

        vram_available = resources.gpu.vram_free_mb if resources.gpu else 0
        vram_limit = system_monitor.get_vram_limit_mb(config.get("vram_limit_percent", 95))
        ram_available = resources.ram_free_mb
        ram_limit = system_monitor.get_ram_limit_mb(config.get("ram_limit_percent", 85))

        return {
            "vram_needed_mb": round(vram_needed),
            "ram_needed_mb": round(ram_needed),
            "vram_available_mb": vram_available,
            "vram_limit_mb": vram_limit,
            "ram_available_mb": ram_available,
            "ram_limit_mb": ram_limit,
            "gpu_layers": effective_gpu_layers,
            "total_layers": num_layers + 1,
            "context_memory_mb": round(ctx_memory_mb),
            "fits_vram": vram_needed <= vram_available,
            "fits_ram": ram_needed <= ram_available,
            "feasible": vram_needed <= vram_limit and ram_needed <= ram_limit,
        }

    def recommend_gpu_layers(self, model_path: str) -> int:
        """Auto-calculate optimal GPU layers for available VRAM."""
        from app.services.model_scanner import read_gguf_metadata
        meta = read_gguf_metadata(model_path)
        if not meta:
            return 0

        resources = system_monitor.get_resources()
        if not resources.gpu:
            return 0

        config = load_config()
        vram_limit = system_monitor.get_vram_limit_mb(config.get("vram_limit_percent", 95))
        vram_available = min(resources.gpu.vram_free_mb, vram_limit)

        num_layers = meta.num_layers or 32
        total_layers = num_layers + 1
        layer_size_mb = meta.file_size_mb / total_layers

        # Reserve some VRAM for context
        vram_for_layers = vram_available * 0.85  # 85% for layers, 15% for context

        max_layers = int(vram_for_layers / layer_size_mb)
        return min(max_layers, total_layers)

    async def load_model(self, model_id: str, context_length: int | None = None) -> bool:
        """Start llama-server with the specified model."""
        if self._loading:
            return False

        # Unload current model first
        await self.unload_model()

        models = self.load_registry()
        model = next((m for m in models if m.id == model_id), None)
        if not model:
            return False

        if not binary_manager.is_available:
            return False

        self._loading = True
        self._load_progress = "Starting llama-server..."

        try:
            ctx = context_length or model.context_default
            gpu_layers = model.gpu_layers

            if gpu_layers < 0:
                gpu_layers = self.recommend_gpu_layers(model.path)

            # Check feasibility
            estimate = self.estimate_resources(model.path, gpu_layers, ctx)
            if not estimate.get("feasible", False):
                self._load_progress = f"Cannot load: needs {estimate.get('vram_needed_mb', '?')}MB VRAM + {estimate.get('ram_needed_mb', '?')}MB RAM"
                self._loading = False
                return False

            cmd = [
                str(binary_manager.binary_path),
                "-m", model.path,
                "-c", str(ctx),
                "-ngl", str(gpu_layers),
                "--host", "127.0.0.1",
                "--port", str(self._server_port),
                "-fa",  # flash attention
            ]

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            self._current_model = model
            self._load_progress = "Loading model layers..."

            # Wait for server to be ready (poll health endpoint)
            import aiohttp
            for i in range(120):  # 2 min timeout
                await asyncio.sleep(1)
                if self._process.returncode is not None:
                    stderr = await self._process.stderr.read()
                    self._load_progress = f"Server crashed: {stderr.decode()[:200]}"
                    self._loading = False
                    self._process = None
                    self._current_model = None
                    return False
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.server_url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                            data = await resp.json()
                            if data.get("status") == "ok":
                                self._load_progress = "Ready"
                                self._loading = False
                                return True
                            elif data.get("status") == "loading model":
                                progress = data.get("progress", 0)
                                self._load_progress = f"Loading model... {progress:.0%}"
                except Exception:
                    self._load_progress = f"Waiting for server... ({i+1}s)"

            self._load_progress = "Timeout waiting for server"
            self._loading = False
            await self.unload_model()
            return False

        except Exception as e:
            self._load_progress = f"Error: {str(e)}"
            self._loading = False
            return False

    async def unload_model(self):
        """Stop the current llama-server process."""
        if self._process:
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except Exception:
                pass
            self._process = None
        self._current_model = None
        self._load_progress = ""

    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "loading": self.is_loading,
            "load_progress": self._load_progress,
            "current_model": asdict(self._current_model) if self._current_model else None,
            "server_url": self.server_url if self.is_loaded else None,
        }

# Singleton
model_manager = ModelManager()
```

**Step 3: Create models router**

```python
# app/routers/models.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.model_scanner import scan_models_directory, read_gguf_metadata
from app.services.model_manager import model_manager
from app.config import load_config
import dataclasses

router = APIRouter(prefix="/api/models", tags=["models"])

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
    name: str | None = None
    gpu_layers: int | None = None
    context_default: int | None = None
    context_recommended: int | None = None
    context_max: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    repeat_penalty: float | None = None

class LoadModelRequest(BaseModel):
    context_length: int | None = None

@router.get("/scan")
async def scan_models():
    config = load_config()
    models_dir = config.get("models_directory", "")
    models = scan_models_directory(models_dir)
    return {"models": [dataclasses.asdict(m) for m in models]}

@router.get("/scan/{path:path}")
async def scan_model_file(path: str):
    meta = read_gguf_metadata(path)
    if not meta:
        raise HTTPException(404, "Could not read GGUF file")
    return dataclasses.asdict(meta)

@router.get("/registry")
async def list_registered():
    models = model_manager.load_registry()
    return {"models": [dataclasses.asdict(m) for m in models]}

@router.post("/registry")
async def add_model(req: AddModelRequest):
    model = model_manager.add_model(**req.model_dump())
    return dataclasses.asdict(model)

@router.put("/registry/{model_id}")
async def update_model(model_id: str, req: UpdateModelRequest):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    model = model_manager.update_model(model_id, **updates)
    if not model:
        raise HTTPException(404, "Model not found")
    return dataclasses.asdict(model)

@router.delete("/registry/{model_id}")
async def remove_model(model_id: str):
    model_manager.remove_model(model_id)
    return {"ok": True}

@router.post("/load/{model_id}")
async def load_model(model_id: str, req: LoadModelRequest = LoadModelRequest()):
    success = await model_manager.load_model(model_id, req.context_length)
    return {"success": success, **model_manager.get_status()}

@router.post("/unload")
async def unload_model():
    await model_manager.unload_model()
    return {"success": True}

@router.get("/status")
async def model_status():
    return model_manager.get_status()

@router.get("/estimate/{model_id}")
async def estimate_resources(model_id: str, gpu_layers: int = -1, context_length: int = 4096):
    models = model_manager.load_registry()
    model = next((m for m in models if m.id == model_id), None)
    if not model:
        raise HTTPException(404, "Model not found")
    return model_manager.estimate_resources(model.path, gpu_layers, context_length)

@router.get("/recommend-layers")
async def recommend_layers(model_path: str):
    layers = model_manager.recommend_gpu_layers(model_path)
    return {"recommended_gpu_layers": layers}
```

**Step 4: Register router in main.py**

**Step 5: Commit**
```
git add -A && git commit -m "feat: model scanner, GGUF metadata reader, model manager with GPU layer auto-detection"
```

---

## Task 5: Session Manager

**Files:**
- Create: `app/services/session_manager.py`
- Create: `app/routers/sessions.py`
- Modify: `app/main.py`

**Step 1: Create session manager with SQLite**

```python
# app/services/session_manager.py
import aiosqlite
import json
import time
import uuid
from pathlib import Path
from app.config import SESSIONS_DIR

DB_PATH = SESSIONS_DIR / "sessions.db"

class SessionManager:
    async def init_db(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
            await db.commit()

    async def create_session(self, title: str = "New Chat", model_id: str = None) -> dict:
        session_id = str(uuid.uuid4())[:8]
        now = time.time()
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO sessions (id, title, model_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, title, model_id, now, now)
            )
            await db.commit()
        return {"id": session_id, "title": title, "model_id": model_id, "created_at": now, "updated_at": now}

    async def list_sessions(self) -> list[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_session(self, session_id: str) -> dict | None:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def update_session(self, session_id: str, title: str = None, model_id: str = None):
        updates = []
        params = []
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if model_id is not None:
            updates.append("model_id = ?")
            params.append(model_id)
        updates.append("updated_at = ?")
        params.append(time.time())
        params.append(session_id)

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?", params)
            await db.commit()

    async def delete_session(self, session_id: str):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await db.commit()

    async def add_message(self, session_id: str, role: str, content: str, token_count: int = 0) -> dict:
        now = time.time()
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, token_count) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, now, token_count)
            )
            await db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            await db.commit()
            return {"id": cursor.lastrowid, "session_id": session_id, "role": role, "content": content, "timestamp": now}

    async def get_messages(self, session_id: str) -> list[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,)
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

# Singleton
session_manager = SessionManager()
```

**Step 2: Create sessions router**

```python
# app/routers/sessions.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.session_manager import session_manager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

class CreateSessionRequest(BaseModel):
    title: str = "New Chat"
    model_id: str | None = None

class UpdateSessionRequest(BaseModel):
    title: str | None = None
    model_id: str | None = None

@router.get("/")
async def list_sessions():
    sessions = await session_manager.list_sessions()
    return {"sessions": sessions}

@router.post("/")
async def create_session(req: CreateSessionRequest):
    return await session_manager.create_session(req.title, req.model_id)

@router.get("/{session_id}")
async def get_session(session_id: str):
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session

@router.put("/{session_id}")
async def update_session(session_id: str, req: UpdateSessionRequest):
    await session_manager.update_session(session_id, req.title, req.model_id)
    return {"ok": True}

@router.delete("/{session_id}")
async def delete_session(session_id: str):
    await session_manager.delete_session(session_id)
    return {"ok": True}

@router.get("/{session_id}/messages")
async def get_messages(session_id: str):
    messages = await session_manager.get_messages(session_id)
    return {"messages": messages}
```

**Step 3: Register router + init DB on startup in main.py**

**Step 4: Commit**
```
git add -A && git commit -m "feat: session manager with SQLite persistence"
```

---

## Task 6: Chat Backend with WebSocket Streaming

**Files:**
- Create: `app/services/prompt_composer.py`
- Create: `app/routers/chat.py`
- Modify: `app/main.py`

**Step 1: Create prompt composer**

```python
# app/services/prompt_composer.py
from pathlib import Path
from app.config import APP_DIR

class PromptComposer:
    def compose_system_prompt(self) -> str:
        parts = []
        for filename in ["system-prompt.md", "identity.md", "user.md", "memory.md"]:
            path = APP_DIR / filename
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(content)
        return "\n\n---\n\n".join(parts)

    def get_file(self, filename: str) -> str:
        path = APP_DIR / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def save_file(self, filename: str, content: str):
        allowed = ["system-prompt.md", "identity.md", "user.md", "memory.md"]
        if filename not in allowed:
            raise ValueError(f"Cannot write to {filename}")
        path = APP_DIR / filename
        path.write_text(content, encoding="utf-8")

prompt_composer = PromptComposer()
```

**Step 2: Create chat WebSocket handler**

```python
# app/routers/chat.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.model_manager import model_manager
from app.services.session_manager import session_manager
from app.services.prompt_composer import prompt_composer
import aiohttp
import json

router = APIRouter(tags=["chat"])

@router.websocket("/ws/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("content", "")
            if not user_message:
                continue

            if not model_manager.is_loaded:
                await websocket.send_json({"type": "error", "content": "No model loaded"})
                continue

            # Save user message
            await session_manager.add_message(session_id, "user", user_message)

            # Build messages array
            messages = await session_manager.get_messages(session_id)
            system_prompt = prompt_composer.compose_system_prompt()

            chat_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

            # Get model params
            model = model_manager.current_model
            payload = {
                "model": "local",
                "messages": chat_messages,
                "stream": True,
                "temperature": model.temperature if model else 0.7,
                "top_p": model.top_p if model else 0.9,
                "repeat_penalty": model.repeat_penalty if model else 1.1,
            }

            # Stream from llama-server
            full_response = ""
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{model_manager.server_url}/v1/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        await websocket.send_json({"type": "start"})

                        async for line in resp.content:
                            line = line.decode("utf-8").strip()
                            if not line or not line.startswith("data: "):
                                continue
                            line = line[6:]  # Remove "data: "
                            if line == "[DONE]":
                                break

                            try:
                                chunk = json.loads(line)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    full_response += token
                                    await websocket.send_json({"type": "token", "content": token})
                            except json.JSONDecodeError:
                                continue

                        await websocket.send_json({"type": "done", "content": full_response})

            except Exception as e:
                await websocket.send_json({"type": "error", "content": f"Stream error: {str(e)}"})
                full_response = f"[Error: {str(e)}]"

            # Save assistant response
            if full_response:
                await session_manager.add_message(session_id, "assistant", full_response)

    except WebSocketDisconnect:
        pass
```

**Step 3: Register router in main.py**

**Step 4: Commit**
```
git add -A && git commit -m "feat: chat WebSocket streaming with prompt composition"
```

---

## Task 7: Prompt Files & Config Router

**Files:**
- Create: `app/routers/config.py`
- Modify: `app/main.py`

**Step 1: Create config/prompt router**

```python
# app/routers/config.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.config import load_config, save_config
from app.services.prompt_composer import prompt_composer

router = APIRouter(prefix="/api/config", tags=["config"])

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
    return load_config()

@router.put("/")
async def update_config(req: ConfigUpdateRequest):
    config = load_config()
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    config.update(updates)
    save_config(config)
    return config

@router.get("/prompt/{filename}")
async def get_prompt_file(filename: str):
    allowed = ["system-prompt.md", "identity.md", "user.md", "memory.md"]
    if filename not in allowed:
        raise HTTPException(400, f"Invalid file: {filename}")
    content = prompt_composer.get_file(filename)
    return {"filename": filename, "content": content}

@router.put("/prompt/{filename}")
async def save_prompt_file(filename: str, req: PromptFileRequest):
    try:
        prompt_composer.save_file(filename, req.content)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(400, str(e))
```

**Step 2: Register router**

**Step 3: Commit**
```
git add -A && git commit -m "feat: config and prompt file management endpoints"
```

---

## Task 8: Web UI — HTML/CSS Shell

**Files:**
- Create: `static/index.html`
- Create: `static/css/style.css`
- Create: `static/js/app.js`
- Create: `static/js/api.js`
- Create: `static/js/chat.js`
- Create: `static/js/models.js`
- Create: `static/js/resources.js`
- Modify: `app/main.py` (serve index.html at /)

This is the largest task. The UI includes:
- Sidebar: sessions list, model picker, resource monitors (CPU/GPU/RAM)
- Main area: chat with streaming, markdown rendering
- Model picker modal: browse GGUFs, config sliders, resource estimate, load/unload
- Settings modal: edit system prompt, identity, user, memory files
- Model loading progress bar
- Default model + load-on-start toggle in settings
- Hot-swap models (unload current, load new)

Full implementation in code — vanilla HTML/CSS/JS, dark theme, no dependencies except a markdown renderer (marked.js from CDN).

**Step 1-8:** Create all static files (detailed code in implementation)

**Step 9: Update main.py to serve index.html at root**

```python
from fastapi.responses import FileResponse

@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))
```

**Step 10: Commit**
```
git add -A && git commit -m "feat: web UI with chat, model picker, resource monitors, and settings"
```

---

## Task 9: Startup Flow & Default Model

**Files:**
- Modify: `app/main.py`

**Step 1: Add startup event**

On app startup:
1. Ensure dirs exist
2. Init session DB
3. Check if llama-server binary exists, if not — UI shows download prompt
4. If `load_model_on_start` is true and `default_model_id` is set, auto-load model

```python
@app.on_event("startup")
async def startup():
    from app.services.session_manager import session_manager
    await session_manager.init_db()

    config = load_config()
    if config.get("load_model_on_start") and config.get("default_model_id"):
        from app.services.model_manager import model_manager
        from app.services.binary_manager import binary_manager
        if binary_manager.is_available:
            asyncio.create_task(model_manager.load_model(config["default_model_id"]))

@app.on_event("shutdown")
async def shutdown():
    from app.services.model_manager import model_manager
    await model_manager.unload_model()
```

**Step 2: Commit**
```
git add -A && git commit -m "feat: startup flow with auto-load default model and graceful shutdown"
```

---

## Task 10: Integration Test & First Push

**Step 1: Manual integration test**
1. `python run.py`
2. Open http://127.0.0.1:8080
3. Verify: resource monitors showing CPU/GPU/RAM
4. Download llama-server binary via UI
5. Scan models, pick one, configure GPU layers
6. Load model, verify progress bar
7. Create chat session, send message, verify streaming
8. Reload page, verify session persistence

**Step 2: Push to GitHub**
```
git push -u origin main
```

---

Plan complete and saved to `docs/plans/2026-03-10-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?
