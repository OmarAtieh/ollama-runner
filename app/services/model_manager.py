"""Model manager: registry, resource estimation, and llama-server lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import psutil

from app.config import APP_DIR, MODELS_DIR, load_config
from app.services.binary_manager import BinaryManager
from app.services.model_scanner import read_gguf_metadata
from app.services.system_monitor import SystemMonitor

log = logging.getLogger(__name__)


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
    """Manages one llama-server process at a time, plus a JSON-based model registry."""

    _instance: ModelManager | None = None

    def __init__(self, app_dir: Path | None = None) -> None:
        self._app_dir = app_dir or APP_DIR
        self._registry_path = self._app_dir / "models.json"
        self._server_port = 8081
        self._process: asyncio.subprocess.Process | None = None
        self._current_model_id: str | None = None
        self._loading = False
        self._load_progress: float = 0.0
        self._last_error: str | None = None

    @classmethod
    def instance(cls) -> ModelManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- Registry ------------------------------------------------------------

    def load_registry(self) -> list[ModelConfig]:
        """Load models from the JSON registry file."""
        if not self._registry_path.exists():
            return []
        try:
            data = json.loads(self._registry_path.read_text(encoding="utf-8"))
            return [ModelConfig(**item) for item in data]
        except (json.JSONDecodeError, TypeError, OSError) as exc:
            log.error("Failed to load registry: %s", exc)
            return []

    def save_registry(self, models: list[ModelConfig]) -> None:
        """Save models to the JSON registry file."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(m) for m in models]
        self._registry_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def _generate_id(self, path: str) -> str:
        """Generate a stable ID from the model file path."""
        return hashlib.md5(path.encode("utf-8")).hexdigest()[:12]

    def add_model(self, name: str, path: str, **kwargs) -> ModelConfig:
        """Add or upsert a model in the registry."""
        model_id = self._generate_id(path)
        models = self.load_registry()

        # Upsert: replace existing with same ID
        models = [m for m in models if m.id != model_id]

        config = ModelConfig(id=model_id, name=name, path=path, **kwargs)
        models.append(config)
        self.save_registry(models)
        return config

    def remove_model(self, model_id: str) -> None:
        """Remove a model from the registry."""
        models = self.load_registry()
        models = [m for m in models if m.id != model_id]
        self.save_registry(models)

    def update_model(self, model_id: str, **kwargs) -> ModelConfig | None:
        """Update model config fields. Returns updated config or None."""
        models = self.load_registry()
        for i, m in enumerate(models):
            if m.id == model_id:
                d = asdict(m)
                d.update(kwargs)
                models[i] = ModelConfig(**d)
                self.save_registry(models)
                return models[i]
        return None

    def get_model(self, model_id: str) -> ModelConfig | None:
        """Get a single model by ID."""
        for m in self.load_registry():
            if m.id == model_id:
                return m
        return None

    # -- Resource estimation -------------------------------------------------

    def estimate_resources(
        self, model_path: str, gpu_layers: int, context_length: int
    ) -> dict:
        """Estimate VRAM and RAM needs for a model configuration."""
        meta = read_gguf_metadata(Path(model_path))
        monitor = SystemMonitor.instance()
        config = load_config()

        file_size_mb = 0.0
        num_layers = 0
        embedding_length = 0

        if meta:
            file_size_mb = meta.file_size_mb
            num_layers = meta.num_layers or 0
            embedding_length = meta.embedding_length or 0
        else:
            # Fallback: use file size
            p = Path(model_path)
            if p.exists():
                file_size_mb = p.stat().st_size / (1024 * 1024)

        total_layers = num_layers if num_layers > 0 else 1
        layer_size = file_size_mb / (total_layers + 1)
        vram_needed = gpu_layers * layer_size

        # Context memory: embedding * context * 2 (K+V) * 2 (bytes per fp16) / 1MB
        context_memory_mb = 0.0
        if embedding_length > 0:
            context_memory_mb = (
                embedding_length * context_length * 2 * 2 / (1024 * 1024)
            )

        vram_needed += context_memory_mb
        ram_needed = file_size_mb - (gpu_layers * layer_size) + context_memory_mb

        gpu = monitor.get_gpu_info()
        resources = monitor.get_resources()

        vram_available = gpu.vram_free_mb if gpu else 0
        vram_total = gpu.vram_total_mb if gpu else 0
        ram_available = resources.ram_free_mb

        vram_limit = monitor.get_vram_limit_mb(config.get("vram_limit_percent", 95))
        ram_limit = monitor.get_ram_limit_mb(config.get("ram_limit_percent", 85))

        fits_vram = vram_needed <= vram_available
        fits_ram = ram_needed <= ram_available
        feasible = fits_ram and (gpu_layers == 0 or fits_vram)

        return {
            "vram_needed_mb": round(vram_needed, 1),
            "ram_needed_mb": round(ram_needed, 1),
            "vram_available_mb": vram_available,
            "ram_available_mb": ram_available,
            "vram_limit_mb": vram_limit,
            "ram_limit_mb": ram_limit,
            "gpu_layers": gpu_layers,
            "total_layers": total_layers,
            "context_memory_mb": round(context_memory_mb, 1),
            "fits_vram": fits_vram,
            "fits_ram": fits_ram,
            "feasible": feasible,
        }

    def recommend_gpu_layers(self, model_path: str) -> int:
        """Auto-calculate GPU layers based on available VRAM (85% budget)."""
        meta = read_gguf_metadata(Path(model_path))
        if not meta or not meta.num_layers:
            return 0

        monitor = SystemMonitor.instance()
        gpu = monitor.get_gpu_info()
        if not gpu:
            return 0

        # Use 85% of available VRAM for layers, reserve 15% for KV cache
        vram_budget = gpu.vram_free_mb * 0.85
        total_layers = meta.num_layers
        layer_size = meta.file_size_mb / (total_layers + 1)

        if layer_size <= 0:
            return 0

        recommended = int(vram_budget / layer_size)
        # Clamp to total layers
        return min(recommended, total_layers)

    # -- Model loading / unloading -------------------------------------------

    async def load_model(self, model_id: str, context_length: int | None = None) -> bool:
        """Load a model by starting llama-server. Returns True on success."""
        self._last_error = None

        # Look up model in registry
        model = self.get_model(model_id)
        if model is None:
            self._last_error = f"Model {model_id} not found in registry"
            log.error(self._last_error)
            return False

        # Check binary
        bm = BinaryManager.instance()
        if not bm.is_available:
            self._last_error = "llama-server binary not available"
            log.error(self._last_error)
            return False

        # Unload current model first
        await self.unload_model()

        self._loading = True
        self._load_progress = 0.0

        try:
            ctx = context_length or model.context_default

            # GPU layers
            gpu_layers = model.gpu_layers
            if gpu_layers == -1:
                gpu_layers = self.recommend_gpu_layers(model.path)

            # Pre-load feasibility check
            estimate = self.estimate_resources(model.path, gpu_layers, ctx)
            if not estimate["feasible"]:
                self._last_error = (
                    f"Model does not fit in available resources. "
                    f"VRAM needed: {estimate['vram_needed_mb']:.0f}MB, "
                    f"available: {estimate['vram_available_mb']}MB. "
                    f"RAM needed: {estimate['ram_needed_mb']:.0f}MB, "
                    f"available: {estimate['ram_available_mb']}MB."
                )
                log.error(self._last_error)
                self._loading = False
                return False

            # Build command
            threads = max(1, (psutil.cpu_count(logical=False) or 2) - 1)
            cmd = [
                str(bm.binary_path),
                "-m", model.path,
                "-c", str(ctx),
                "-ngl", str(gpu_layers),
                "--host", "127.0.0.1",
                "--port", str(self._server_port),
                "-fa",
                "--mlock",
                "-t", str(threads),
                "--batch-size", "512",
                "--ubatch-size", "512",
            ]

            # Windows-specific: disable mmap (slower with CUDA on Windows)
            if platform.system() == "Windows":
                cmd.append("--no-mmap")

            log.info("Starting llama-server: %s", " ".join(cmd))

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Poll health endpoint for readiness (2 min timeout)
            health_url = f"http://127.0.0.1:{self._server_port}/health"
            deadline = asyncio.get_event_loop().time() + 120

            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(1)

                # Check if process died
                if self._process.returncode is not None:
                    self._last_error = f"llama-server exited with code {self._process.returncode}"
                    log.error(self._last_error)
                    self._loading = False
                    self._process = None
                    return False

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                status = data.get("status", "")
                                if status == "ok":
                                    self._current_model_id = model_id
                                    self._loading = False
                                    self._load_progress = 1.0
                                    log.info("Model %s loaded successfully", model.name)
                                    return True
                                elif "progress" in data:
                                    self._load_progress = data["progress"]
                except Exception:
                    pass

            # Timeout
            self._last_error = "Timed out waiting for llama-server to become ready"
            log.error(self._last_error)
            await self.unload_model()
            self._loading = False
            return False

        except Exception as exc:
            self._last_error = str(exc)
            log.error("Failed to load model: %s", exc)
            self._loading = False
            return False

    async def unload_model(self) -> None:
        """Terminate the current llama-server process gracefully."""
        if self._process is None:
            return

        try:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except asyncio.TimeoutError:
                log.warning("llama-server did not exit gracefully, killing")
                self._process.kill()
                await self._process.wait()
        except ProcessLookupError:
            pass
        except Exception as exc:
            log.error("Error unloading model: %s", exc)

        self._process = None
        self._current_model_id = None
        self._load_progress = 0.0
        self._loading = False
        log.info("Model unloaded")

    def get_status(self) -> dict:
        """Get current model loading/running status."""
        model = None
        if self._current_model_id:
            cfg = self.get_model(self._current_model_id)
            if cfg:
                model = asdict(cfg)

        return {
            "loaded": self._current_model_id is not None and not self._loading,
            "loading": self._loading,
            "load_progress": self._load_progress,
            "current_model": model,
            "server_url": f"http://127.0.0.1:{self._server_port}" if self._current_model_id else None,
            "error": self._last_error,
        }
