"""Model manager: registry, resource estimation, and llama-server lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import psutil

from app.config import APP_DIR, MODELS_DIR, load_config
from app.services.binary_manager import BinaryManager, BinaryVariant
from app.services.model_scanner import (
    infer_capabilities,
    infer_quant_from_filename,
    is_onebit_quant,
    read_gguf_metadata,
)
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
    notes: str = ""
    cache_type_k: str = "f16"     # KV cache key type: f16, q8_0, q4_0, tbq3_0, tbq4_0
    cache_type_v: str = "f16"     # KV cache value type: f16, q8_0, q4_0, tbq3_0, tbq4_0
    speculative: str = "none"     # none, ngram
    capabilities: list[str] = field(default_factory=list)


class ModelManager:
    """Manages one llama-server process at a time, plus a JSON-based model registry."""

    _instance: ModelManager | None = None

    def __init__(self, app_dir: Path | None = None) -> None:
        self._app_dir = app_dir or APP_DIR
        self._registry_path = self._app_dir / "models.json"
        self._server_port = 8081
        self._process: subprocess.Popen | None = None
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
            # Filter to only known fields to handle schema changes gracefully
            known_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
            results = []
            dirty = False
            for item in data:
                filtered = {k: v for k, v in item.items() if k in known_fields}
                try:
                    model = ModelConfig(**filtered)
                    # Backfill capabilities if empty
                    if not model.capabilities:
                        name_for_caps = f"{model.name} {Path(model.path).name} {Path(model.path).parent.name}"
                        model.capabilities = infer_capabilities(name_for_caps)
                        if model.capabilities:
                            dirty = True
                    results.append(model)
                except (TypeError, ValueError) as exc:
                    log.warning("Skipping invalid registry entry %s: %s", item.get("id", "?"), exc)
            # Persist backfilled capabilities
            if dirty:
                self.save_registry(results)
            return results
        except (json.JSONDecodeError, OSError) as exc:
            log.error("Failed to load registry: %s", exc)
            return []

    def save_registry(self, models: list[ModelConfig]) -> None:
        """Save models to the JSON registry file atomically."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(m) for m in models]
        content = json.dumps(data, indent=2)
        # Write to temp file then rename for atomicity
        tmp_path = self._registry_path.with_suffix(".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(self._registry_path)

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

    # -- Launch helpers ------------------------------------------------------

    def _select_binary_variant(self, model_path: str) -> BinaryVariant:
        """Determine which binary variant to use.

        Logic:
        - 1-bit models REQUIRE the custom binary
        - If a custom binary is registered, prefer it for all models
          (it includes all upstream features + optimizations)
        - Otherwise fall back to primary
        """
        bm = BinaryManager.instance()

        # 1-bit models must use custom
        meta = read_gguf_metadata(Path(model_path))
        quant = meta.quantization if meta else infer_quant_from_filename(Path(model_path).name)
        if is_onebit_quant(quant):
            return BinaryVariant.CUSTOM

        # Prefer custom for all models when available (it's a superset)
        if bm.is_variant_available(BinaryVariant.CUSTOM):
            return BinaryVariant.CUSTOM

        return BinaryVariant.PRIMARY

    def _build_launch_cmd(
        self,
        model: ModelConfig,
        binary_path: Path,
        gpu_layers: int,
        context_length: int,
    ) -> list[str]:
        """Build the llama-server command with all optimized flags."""
        # Smart thread count: 1 when fully offloaded, multi otherwise
        fully_offloaded = gpu_layers >= 99
        threads = 1 if fully_offloaded else max(1, (psutil.cpu_count(logical=False) or 2) - 1)

        cmd = [
            str(binary_path),
            "-m", model.path,
            "-c", str(context_length),
            "-ngl", str(gpu_layers),
            "--host", "127.0.0.1",
            "--port", str(self._server_port),
            "-fa", "on",
            "--mlock",
            "-t", str(threads),
            "--batch-size", "512",
            "--ubatch-size", "512",
        ]

        # KV cache quantization
        if model.cache_type_k != "f16":
            cmd.extend(["--cache-type-k", model.cache_type_k])
        if model.cache_type_v != "f16":
            cmd.extend(["--cache-type-v", model.cache_type_v])

        # Speculative decoding
        if model.speculative == "ngram":
            cmd.extend([
                "--spec-type", "ngram-mod",
                "--spec-ngram-size-n", "12",
                "--spec-ngram-size-m", "48",
                "--draft-max", "16",
            ])

        # Windows-specific
        if platform.system() == "Windows":
            cmd.append("--no-mmap")

        return cmd

    def _build_launch_env(self) -> dict[str, str]:
        """Build environment variables for the llama-server process."""
        env = os.environ.copy()
        env["GGML_CUDA_GRAPH_OPT"] = "1"
        env["CUDA_SCALE_LAUNCH_QUEUES"] = "4x"
        return env

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

        # Select binary variant based on model quantization
        bm = BinaryManager.instance()
        variant = self._select_binary_variant(model.path)

        if not bm.is_variant_available(variant):
            variant_label = "custom (1-bit fork)" if variant == BinaryVariant.CUSTOM else "primary"
            self._last_error = (
                f"llama-server binary not available for {variant_label} variant. "
                + ("Register the custom fork binary via /api/system/binary/register." if variant == BinaryVariant.CUSTOM else "Download the primary binary first.")
            )
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

            # Build optimized command and environment
            binary_path = bm.binary_path_for(variant)
            cmd = self._build_launch_cmd(model, binary_path, gpu_layers, ctx)
            env = self._build_launch_env()

            log.info("Starting llama-server (%s): %s", variant.value, " ".join(cmd))

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )

            # Elevate process priority for lower latency
            try:
                import psutil as _psutil
                p = _psutil.Process(self._process.pid)
                if platform.system() == "Windows":
                    p.nice(_psutil.HIGH_PRIORITY_CLASS)
                else:
                    p.nice(-10)
            except Exception:
                pass  # non-critical

            # Poll health endpoint for readiness (2 min timeout)
            health_url = f"http://127.0.0.1:{self._server_port}/health"
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 120

            while loop.time() < deadline:
                await asyncio.sleep(1)

                # Check if process died
                ret = self._process.poll()
                if ret is not None:
                    stderr_out = ""
                    try:
                        raw = self._process.stderr.read()
                        stderr_out = raw.decode("utf-8", errors="replace")[-500:]
                    except Exception:
                        pass
                    self._last_error = f"llama-server exited with code {ret}"
                    if stderr_out:
                        self._last_error += f": {stderr_out.strip()}"
                    log.error(self._last_error)
                    self._loading = False
                    self._process = None
                    return False

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                            data = await resp.json()
                            if resp.status == 200:
                                status = data.get("status", "")
                                if status == "ok":
                                    self._current_model_id = model_id
                                    self._loading = False
                                    self._load_progress = 1.0
                                    log.info("Model %s loaded successfully", model.name)
                                    return True
                                elif "progress" in data:
                                    self._load_progress = data["progress"]
                            elif resp.status == 503:
                                # Server is up but still loading
                                progress = data.get("progress", 0)
                                if progress > 0:
                                    self._load_progress = progress
                                else:
                                    # No progress info — use time-based estimate
                                    elapsed = loop.time() - (deadline - 120)
                                    # Asymptotic: approach 0.9 over 120s
                                    self._load_progress = min(0.9, elapsed / 130)
                except Exception:
                    pass

            # Timeout
            self._last_error = "Timed out waiting for llama-server to become ready"
            log.error(self._last_error)
            await self.unload_model()
            self._loading = False
            return False

        except Exception as exc:
            import traceback
            self._last_error = str(exc) or repr(exc)
            log.error("Failed to load model: %s\n%s", exc, traceback.format_exc())
            self._loading = False
            return False

    async def unload_model(self) -> None:
        """Terminate the current llama-server process gracefully."""
        if self._process is not None:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log.warning("llama-server did not exit gracefully, killing")
                    self._process.kill()
                    self._process.wait(timeout=5)
            except ProcessLookupError:
                pass
            except Exception as exc:
                log.error("Error unloading model: %s", exc)
            self._process = None
        else:
            # No tracked process — kill any llama-server listening on our port
            self._kill_server_on_port()

        self._current_model_id = None
        self._load_progress = 0.0
        self._loading = False
        self._last_error = None
        log.info("Model unloaded")

    def _kill_server_on_port(self) -> None:
        """Kill any process listening on our server port (handles orphaned servers)."""
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if conn.laddr.port == self._server_port and conn.status == "LISTEN":
                    proc = psutil.Process(conn.pid)
                    log.info("Killing orphaned llama-server (PID %d)", conn.pid)
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError) as exc:
            log.warning("Could not kill server on port %d: %s", self._server_port, exc)

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

    async def recover_state(self) -> None:
        """Detect if llama-server is running on our port (e.g. after app restart).

        Probes the health endpoint and, if a healthy server is found,
        recovers _current_model_id from its /v1/models response.
        """
        if self._current_model_id or self._loading:
            return  # Already tracking a model

        import aiohttp
        health_url = f"http://127.0.0.1:{self._server_port}/health"
        models_url = f"http://127.0.0.1:{self._server_port}/v1/models"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status != 200:
                        return
                    data = await resp.json()
                    if data.get("status") != "ok":
                        return

                # Server is healthy — figure out which model
                async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        models_data = await resp.json()
                        model_id_from_server = None
                        if models_data.get("data"):
                            server_model_name = models_data["data"][0].get("id", "")
                            # Match against registry by filename
                            for m in self.load_registry():
                                if Path(m.path).name.lower() in server_model_name.lower():
                                    model_id_from_server = m.id
                                    break
                            # If no exact match, try the first registry model (user probably loaded it)
                            if not model_id_from_server:
                                registry = self.load_registry()
                                if registry:
                                    # Use file stem matching
                                    for m in registry:
                                        if Path(m.path).stem.lower() in server_model_name.lower():
                                            model_id_from_server = m.id
                                            break

                        if model_id_from_server:
                            self._current_model_id = model_id_from_server
                            self._load_progress = 1.0
                            self._loading = False
                            log.info("Recovered running model: %s", model_id_from_server)
                        else:
                            # Server running but can't identify model — still mark as connected
                            # Use first registry model as best guess
                            registry = self.load_registry()
                            if registry:
                                self._current_model_id = registry[0].id
                                self._load_progress = 1.0
                                self._loading = False
                                log.info("Recovered running server, assuming model: %s", registry[0].name)
        except Exception:
            pass  # Server not running, that's fine
