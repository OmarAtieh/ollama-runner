"""System resource monitor for GPU (NVIDIA), RAM, and CPU tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass

import psutil

# Prime the CPU percent counter so subsequent non-blocking calls return real values.
psutil.cpu_percent(interval=None)


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
    """Monitors system resources (CPU, RAM, GPU).

    GPU info is lazy-initialized and cached for 1 second.
    Use SystemMonitor.instance() for the singleton.
    """

    _instance: SystemMonitor | None = None

    def __init__(self) -> None:
        self._nvml_initialized = False
        self._nvml_available = False
        self._gpu_handle = None
        self._gpu_cache: GpuInfo | None = None
        self._gpu_cache_time: float = 0.0

    @classmethod
    def instance(cls) -> SystemMonitor:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- GPU (pynvml, lazy init) ------------------------------------------

    def _init_nvml(self) -> None:
        """Initialize pynvml once on first GPU query."""
        if self._nvml_initialized:
            return
        self._nvml_initialized = True
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
        except Exception:
            self._nvml_available = False

    def get_gpu_info(self) -> GpuInfo | None:
        """Read NVIDIA GPU info via pynvml. Cached for 1 second."""
        self._init_nvml()
        if not self._nvml_available:
            return None

        now = time.monotonic()
        if self._gpu_cache is not None and (now - self._gpu_cache_time) < 1.0:
            return self._gpu_cache

        try:
            import pynvml
            handle = self._gpu_handle

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = 0

            try:
                display_mode = pynvml.nvmlDeviceGetDisplayMode(handle)
                is_display = display_mode != 0
            except Exception:
                is_display = False

            info = GpuInfo(
                name=name,
                vram_total_mb=int(mem.total // (1024 * 1024)),
                vram_used_mb=int(mem.used // (1024 * 1024)),
                vram_free_mb=int(mem.free // (1024 * 1024)),
                gpu_utilization=int(util.gpu),
                temperature=int(temp),
                is_display_attached=is_display,
            )
            self._gpu_cache = info
            self._gpu_cache_time = now
            return info
        except Exception:
            return None

    # -- CPU / RAM --------------------------------------------------------

    def get_resources(self) -> SystemResources:
        """Return a full snapshot of CPU, RAM, and GPU resources."""
        vm = psutil.virtual_memory()
        return SystemResources(
            cpu_percent=psutil.cpu_percent(interval=None),
            cpu_count=psutil.cpu_count(logical=True) or 1,
            ram_total_mb=int(vm.total // (1024 * 1024)),
            ram_used_mb=int(vm.used // (1024 * 1024)),
            ram_free_mb=int(vm.available // (1024 * 1024)),
            gpu=self.get_gpu_info(),
        )

    # -- Limit helpers ----------------------------------------------------

    def get_vram_limit_mb(self, limit_percent: int = 95) -> int:
        """Max usable VRAM in MB based on a percentage of total."""
        gpu = self.get_gpu_info()
        if gpu is None:
            return 0
        return int(gpu.vram_total_mb * limit_percent / 100)

    def get_ram_limit_mb(self, limit_percent: int = 85) -> int:
        """Max usable RAM in MB based on a percentage of total."""
        vm = psutil.virtual_memory()
        total_mb = int(vm.total // (1024 * 1024))
        return int(total_mb * limit_percent / 100)
