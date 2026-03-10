"""Auto-download and manage the llama-server CUDA binary from llama.cpp releases."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import aiohttp

from app.config import BIN_DIR

log = logging.getLogger(__name__)

GITHUB_LATEST_RELEASE_URL = (
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
)


class BinaryManager:
    """Downloads and manages the llama-server binary."""

    _instance: BinaryManager | None = None

    def __init__(self, bin_dir: Path | None = None) -> None:
        self._bin_dir = bin_dir or BIN_DIR
        self._bin_dir.mkdir(parents=True, exist_ok=True)
        self._binary_path = self._bin_dir / "llama-server.exe"
        self._download_progress: float = 0.0
        self._download_status: str = "idle"  # idle|downloading|extracting|ready|error

    @classmethod
    def instance(cls) -> BinaryManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- Properties ----------------------------------------------------------

    @property
    def binary_path(self) -> Path:
        return self._binary_path

    @property
    def is_available(self) -> bool:
        return self._binary_path.exists()

    @property
    def download_progress(self) -> float:
        return self._download_progress

    @property
    def download_status(self) -> str:
        return self._download_status

    # -- Public API ----------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "available": self.is_available,
            "path": str(self._binary_path),
            "status": self._download_status,
            "progress": self._download_progress,
        }

    def _detect_cuda_version(self) -> str:
        """Detect installed CUDA version from nvidia-smi."""
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10,
            )
            # Parse "CUDA Version: 13.1" from output
            for line in result.stdout.splitlines():
                if "CUDA Version:" in line:
                    import re
                    match = re.search(r"CUDA Version:\s*([\d.]+)", line)
                    if match:
                        return match.group(1)
        except Exception:
            pass
        return "12.4"  # fallback

    async def _fetch_release_data(self) -> dict | None:
        """Fetch latest release data from GitHub."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    GITHUB_LATEST_RELEASE_URL,
                    headers={"Accept": "application/vnd.github+json"},
                ) as resp:
                    if resp.status != 200:
                        log.warning("GitHub API returned status %s", resp.status)
                        return None
                    return await resp.json()
        except Exception as exc:
            log.error("Failed to query GitHub releases: %s", exc)
            return None

    async def get_latest_release_url(self) -> str | None:
        """Query GitHub API for the latest llama.cpp release and return the
        Windows CUDA zip asset URL, or None on failure."""
        data = await self._fetch_release_data()
        if not data:
            return None

        cuda_ver = self._detect_cuda_version()
        # Normalize: "13.1" -> "13.1", try to match against asset names
        cuda_major_minor = cuda_ver  # e.g. "13.1" or "12.4"

        assets = data.get("assets", [])

        # Find the main binary zip (not cudart)
        def find_binary(cuda_tag: str) -> str | None:
            for asset in assets:
                name: str = asset.get("name", "").lower()
                if (name.endswith(".zip") and "win" in name and
                    "cuda" in name and cuda_tag in name and
                    "vulkan" not in name and "cudart" not in name):
                    return asset["browser_download_url"]
            return None

        # Try exact CUDA version match first, then fallback
        url = find_binary(f"cuda-{cuda_major_minor}")
        if not url:
            url = find_binary("cuda-12.4")  # widely compatible fallback
        if not url:
            url = find_binary("cuda")  # any CUDA build
        return url

    async def _get_cudart_url(self) -> str | None:
        """Get the CUDA runtime DLL zip URL matching our CUDA version."""
        data = await self._fetch_release_data()
        if not data:
            return None

        cuda_ver = self._detect_cuda_version()
        assets = data.get("assets", [])

        def find_cudart(cuda_tag: str) -> str | None:
            for asset in assets:
                name: str = asset.get("name", "").lower()
                if (name.endswith(".zip") and "win" in name and
                    "cudart" in name and cuda_tag in name):
                    return asset["browser_download_url"]
            return None

        url = find_cudart(f"cuda-{cuda_ver}")
        if not url:
            url = find_cudart("cuda-12.4")
        return url

    async def _download_file(self, url: str, dest: Path, label: str = "") -> bool:
        """Download a file with progress tracking."""
        try:
            timeout = aiohttp.ClientTimeout(total=600)  # 10 min timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        log.error("Download %s failed with status %s", label, resp.status)
                        return False

                    total = int(resp.headers.get("Content-Length", 0))
                    downloaded = 0

                    with open(dest, "wb") as f:
                        async for chunk in resp.content.iter_chunked(65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                self._download_progress = downloaded / total

            return True
        except Exception as exc:
            log.error("Download %s failed: %s", label, exc)
            return False

    async def download_and_install(self) -> bool:
        """Download and extract llama-server into bin_dir. Returns True on success."""
        self._download_progress = 0.0
        self._download_status = "downloading"

        url = await self.get_latest_release_url()
        if url is None:
            log.error("Could not find a download URL for llama-server")
            self._download_status = "error"
            return False

        zip_path = self._bin_dir / "llama-server.zip"
        cudart_zip_path = self._bin_dir / "cudart.zip"

        try:
            # -- Download main binary --------------------------------------------
            log.info("Downloading llama-server from %s", url)
            if not await self._download_file(url, zip_path, "llama-server"):
                self._download_status = "error"
                return False

            self._download_progress = 1.0

            # -- Extract main binary ---------------------------------------------
            self._download_status = "extracting"
            self._extract_zip(zip_path)
            zip_path.unlink(missing_ok=True)

            # -- Download CUDA runtime DLLs if needed ----------------------------
            cudart_url = await self._get_cudart_url()
            if cudart_url:
                log.info("Downloading CUDA runtime DLLs...")
                self._download_status = "downloading"
                self._download_progress = 0.0
                if await self._download_file(cudart_url, cudart_zip_path, "cudart"):
                    self._download_status = "extracting"
                    self._extract_zip(cudart_zip_path)
                    cudart_zip_path.unlink(missing_ok=True)
                else:
                    log.warning("CUDA runtime download failed, llama-server may still work if CUDA is installed system-wide")
                    cudart_zip_path.unlink(missing_ok=True)

            # -- Verify ----------------------------------------------------------
            if self._binary_path.exists():
                self._download_status = "ready"
                log.info("llama-server installed at %s", self._binary_path)
                return True
            else:
                log.error("llama-server.exe not found after extraction")
                self._download_status = "error"
                return False

        except Exception as exc:
            log.error("Download/install failed: %s", exc)
            self._download_status = "error"
            zip_path.unlink(missing_ok=True)
            cudart_zip_path.unlink(missing_ok=True)
            return False

    # -- Private helpers -----------------------------------------------------

    def _extract_zip(self, zip_path: Path) -> None:
        """Extract llama-server.exe and .dll files flat into bin_dir."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name.lower()
                if name == "llama-server.exe" or name.endswith(".dll"):
                    # Extract flat — use just the filename, not the nested path
                    target = self._bin_dir / Path(info.filename).name
                    with zf.open(info) as src, open(target, "wb") as dst:
                        dst.write(src.read())
