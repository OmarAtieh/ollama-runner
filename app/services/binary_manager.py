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

    async def get_latest_release_url(self) -> str | None:
        """Query GitHub API for the latest llama.cpp release and return the
        Windows CUDA zip asset URL, or None on failure."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    GITHUB_LATEST_RELEASE_URL,
                    headers={"Accept": "application/vnd.github+json"},
                ) as resp:
                    if resp.status != 200:
                        log.warning("GitHub API returned status %s", resp.status)
                        return None
                    data = await resp.json()
        except Exception as exc:
            log.error("Failed to query GitHub releases: %s", exc)
            return None

        assets = data.get("assets", [])
        candidates: list[dict] = []
        for asset in assets:
            name: str = asset.get("name", "").lower()
            if name.endswith(".zip") and "win" in name and "cuda" in name:
                candidates.append(asset)

        if not candidates:
            return None

        # Prefer non-vulkan builds
        non_vulkan = [a for a in candidates if "vulkan" not in a["name"].lower()]
        chosen = non_vulkan[0] if non_vulkan else candidates[0]
        return chosen["browser_download_url"]

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

        try:
            # -- Download with progress tracking ---------------------------------
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        log.error("Download failed with status %s", resp.status)
                        self._download_status = "error"
                        return False

                    total = int(resp.headers.get("Content-Length", 0))
                    downloaded = 0

                    with open(zip_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                self._download_progress = downloaded / total

            self._download_progress = 1.0

            # -- Extract ---------------------------------------------------------
            self._download_status = "extracting"
            self._extract_zip(zip_path)

            # -- Clean up --------------------------------------------------------
            zip_path.unlink(missing_ok=True)

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
