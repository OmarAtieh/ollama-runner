"""System-level optimizations for inference performance."""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from app.config import APP_DIR, BIN_DIR, MODELS_DIR

log = logging.getLogger(__name__)


class SystemOptimizer:
    """Detects and applies system-level performance optimizations."""

    @staticmethod
    def check_defender_exclusions() -> dict:
        """Check if Windows Defender exclusions are configured.
        Returns dict with 'configured' bool and 'missing' list of paths that need exclusion."""
        if platform.system() != "Windows":
            return {"configured": True, "missing": []}

        paths_to_exclude = [
            str(APP_DIR),
            str(BIN_DIR),
            str(MODELS_DIR),
        ]
        extensions_to_exclude = [".gguf"]

        missing = []
        try:
            # Query current exclusions via PowerShell
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-MpPreference | Select-Object -ExpandProperty ExclusionPath"],
                capture_output=True, text=True, timeout=10,
            )
            current_paths = set(result.stdout.strip().splitlines())

            for path in paths_to_exclude:
                if path not in current_paths:
                    missing.append({"type": "path", "value": path})

            result_ext = subprocess.run(
                ["powershell", "-Command",
                 "Get-MpPreference | Select-Object -ExpandProperty ExclusionExtension"],
                capture_output=True, text=True, timeout=10,
            )
            current_exts = set(result_ext.stdout.strip().splitlines())

            for ext in extensions_to_exclude:
                if ext not in current_exts:
                    missing.append({"type": "extension", "value": ext})

        except Exception as exc:
            log.warning("Could not check Defender exclusions: %s", exc)
            # If we can't check, assume not configured
            for path in paths_to_exclude:
                missing.append({"type": "path", "value": path})
            for ext in extensions_to_exclude:
                missing.append({"type": "extension", "value": ext})

        return {"configured": len(missing) == 0, "missing": missing}

    @staticmethod
    def get_defender_exclusion_commands() -> list[str]:
        """Return PowerShell commands to add Defender exclusions (requires admin)."""
        return [
            f'Add-MpPreference -ExclusionPath "{APP_DIR}"',
            f'Add-MpPreference -ExclusionPath "{BIN_DIR}"',
            f'Add-MpPreference -ExclusionPath "{MODELS_DIR}"',
            'Add-MpPreference -ExclusionExtension ".gguf"',
        ]

    @staticmethod
    def check_gpu_clocks() -> dict:
        """Check if GPU clocks are locked for consistent performance."""
        if platform.system() != "Windows":
            return {"locked": False, "available": False}

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.current.graphics,clocks.max.graphics",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) == 2:
                    current = int(parts[0])
                    max_clock = int(parts[1])
                    return {
                        "locked": current >= max_clock - 50,  # within 50MHz = locked
                        "available": True,
                        "current_mhz": current,
                        "max_mhz": max_clock,
                    }
        except Exception as exc:
            log.warning("Could not check GPU clocks: %s", exc)

        return {"locked": False, "available": False}

    @staticmethod
    def get_optimization_status() -> dict:
        """Get a summary of all optimization states."""
        return {
            "defender_exclusions": SystemOptimizer.check_defender_exclusions(),
            "gpu_clocks": SystemOptimizer.check_gpu_clocks(),
        }
