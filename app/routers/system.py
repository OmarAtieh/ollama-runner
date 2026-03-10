"""System resource monitoring and binary management endpoints."""

from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks

from app.services.system_monitor import SystemMonitor
from app.services.binary_manager import BinaryManager

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/resources")
async def get_resources():
    """Return a full snapshot of system resources."""
    monitor = SystemMonitor.instance()
    resources = monitor.get_resources()
    return asdict(resources)


@router.get("/binary/status")
async def binary_status():
    """Return the current status of the llama-server binary."""
    bm = BinaryManager.instance()
    return bm.get_status()


@router.post("/binary/download")
async def binary_download():
    """Trigger download and installation of the llama-server binary."""
    bm = BinaryManager.instance()
    result = await bm.download_and_install()
    return bm.get_status()
