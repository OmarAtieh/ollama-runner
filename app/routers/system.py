"""System resource monitoring endpoint."""

from dataclasses import asdict

from fastapi import APIRouter

from app.services.system_monitor import SystemMonitor

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/resources")
async def get_resources():
    """Return a full snapshot of system resources."""
    monitor = SystemMonitor.instance()
    resources = monitor.get_resources()
    return asdict(resources)
