"""Tests for the system resource monitor service and router."""

import pytest
from fastapi.testclient import TestClient

from app.services.system_monitor import SystemMonitor, SystemResources, GpuInfo
from app.main import app


client = TestClient(app)


class TestSystemMonitorInstantiation:
    def test_can_instantiate(self):
        monitor = SystemMonitor()
        assert monitor is not None

    def test_singleton_returns_same_instance(self):
        a = SystemMonitor.instance()
        b = SystemMonitor.instance()
        assert a is b


class TestGetResources:
    def test_returns_system_resources(self):
        monitor = SystemMonitor()
        res = monitor.get_resources()
        assert isinstance(res, SystemResources)

    def test_cpu_percent_non_negative(self):
        monitor = SystemMonitor()
        res = monitor.get_resources()
        assert res.cpu_percent >= 0

    def test_ram_total_positive(self):
        monitor = SystemMonitor()
        res = monitor.get_resources()
        assert res.ram_total_mb > 0

    def test_ram_used_less_than_total(self):
        monitor = SystemMonitor()
        res = monitor.get_resources()
        assert res.ram_used_mb <= res.ram_total_mb

    def test_ram_free_non_negative(self):
        monitor = SystemMonitor()
        res = monitor.get_resources()
        assert res.ram_free_mb >= 0

    def test_cpu_count_positive(self):
        monitor = SystemMonitor()
        res = monitor.get_resources()
        assert res.cpu_count > 0

    def test_gpu_is_gpuinfo_or_none(self):
        monitor = SystemMonitor()
        res = monitor.get_resources()
        assert res.gpu is None or isinstance(res.gpu, GpuInfo)


class TestResourceLimits:
    def test_vram_limit_returns_int(self):
        monitor = SystemMonitor()
        limit = monitor.get_vram_limit_mb()
        assert isinstance(limit, int)

    def test_vram_limit_zero_when_no_gpu(self):
        monitor = SystemMonitor()
        gpu = monitor.get_gpu_info()
        limit = monitor.get_vram_limit_mb()
        if gpu is None:
            assert limit == 0
        else:
            assert limit > 0

    def test_ram_limit_returns_int(self):
        monitor = SystemMonitor()
        limit = monitor.get_ram_limit_mb()
        assert isinstance(limit, int)

    def test_ram_limit_positive(self):
        monitor = SystemMonitor()
        limit = monitor.get_ram_limit_mb()
        assert limit > 0

    def test_ram_limit_less_than_total(self):
        monitor = SystemMonitor()
        res = monitor.get_resources()
        limit = monitor.get_ram_limit_mb()
        assert limit <= res.ram_total_mb

    def test_custom_limit_percent(self):
        monitor = SystemMonitor()
        limit_50 = monitor.get_ram_limit_mb(limit_percent=50)
        limit_85 = monitor.get_ram_limit_mb(limit_percent=85)
        assert limit_50 < limit_85


class TestGpuInfo:
    def test_gpu_info_is_none_or_valid(self):
        monitor = SystemMonitor()
        gpu = monitor.get_gpu_info()
        if gpu is not None:
            assert gpu.name != ""
            assert gpu.vram_total_mb > 0
            assert gpu.vram_free_mb >= 0
            assert gpu.vram_used_mb >= 0


class TestSystemEndpoint:
    def test_resources_endpoint_returns_200(self):
        response = client.get("/api/system/resources")
        assert response.status_code == 200

    def test_resources_endpoint_has_expected_fields(self):
        response = client.get("/api/system/resources")
        data = response.json()
        assert "cpu_percent" in data
        assert "cpu_count" in data
        assert "ram_total_mb" in data
        assert "ram_used_mb" in data
        assert "ram_free_mb" in data
        assert "gpu" in data

    def test_resources_endpoint_values_sensible(self):
        response = client.get("/api/system/resources")
        data = response.json()
        assert data["cpu_percent"] >= 0
        assert data["ram_total_mb"] > 0
        assert data["cpu_count"] > 0
