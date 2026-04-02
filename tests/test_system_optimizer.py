"""Tests for system optimizer."""

import platform
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.services.system_optimizer import SystemOptimizer


class TestDefenderExclusions:
    def test_returns_configured_on_non_windows(self):
        with patch("platform.system", return_value="Linux"):
            result = SystemOptimizer.check_defender_exclusions()
            assert result["configured"] is True
            assert result["missing"] == []

    def test_returns_missing_when_not_excluded(self):
        with patch("platform.system", return_value="Windows"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="C:\\SomeOtherPath\n",
                returncode=0,
            )
            result = SystemOptimizer.check_defender_exclusions()
            assert result["configured"] is False
            assert len(result["missing"]) > 0

    def test_get_commands_returns_powershell(self):
        commands = SystemOptimizer.get_defender_exclusion_commands()
        assert len(commands) == 4
        assert all("Add-MpPreference" in cmd for cmd in commands)


class TestGpuClocks:
    def test_returns_unavailable_on_non_windows(self):
        with patch("platform.system", return_value="Linux"):
            result = SystemOptimizer.check_gpu_clocks()
            assert result["available"] is False

    def test_parses_nvidia_smi_output(self):
        with patch("platform.system", return_value="Windows"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="1620, 1620\n",
                returncode=0,
            )
            result = SystemOptimizer.check_gpu_clocks()
            assert result["available"] is True
            assert result["locked"] is True
            assert result["current_mhz"] == 1620
            assert result["max_mhz"] == 1620

    def test_detects_unlocked_clocks(self):
        with patch("platform.system", return_value="Windows"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="300, 1620\n",
                returncode=0,
            )
            result = SystemOptimizer.check_gpu_clocks()
            assert result["locked"] is False
            assert result["current_mhz"] == 300


class TestOptimizationStatus:
    def test_returns_all_sections(self):
        with patch("platform.system", return_value="Linux"):
            result = SystemOptimizer.get_optimization_status()
            assert "defender_exclusions" in result
            assert "gpu_clocks" in result
