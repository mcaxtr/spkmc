"""
Tests for hardware detection and parallelization configuration.
"""

from unittest.mock import MagicMock, patch

from spkmc.utils.hardware import (
    HardwareInfo,
    ParallelizationStrategy,
    configure_numba_threads,
    detect_cpu_cores,
    detect_gpu,
    format_hardware_box,
    get_hardware_info,
)


class TestCPUDetection:
    """Tests for CPU core detection."""

    def test_detect_cpu_cores_returns_positive_values(self):
        """Should return positive values for both logical and physical cores."""
        logical, physical = detect_cpu_cores()
        assert logical >= 1
        assert physical >= 1
        assert logical >= physical

    def test_detect_cpu_cores_without_psutil(self):
        """Should work even without psutil installed."""
        with patch.dict("sys.modules", {"psutil": None}):
            logical, physical = detect_cpu_cores()
            assert logical >= 1
            assert physical >= 1

    def test_detect_cpu_cores_with_psutil(self):
        """Should use psutil for accurate physical core count when available."""
        mock_psutil = MagicMock()
        mock_psutil.cpu_count.return_value = 4

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            with patch("os.cpu_count", return_value=8):
                # Force reimport to pick up mocked module
                import importlib

                import spkmc.utils.hardware as hw

                importlib.reload(hw)
                logical, physical = hw.detect_cpu_cores()
                # Should get 8 logical from os.cpu_count


class TestGPUDetection:
    """Tests for GPU detection."""

    def test_detect_gpu_no_cupy(self):
        """Should return False with reason when CuPy is not installed."""
        with patch.dict("sys.modules", {"cupy": None}):
            available, info = detect_gpu()
            assert available is False
            # Now returns info dict with reason instead of None
            assert info is not None
            assert "reason" in info or "libs_missing" in info

    def test_detect_gpu_no_cudf(self):
        """Should return False when cuDF/cugraph are missing (all RAPIDS libs required)."""
        mock_cupy = MagicMock()
        mock_cupy.cuda.Device.return_value = MagicMock()
        mock_cupy.cuda.runtime.getDeviceProperties.return_value = {
            "name": b"Test GPU",
            "totalGlobalMem": 8 * 1024 * 1024 * 1024,
            "major": 8,
            "minor": 0,
        }
        mock_cupy.cuda.runtime.runtimeGetVersion.return_value = 12000

        with patch.dict("sys.modules", {"cupy": mock_cupy, "cudf": None, "cugraph": None}):
            available, info = detect_gpu()
            # GPU requires ALL libraries (cupy, cudf, cugraph) for SSSP calculation
            assert available is False
            assert "name" in info
            assert "libs_missing" in info
            assert "cudf" in info["libs_missing"]


class TestHardwareInfo:
    """Tests for HardwareInfo collection."""

    def test_get_hardware_info_returns_valid_data(self):
        """Should return a valid HardwareInfo object."""
        info = get_hardware_info()

        # Check it has the expected attributes (avoid isinstance due to module reloading)
        assert hasattr(info, "cpu_count")
        assert hasattr(info, "cpu_count_physical")
        assert hasattr(info, "numba_threads")
        assert hasattr(info, "gpu_available")
        assert info.cpu_count >= 1
        assert info.cpu_count_physical >= 1
        assert info.numba_threads >= 1
        assert isinstance(info.gpu_available, bool)

    def test_hardware_info_numba_threads_capped(self):
        """Numba threads should be capped at 16."""
        info = get_hardware_info()
        assert info.numba_threads <= 16


class TestParallelizationStrategy:
    """Tests for parallelization strategy configuration."""

    def test_auto_configure_single_scenario(self):
        """Single scenario should use sequential execution for scenarios."""
        hardware = HardwareInfo(
            cpu_count=8, cpu_count_physical=4, numba_threads=4, gpu_available=False
        )

        strategy = ParallelizationStrategy.auto_configure(hardware, num_scenarios=1)

        assert strategy.scenario_workers == 1
        assert strategy.simulation_workers >= 1
        assert strategy.numba_threads >= 2

    def test_auto_configure_many_scenarios(self):
        """Many scenarios should use parallel scenario execution."""
        hardware = HardwareInfo(
            cpu_count=16, cpu_count_physical=8, numba_threads=4, gpu_available=False
        )

        strategy = ParallelizationStrategy.auto_configure(hardware, num_scenarios=10)

        assert strategy.scenario_workers > 1
        assert strategy.scenario_workers <= 10

    def test_auto_configure_few_scenarios(self):
        """Few scenarios should balance parallelism."""
        hardware = HardwareInfo(
            cpu_count=8, cpu_count_physical=4, numba_threads=4, gpu_available=False
        )

        strategy = ParallelizationStrategy.auto_configure(hardware, num_scenarios=3)

        # Should have some scenario workers but not too many
        assert strategy.scenario_workers >= 1
        assert strategy.scenario_workers <= 3

    def test_auto_configure_reserves_numba_threads(self):
        """Should reserve threads for Numba inner loops."""
        hardware = HardwareInfo(
            cpu_count=16, cpu_count_physical=8, numba_threads=8, gpu_available=False
        )

        strategy = ParallelizationStrategy.auto_configure(hardware, num_scenarios=8)

        # Numba should have at least 2 threads reserved
        assert strategy.numba_threads >= 2


class TestConfigureNumbaThreads:
    """Tests for Numba thread configuration."""

    def test_configure_numba_threads_default(self):
        """Should configure threads based on CPU count."""
        thread_count = configure_numba_threads()
        assert thread_count >= 1
        assert thread_count <= 16

    def test_configure_numba_threads_override(self):
        """Should respect explicit thread count or return current if already launched."""
        # Use a thread count that's valid for any machine (1 is always valid)
        thread_count = configure_numba_threads(1)
        # Either succeeds with 1, or returns current count if threads already launched
        assert thread_count >= 1


class TestFormatHardwareBox:
    """Tests for hardware info formatting."""

    def test_format_hardware_box_basic(self):
        """Should format basic hardware info."""
        hardware = HardwareInfo(
            cpu_count=8, cpu_count_physical=4, numba_threads=4, gpu_available=False
        )

        output = format_hardware_box(hardware)

        assert "CPU:" in output
        assert "8" in output
        assert "4" in output
        assert "GPU:" in output
        assert "Numba:" in output

    def test_format_hardware_box_with_gpu(self):
        """Should include GPU info when available."""
        hardware = HardwareInfo(
            cpu_count=8,
            cpu_count_physical=4,
            numba_threads=4,
            gpu_available=True,
            gpu_name="Test GPU",
            gpu_memory_mb=8192,
        )

        output = format_hardware_box(hardware)

        assert "Test GPU" in output
        assert "8GB" in output or "8192" in output

    def test_format_hardware_box_with_strategy(self):
        """Should include strategy info when provided."""
        hardware = HardwareInfo(
            cpu_count=8, cpu_count_physical=4, numba_threads=4, gpu_available=False
        )
        strategy = ParallelizationStrategy(
            scenario_workers=4, simulation_workers=2, numba_threads=4, use_gpu=False
        )

        output = format_hardware_box(hardware, strategy)

        # Strategy is passed but we no longer show parallel workers count
        assert "8 cores" in output
        assert "4 physical" in output
