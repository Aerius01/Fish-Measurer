"""
Pytest configuration and shared fixtures.

This module provides test fixtures and configuration that can be used
across all test modules.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


# ========================
# Mock Camera Fixtures
# ========================

@pytest.fixture
def mock_camera():
    """Create a mock camera for testing."""
    camera = Mock()
    camera.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera.config = Mock(framerate=30)
    camera.get_frames = Mock(return_value=([], []))
    return camera


@pytest.fixture
def mock_camera_with_frames():
    """Create a mock camera that returns test frames."""
    camera = Mock()

    # Create test frames
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    camera.current_frame = test_frame

    camera.config = Mock(framerate=30)

    def mock_get_frames(count):
        raw_frames = [test_frame.copy() for _ in range(count)]
        binary_frames = [None for _ in range(count)]
        return raw_frames, binary_frames

    camera.get_frames = mock_get_frames

    return camera


# ========================
# Mock Vision Component Fixtures
# ========================

@pytest.fixture
def mock_aruco_detector():
    """Create a mock ArUco detector."""
    detector = Mock()
    detector.detect_markers = Mock(return_value=None)
    detector.calculate_calibration = Mock()
    detector.get_average_calibration = Mock(return_value=None)
    detector.convert_pixels_to_length = Mock(return_value=None)
    detector.clear_calibration = Mock()
    return detector


@pytest.fixture
def mock_fish_processor():
    """Create a mock fish processor."""
    processor = Mock()

    # Create a mock result
    mock_result = Mock()
    mock_result.success = True
    mock_result.total_length_pixels = 100.0
    mock_result.processing_log = []

    processor.process_fish = Mock(return_value=mock_result)

    return processor


# ========================
# Configuration Fixtures
# ========================

@pytest.fixture
def mock_config():
    """Create a mock measurement configuration."""
    from logic import MeasurementConfig
    config = MeasurementConfig()
    config.set_threshold(50)
    return config


@pytest.fixture
def mock_folder_manager():
    """Create a mock folder manager."""
    manager = Mock()
    manager.setup_folders = Mock()
    manager.get_raw_path = Mock(return_value="/tmp/raw_0.jpg")
    manager.get_skeleton_path = Mock(return_value="/tmp/skeleton_0.jpg")
    manager.reset = Mock()
    return manager


# ========================
# Test Data Fixtures
# ========================

@pytest.fixture
def test_image():
    """Create a test image (640x480 BGR)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_binary_mask():
    """Create a test binary mask (640x480)."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    # Add a white region in the center
    mask[200:280, 270:370] = 255
    return mask


@pytest.fixture
def test_frame_sequence():
    """Create a sequence of test frames."""
    frames = []
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


# ========================
# Marker Configuration
# ========================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may require hardware)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "gui: Tests requiring GUI components"
    )
    config.addinivalue_line(
        "markers", "camera: Tests requiring camera hardware"
    )


# ========================
# Test Utility Functions
# ========================

def create_test_measurement_result(length_pixels=100.0, success=True):
    """
    Create a mock measurement result for testing.

    Args:
        length_pixels: Length in pixels
        success: Whether measurement was successful

    Returns:
        Mock measurement result
    """
    result = Mock()
    result.success = success
    result.total_length_pixels = length_pixels
    result.fil_length_pixels = length_pixels
    result.processing_log = ["Test log entry"]
    result.visualizations = {}
    return result


def create_test_calibration_data(pixels_per_cm=10.0, marker_count=5):
    """
    Create mock calibration data for testing.

    Args:
        pixels_per_cm: Calibration ratio
        marker_count: Number of calibration samples

    Returns:
        Mock calibration data
    """
    calibration = Mock()
    calibration.pixels_per_cm = pixels_per_cm
    calibration.marker_count = marker_count
    return calibration
