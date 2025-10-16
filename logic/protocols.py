"""
Protocol interfaces for dependency injection.

This module defines the contracts (protocols) that components must implement,
enabling dependency injection, testability, and loose coupling.
"""

from typing import Protocol, Optional, Tuple, List, Any, Callable
from dataclasses import dataclass
import numpy as np


# ========================
# Camera Protocols
# ========================

class ICameraProvider(Protocol):
    """Protocol for camera frame providers."""

    @property
    def current_frame(self) -> Optional[np.ndarray]:
        """Get the current camera frame."""
        ...

    @property
    def config(self) -> Any:
        """Get camera configuration."""
        ...

    def get_frames(self, count: int) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        """
        Capture a specific number of frames.

        Args:
            count: Number of frames to capture

        Returns:
            Tuple of (raw_frames, binarized_frames)
        """
        ...


class ICameraSettings(Protocol):
    """Protocol for camera settings."""

    framerate: float
    exposure_ms: Optional[float]
    gain_mode: str
    white_balance_mode: str


# ========================
# Calibration Protocols
# ========================

@dataclass
class ICalibrationData:
    """Protocol for calibration data."""
    pixels_per_cm: float
    marker_count: int


class ICalibrationProvider(Protocol):
    """Protocol for calibration providers."""

    def detect_markers(self, frame: np.ndarray) -> Any:
        """
        Detect ArUco markers in frame.

        Args:
            frame: Input frame

        Returns:
            Detected markers or None
        """
        ...

    def calculate_calibration(self, markers: Any) -> None:
        """
        Calculate calibration from detected markers.

        Args:
            markers: Detected markers
        """
        ...

    def get_average_calibration(self) -> Optional[ICalibrationData]:
        """
        Get averaged calibration data.

        Returns:
            Calibration data or None
        """
        ...

    def convert_pixels_to_length(self, pixels: float) -> Optional[float]:
        """
        Convert pixel measurement to length.

        Args:
            pixels: Measurement in pixels

        Returns:
            Length in cm or None
        """
        ...

    def clear_calibration(self) -> None:
        """Clear calibration history."""
        ...


# ========================
# Processing Protocols
# ========================

class IFishProcessor(Protocol):
    """Protocol for fish processing."""

    def process_fish(
        self,
        raw_frame: np.ndarray,
        binary_mask: np.ndarray,
        fish_id: str
    ) -> Any:
        """
        Process fish measurement from frames.

        Args:
            raw_frame: Raw camera frame
            binary_mask: Binary mask of fish
            fish_id: Identifier for this fish

        Returns:
            Fish measurement result
        """
        ...


# ========================
# Configuration Protocols
# ========================

class IConfiguration(Protocol):
    """Protocol for configuration management."""

    def get_threshold(self) -> int:
        """Get background subtraction threshold."""
        ...

    def set_threshold(self, value: int) -> None:
        """Set background subtraction threshold."""
        ...

    def get_fish_id(self) -> Optional[str]:
        """Get current fish ID."""
        ...

    def set_fish_id(self, fish_id: Optional[str]) -> None:
        """Set fish ID."""
        ...

    def should_stop(self) -> bool:
        """Check if processing should stop."""
        ...

    def request_stop(self) -> None:
        """Request processing to stop."""
        ...

    def reset_stop(self) -> None:
        """Reset stop request."""
        ...

    def add_error(self, category: str, message: str) -> None:
        """Add an error message."""
        ...

    def get_errors(self, category: str) -> List[str]:
        """Get errors for a category."""
        ...

    def clear_errors(self, category: Optional[str] = None) -> None:
        """Clear errors."""
        ...

    def reset(self) -> None:
        """Reset configuration."""
        ...


# ========================
# Storage Protocols
# ========================

class IFolderManager(Protocol):
    """Protocol for folder management."""

    def setup_folders(self, base_path: str, fish_id: Optional[str]) -> None:
        """
        Setup folder structure.

        Args:
            base_path: Base output path
            fish_id: Fish identifier
        """
        ...

    def get_raw_path(self, frame_id: int, format: str) -> Any:
        """Get path for raw frame."""
        ...

    def get_skeleton_path(self, frame_id: int, format: str) -> Any:
        """Get path for skeleton frame."""
        ...

    def reset(self) -> None:
        """Reset folder manager."""
        ...


# ========================
# State Management Protocols
# ========================

class IStateMachine(Protocol):
    """Protocol for state machine."""

    @property
    def current_state(self) -> Any:
        """Get current state."""
        ...

    def transition_to(self, new_state: Any, reason: Optional[str] = None, metadata: Optional[dict] = None) -> bool:
        """
        Transition to new state.

        Args:
            new_state: Target state
            reason: Reason for transition
            metadata: Additional metadata

        Returns:
            True if successful
        """
        ...

    def add_observer(self, observer: Any) -> None:
        """Add state observer."""
        ...

    def remove_observer(self, observer: Any) -> None:
        """Remove state observer."""
        ...

    def reset(self) -> None:
        """Reset state machine."""
        ...


# ========================
# Measurement Protocols
# ========================

class IMeasurementOrchestrator(Protocol):
    """Protocol for measurement orchestration."""

    @property
    def state_machine(self) -> IStateMachine:
        """Get state machine."""
        ...

    def start_calibration(self) -> bool:
        """Start calibration."""
        ...

    def update_calibration(self, frame: np.ndarray) -> Tuple[bool, int]:
        """Update calibration with frame."""
        ...

    def complete_calibration(self) -> Any:
        """Complete calibration."""
        ...

    def start_background_training(self, duration_seconds: int, async_mode: bool) -> bool:
        """Start background training."""
        ...

    def is_training_complete(self) -> bool:
        """Check if training is complete."""
        ...

    def start_processing(self, num_frames: int, async_mode: bool) -> bool:
        """Start frame processing."""
        ...

    def is_processing_complete(self) -> bool:
        """Check if processing is complete."""
        ...

    def cancel_processing(self) -> None:
        """Cancel ongoing processing."""
        ...

    def set_output_folder(self, folder: str) -> None:
        """Set output folder."""
        ...

    def set_image_format(self, format: str) -> None:
        """Set image format."""
        ...

    def set_fish_id(self, fish_id: str) -> None:
        """Set fish ID."""
        ...

    def set_progress_callback(self, callback: Any) -> None:
        """Set progress callback."""
        ...

    def reset(self) -> None:
        """Reset orchestrator."""
        ...


# ========================
# Statistics Protocols
# ========================

class IMeasurementStatistics(Protocol):
    """Protocol for measurement statistics."""

    @staticmethod
    def filter_outliers(measurements: List[Any]) -> List[Any]:
        """Filter outliers from measurements."""
        ...

    @staticmethod
    def export_data(
        measurements: List[Any],
        folder_manager: IFolderManager,
        image_format: str,
        watermarker: Any
    ) -> None:
        """Export measurement data."""
        ...


# ========================
# Helper type aliases
# ========================

# Type alias for progress callbacks
ProgressCallback = Callable[[int, int, str], None]

# Type alias for state observers
StateObserver = Callable[[Any], None]
