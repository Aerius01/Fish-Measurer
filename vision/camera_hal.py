"""
Hardware Abstraction Layer (HAL) for Basler camera operations.

This module provides a clean interface for Basler cameras using pypylon,
with proper encapsulation and unified API for GUI implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
from contextlib import contextmanager
import threading
import time
import logging

import cv2
import numpy as np

try:
    from pypylon import pylon, genicam
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CameraSettings:
    """Configuration settings for camera operations."""
    # Capture settings
    framerate: float = 30.0
    timeout_ms: int = 5000

    # Image processing settings
    exposure_ms: Optional[float] = None  # None = auto
    gain_mode: str = "Once"  # "Once", "Continuous", "Off"
    white_balance_mode: str = "Off"  # "Once", "Continuous", "Off"

    # Frame format
    pixel_format: str = "BGR8"  # BGR8, RGB8, Mono8, etc.

    def __post_init__(self):
        """Validate settings."""
        if self.framerate <= 0:
            raise ValueError("Framerate must be positive")
        if self.exposure_ms is not None and self.exposure_ms < 0:
            raise ValueError("Exposure must be non-negative")


@dataclass
class CameraInfo:
    """Information about a camera device."""
    device_id: str
    model_name: str
    serial_number: str
    camera_type: str  # "basler"


@dataclass
class CameraCapabilities:
    """Camera capabilities and available options."""
    name: str
    gain_auto_options: List[str]
    white_balance_auto_options: List[str]


class CameraError(Exception):
    """Exception raised for camera-related errors."""
    pass


class CameraHAL(ABC):
    """
    Abstract Hardware Abstraction Layer for cameras.

    This interface defines the contract that all camera implementations must follow,
    providing a unified API regardless of the underlying hardware.
    """

    def __init__(self, settings: Optional[CameraSettings] = None):
        """
        Initialize the camera HAL.

        Args:
            settings: Camera configuration settings
        """
        self.settings = settings or CameraSettings()
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.RLock()
        self._is_connected = False
        self._is_grabbing = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the camera.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the camera and release resources."""
        pass

    @abstractmethod
    def start_grabbing(self) -> bool:
        """
        Start continuous frame grabbing.

        Returns:
            True if grabbing started successfully
        """
        pass

    @abstractmethod
    def stop_grabbing(self) -> bool:
        """
        Stop continuous frame grabbing.

        Returns:
            True if grabbing stopped successfully
        """
        pass

    @abstractmethod
    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Grab a single frame.

        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        pass

    @abstractmethod
    def configure(self, settings: CameraSettings) -> bool:
        """
        Apply camera settings.

        Args:
            settings: Settings to apply

        Returns:
            True if settings applied successfully
        """
        pass

    @abstractmethod
    def get_info(self) -> CameraInfo:
        """
        Get camera information.

        Returns:
            CameraInfo object with device details
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> CameraCapabilities:
        """
        Get camera capabilities.

        Returns:
            CameraCapabilities object with available options
        """
        pass

    # Common interface methods

    @property
    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self._is_connected

    @property
    def is_grabbing(self) -> bool:
        """Check if camera is currently grabbing frames."""
        return self._is_grabbing

    @property
    def current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame (thread-safe)."""
        with self._frame_lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    @property
    def capabilities(self) -> CameraCapabilities:
        """Get camera capabilities."""
        return self.get_capabilities()

    @contextmanager
    def frame_lock(self):
        """Context manager for thread-safe frame access."""
        with self._frame_lock:
            yield

    # API compatibility aliases

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (alias for current_frame property)."""
        return self.current_frame

    def set_exposure_ms(self, exposure_ms: float) -> bool:
        """Set exposure time in milliseconds (alias for set_exposure)."""
        return self.set_exposure(exposure_ms)

    def set_gain_auto(self, mode: str) -> bool:
        """Set gain auto mode (alias for set_gain_mode)."""
        return self.set_gain_mode(mode)

    def set_white_balance_auto(self, mode: str) -> bool:
        """Set white balance auto mode (alias for set_white_balance_mode)."""
        return self.set_white_balance_mode(mode)

    def set_exposure(self, exposure_ms: float) -> bool:
        """
        Set camera exposure time.

        Args:
            exposure_ms: Exposure time in milliseconds

        Returns:
            True if successful
        """
        self.settings.exposure_ms = exposure_ms
        return self.configure(self.settings)

    def set_gain_mode(self, mode: str) -> bool:
        """
        Set camera gain mode.

        Args:
            mode: "Once", "Continuous", or "Off"

        Returns:
            True if successful
        """
        self.settings.gain_mode = mode
        return self.configure(self.settings)

    def set_white_balance_mode(self, mode: str) -> bool:
        """
        Set white balance mode.

        Args:
            mode: "Once", "Continuous", or "Off"

        Returns:
            True if successful
        """
        self.settings.white_balance_mode = mode
        return self.configure(self.settings)

    def set_framerate(self, framerate: float) -> bool:
        """
        Set capture framerate.

        Args:
            framerate: Frames per second

        Returns:
            True if successful
        """
        self.settings.framerate = framerate
        return self.configure(self.settings)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class BaslerCamera(CameraHAL):
    """Basler camera implementation using pypylon."""

    def __init__(self, settings: Optional[CameraSettings] = None, device_index: int = 0):
        """
        Initialize Basler camera.

        Args:
            settings: Camera settings
            device_index: Index of device to use (default: 0)
        """
        super().__init__(settings)

        if not PYPYLON_AVAILABLE:
            raise CameraError("pypylon library not available")

        self._device_index = device_index
        self._camera: Optional[pylon.InstantCamera] = None
        self._converter: Optional[pylon.ImageFormatConverter] = None
        self._grab_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def connect(self) -> bool:
        """Connect to Basler camera."""
        try:
            # Find available devices
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()

            if len(devices) == 0:
                logger.warning("No Basler cameras found")
                return False

            if self._device_index >= len(devices):
                logger.error(f"Device index {self._device_index} out of range (found {len(devices)} devices)")
                return False

            # Create camera instance
            self._camera = pylon.InstantCamera(
                tl_factory.CreateDevice(devices[self._device_index])
            )

            # Setup converter
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            # Open camera
            self._camera.Open()

            # Apply initial settings immediately so first frames are valid
            self.configure(self.settings)

            self._is_connected = True
            logger.info(f"Connected to Basler camera: {self._camera.GetDeviceInfo().GetModelName()}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Basler camera: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Basler camera."""
        try:
            self.stop_grabbing()

            if self._camera and self._camera.IsOpen():
                self._camera.Close()

            self._camera = None
            self._converter = None
            self._is_connected = False

            logger.info("Disconnected from Basler camera")

        except Exception as e:
            logger.error(f"Error disconnecting from Basler camera: {e}")

    def start_grabbing(self) -> bool:
        """Start continuous frame grabbing."""
        if not self._is_connected or not self._camera:
            logger.error("Camera not connected")
            return False

        try:
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._is_grabbing = True

            # Start grabbing thread
            self._stop_event.clear()
            self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._grab_thread.start()

            logger.info("Started Basler frame grabbing")
            return True

        except Exception as e:
            logger.error(f"Failed to start Basler grabbing: {e}")
            return False

    def stop_grabbing(self) -> bool:
        """Stop continuous frame grabbing."""
        try:
            self._stop_event.set()

            if self._grab_thread and self._grab_thread.is_alive():
                self._grab_thread.join(timeout=2.0)

            if self._camera and self._camera.IsGrabbing():
                self._camera.StopGrabbing()

            self._is_grabbing = False
            logger.info("Stopped Basler frame grabbing")
            return True

        except Exception as e:
            logger.error(f"Failed to stop Basler grabbing: {e}")
            return False

    def _grab_loop(self) -> None:
        """Frame grabbing loop (runs in separate thread)."""
        while not self._stop_event.is_set() and self._camera.IsGrabbing():
            try:
                grab_result = self._camera.RetrieveResult(
                    self.settings.timeout_ms,
                    pylon.TimeoutHandling_ThrowException
                )

                if grab_result.GrabSucceeded():
                    # Convert to BGR
                    image = self._converter.Convert(grab_result)
                    frame = image.GetArray()

                    # Store frame (thread-safe)
                    with self._frame_lock:
                        self._current_frame = frame

                grab_result.Release()

            except genicam.GenericException as e:
                if "Device has been removed" in str(e):
                    logger.error("Basler camera disconnected")
                    self._is_connected = False
                    break
            except Exception as e:
                logger.error(f"Basler grab loop error: {e}")
                break

    def grab_frame(self) -> Optional[np.ndarray]:
        """Grab a single frame."""
        if not self._is_connected or not self._camera:
            return None

        try:
            # Grab one frame
            grab_result = self._camera.GrabOne(self.settings.timeout_ms)

            if grab_result.GrabSucceeded():
                image = self._converter.Convert(grab_result)
                frame = image.GetArray()
                grab_result.Release()
                return frame

            return None

        except Exception as e:
            logger.error(f"Failed to grab Basler frame: {e}")
            return None

    def configure(self, settings: CameraSettings) -> bool:
        """Apply settings to Basler camera."""
        if not self._is_connected or not self._camera:
            return False

        try:
            self.settings = settings

            # Set exposure (convert milliseconds -> microseconds for Basler API)
            if settings.exposure_ms is not None:
                if self._camera.ExposureTime.GetAccessMode() == genicam.RW:
                    # Turn off auto exposure when manual exposure is specified
                    if hasattr(self._camera, 'ExposureAuto'):
                        try:
                            self._camera.ExposureAuto.SetValue('Off')
                        except Exception:
                            pass
                    # Basler expects microseconds; GUI supplies milliseconds
                    desired_us = float(settings.exposure_ms) * 1000.0
                    # Ensure exposure is within valid range (in microseconds)
                    min_exp = self._camera.ExposureTime.GetMin()
                    max_exp = self._camera.ExposureTime.GetMax()
                    exp_value = max(min_exp, min(desired_us, max_exp))
                    self._camera.ExposureTime.SetValue(exp_value)

            # Set gain mode
            if hasattr(self._camera, 'GainAuto'):
                gain_map = {"Once": "Once", "Continuous": "Continuous", "Off": "Off"}
                if settings.gain_mode in gain_map:
                    self._camera.GainAuto.SetValue(gain_map[settings.gain_mode])

            # Set white balance mode
            if hasattr(self._camera, 'BalanceWhiteAuto'):
                wb_map = {"Once": "Once", "Continuous": "Continuous", "Off": "Off"}
                if settings.white_balance_mode in wb_map:
                    self._camera.BalanceWhiteAuto.SetValue(wb_map[settings.white_balance_mode])

            logger.info("Basler camera settings applied")
            return True

        except Exception as e:
            logger.error(f"Failed to configure Basler camera: {e}")
            return False

    def get_info(self) -> CameraInfo:
        """Get Basler camera information."""
        if not self._camera:
            return CameraInfo("unknown", "unknown", "unknown", "basler")

        device_info = self._camera.GetDeviceInfo()

        return CameraInfo(
            device_id=device_info.GetFullName(),
            model_name=device_info.GetModelName(),
            serial_number=device_info.GetSerialNumber(),
            camera_type="basler"
        )

    def get_capabilities(self) -> CameraCapabilities:
        """Get Basler camera capabilities."""
        if not self._camera:
            return CameraCapabilities(
                name="Unknown Basler Camera",
                gain_auto_options=["Off", "Once", "Continuous"],
                white_balance_auto_options=["Off", "Once", "Continuous"]
            )

        # Get actual camera name
        device_info = self._camera.GetDeviceInfo()
        name = device_info.GetModelName()

        # Determine available gain options
        gain_options = []
        if hasattr(self._camera, 'GainAuto'):
            try:
                # Try to get available enum values
                gain_options = ["Off", "Once", "Continuous"]
            except Exception:
                gain_options = ["Off"]
        else:
            gain_options = ["Off"]

        # Determine available white balance options
        wb_options = []
        if hasattr(self._camera, 'BalanceWhiteAuto'):
            try:
                wb_options = ["Off", "Once", "Continuous"]
            except Exception:
                wb_options = ["Off"]
        else:
            wb_options = ["Off"]

        return CameraCapabilities(
            name=name,
            gain_auto_options=gain_options,
            white_balance_auto_options=wb_options
        )


class CameraFactory:
    """
    Factory for creating camera instances.

    This class handles camera detection and instantiation for Basler cameras.
    """

    @staticmethod
    def create_camera(
        device_index: int = 0,
        settings: Optional[CameraSettings] = None
    ) -> Optional[CameraHAL]:
        """
        Create a Basler camera instance.

        Args:
            device_index: Device index to use (default: 0)
            settings: Camera settings to apply

        Returns:
            BaslerCamera instance or None if no camera available
        """
        if not PYPYLON_AVAILABLE:
            logger.error("pypylon library not available - Basler camera support requires pypylon")
            return None

        try:
            camera = BaslerCamera(settings=settings, device_index=device_index)
            if camera.connect():
                logger.info("Created Basler camera instance")
                return camera
            else:
                logger.warning("Failed to connect to Basler camera")
                return None
        except Exception as e:
            logger.error(f"Basler camera initialization failed: {e}")
            return None

    @staticmethod
    def list_available_cameras() -> List[str]:
        """
        List all available Basler cameras.

        Returns:
            List of Basler camera names/IDs
        """
        cameras = []

        # Check for Basler cameras
        if PYPYLON_AVAILABLE:
            try:
                tl_factory = pylon.TlFactory.GetInstance()
                devices = tl_factory.EnumerateDevices()
                for device in devices:
                    cameras.append(f"Basler: {device.GetModelName()} ({device.GetSerialNumber()})")
            except Exception as e:
                logger.warning(f"Error enumerating Basler cameras: {e}")

        return cameras
