"""
Modern camera management module for Fish Measurer application.

This module provides camera management functionality with improved error handling,
type safety, and separation of concerns.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional, List, Tuple, Callable, Any
import logging
from dataclasses import dataclass
from contextlib import contextmanager

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import pypylon
    from pypylon import pylon, genicam
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False
    logging.warning("pypylon not available - camera functionality will be limited")


@dataclass
class CameraConfig:
    """Configuration for camera operations."""
    framerate: int = 30
    number_of_frames: int = 3
    timeout_ms: int = 5000
    pixel_format: Any = None
    bit_alignment: Any = None
    
    def __post_init__(self):
        if PYPYLON_AVAILABLE and self.pixel_format is None:
            self.pixel_format = pylon.PixelType_BGR8packed
        if PYPYLON_AVAILABLE and self.bit_alignment is None:
            self.bit_alignment = pylon.OutputBitAlignment_MsbAligned


class CameraError(Exception):
    """Custom exception for camera-related errors."""
    pass


class CameraManager:
    """
    Modern camera management class with proper resource management and error handling.
    
    This class handles camera initialization, frame capture, and resource cleanup
    with thread-safe operations and proper error handling.
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.logger = logging.getLogger(__name__)
        
        # Camera state
        self._cameras: Optional[Any] = None
        self._current_camera: Optional[Any] = None
        self._converter: Optional[Any] = None
        self._connected = False
        
        # Frame state
        self._current_frame: Optional[np.ndarray] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._binarized_frame: Optional[np.ndarray] = None
        self._new_frame = False
        
        # Threading
        self._frame_lock = threading.Lock()
        self._grab_thread: Optional[threading.Thread] = None
        self._stop_grabbing = threading.Event()
        
        # Callbacks
        self._frame_processor: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None
        
        # Initialize camera if available
        if PYPYLON_AVAILABLE:
            self._initialize_cameras()
    
    @property
    def connected(self) -> bool:
        """Check if camera is connected and operational."""
        return self._connected
    
    @property
    def current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame (thread-safe)."""
        with self._frame_lock:
            return self._current_frame.copy() if self._current_frame is not None else None
    
    def set_frame_processor(self, processor: Callable[[np.ndarray], Optional[np.ndarray]]) -> None:
        """Set a callback function to process raw frames."""
        self._frame_processor = processor
    
    def _initialize_cameras(self) -> None:
        """Initialize available cameras."""
        if not PYPYLON_AVAILABLE:
            raise CameraError("pypylon not available")
        
        try:
            # Setup converter
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = self.config.pixel_format
            self._converter.OutputBitAlignment = self.config.bit_alignment
            
            # Find cameras
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            self.logger.info(f"Found {len(devices)} camera devices")
            
            if len(devices) == 0:
                self._connected = False
                return
            
            # Create camera array
            self._cameras = pylon.InstantCameraArray(len(devices))
            for i, camera in enumerate(self._cameras):
                camera.Attach(tl_factory.CreateDevice(devices[i]))
            
            # Set current camera to first available
            self._current_camera = self._cameras[0]
            self._connected = True
            
            # Start grabbing
            self.start_grabbing()
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            self._connected = False
            raise CameraError(f"Failed to initialize cameras: {e}")
    
    def start_grabbing(self) -> None:
        """Start the frame grabbing process."""
        if not self._connected or not self._current_camera:
            raise CameraError("No camera connected")
        
        try:
            self._current_camera.Open()
            self._current_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            
            # Start grabbing thread
            self._stop_grabbing.clear()
            self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._grab_thread.start()
            
            self.logger.info("Started frame grabbing")
            
        except Exception as e:
            self.logger.error(f"Failed to start grabbing: {e}")
            self._connected = False
            raise CameraError(f"Failed to start grabbing: {e}")
    
    def stop_grabbing(self) -> bool:
        """Stop the frame grabbing process."""
        try:
            self._stop_grabbing.set()
            
            if self._grab_thread and self._grab_thread.is_alive():
                self._grab_thread.join(timeout=2.0)
            
            if self._current_camera and self._current_camera.IsGrabbing():
                self._current_camera.StopGrabbing()
                self._current_camera.Close()
            
            self.logger.info("Stopped frame grabbing")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop grabbing: {e}")
            return False
    
    def _grab_loop(self) -> None:
        """Main frame grabbing loop (runs in separate thread)."""
        while not self._stop_grabbing.is_set() and self._current_camera.IsGrabbing():
            try:
                grab_result = self._current_camera.RetrieveResult(
                    self.config.timeout_ms, 
                    pylon.TimeoutHandling_ThrowException
                )
                
                if grab_result is not None and grab_result.GrabSucceeded():
                    # Convert image
                    image = self._converter.Convert(grab_result)
                    img_array = image.GetArray()
                    frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    grab_result.Release()
                    
                    # Process frame
                    self._process_frame(frame)
                
            except genicam.GenericException as e:
                if ("Device has been removed" in str(e) or 
                    "No grab result data is referenced" in str(e)):
                    self.logger.error(f"Camera disconnected: {e}")
                    self._connected = False
                    break
            except Exception as e:
                self.logger.error(f"Grab loop error: {e}")
                break
    
    def _process_frame(self, raw_frame: np.ndarray) -> None:
        """Process a new frame (thread-safe)."""
        with self._frame_lock:
            self._raw_frame = raw_frame.copy()
            
            # Apply frame processor if available
            if self._frame_processor:
                processed = self._frame_processor(raw_frame)
                self._binarized_frame = processed
                self._current_frame = processed if processed is not None else raw_frame
            else:
                self._binarized_frame = None
                self._current_frame = raw_frame
            
            self._new_frame = True
    
    def change_camera(self, camera_index: int) -> bool:
        """Change to a different camera."""
        if not self._cameras or camera_index >= len(self._cameras):
            return False
        
        try:
            # Stop current grabbing
            if not self.stop_grabbing():
                return False
            
            # Switch camera
            self._current_camera = self._cameras[camera_index]
            
            # Restart grabbing
            self.start_grabbing()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to change camera: {e}")
            self._connected = False
            return False
    
    def get_frames(self, count: int) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        """
        Capture a specific number of frames.
        
        Returns:
            Tuple of (raw_frames, binarized_frames)
        """
        raw_frames = []
        binarized_frames = []
        
        while len(raw_frames) < count:
            with self._frame_lock:
                if self._new_frame:
                    self._new_frame = False
                    if self._raw_frame is not None:
                        raw_frames.append(self._raw_frame.copy())
                        binarized_frames.append(
                            self._binarized_frame.copy() if self._binarized_frame is not None else None
                        )
            
            time.sleep(1.0 / self.config.framerate)
        
        return raw_frames, binarized_frames
    
    @contextmanager
    def frame_lock(self):
        """Context manager for frame access."""
        with self._frame_lock:
            yield
    
    def cleanup(self) -> None:
        """Clean up camera resources."""
        self.stop_grabbing()
        
        if self._cameras:
            try:
                for camera in self._cameras:
                    if camera.IsOpen():
                        camera.Close()
            except Exception as e:
                self.logger.error(f"Error during camera cleanup: {e}")
        
        self.logger.info("Camera cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


