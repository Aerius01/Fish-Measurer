"""Simplified camera controller for the GUI."""

from typing import List, Optional, Any
from dataclasses import dataclass
import threading

from .config import CameraSettings
from .events import publish_error, publish_camera_changed


@dataclass
class CameraInfo:
    """Information about a camera."""
    id: str
    model_name: str
    serial_number: str
    is_available: bool = True


class CameraController:
    """Simplified controller for camera operations."""
    
    def __init__(self, camera_manager=None):
        # Use provided camera manager or create a new one
        if camera_manager is not None:
            self.camera_manager = camera_manager
        else:
            # Import here to avoid circular imports
            from vision import CameraManager
            self.camera_manager = CameraManager()
        
        self.current_settings = CameraSettings()
    
    def initialize(self) -> bool:
        """Initialize camera system."""
        return self.camera_manager.connected
    
    def get_available_cameras(self) -> List[CameraInfo]:
        """Get list of available cameras."""
        cameras = []
        if hasattr(self.camera_manager, '_cameras') and self.camera_manager._cameras:
            for i, cam in enumerate(self.camera_manager._cameras):
                try:
                    info = CameraInfo(
                        id=str(i),
                        model_name=str(cam.GetDeviceInfo().GetModelName()),
                        serial_number=str(cam.GetDeviceInfo().GetSerialNumber())
                    )
                    cameras.append(info)
                except Exception:
                    pass
        return cameras

    def get_current_camera_info(self) -> Optional[CameraInfo]:
        """Get information about the currently connected camera."""
        try:
            if hasattr(self.camera_manager, '_current_camera') and self.camera_manager._current_camera:
                cam = self.camera_manager._current_camera
                info = CameraInfo(
                    id="0",
                    model_name=str(cam.GetDeviceInfo().GetModelName()),
                    serial_number=str(cam.GetDeviceInfo().GetSerialNumber())
                )
                return info
            return None
        except Exception as e:
            publish_error("camera_info", f"Failed to read current camera info: {str(e)}")
            return None
    
    def connect_camera(self, camera_id: str) -> bool:
        """Connect to a specific camera."""
        try:
            camera_index = int(camera_id)
            success = self.camera_manager.change_camera(camera_index)
            if success:
                cameras = self.get_available_cameras()
                camera_info = next((c for c in cameras if c.id == camera_id), None)
                if camera_info:
                    publish_camera_changed(
                        camera_id,
                        {
                            "model_name": camera_info.model_name,
                            "serial_number": camera_info.serial_number
                        }
                    )
            return success
        except Exception as e:
            publish_error("camera_connection", f"Failed to connect to camera: {str(e)}")
            return False
    
    def apply_settings(self, settings: CameraSettings) -> bool:
        """Apply camera settings."""
        try:
            if self.camera_manager._current_camera:
                # Apply exposure (minimum 42ms)
                min_exposure = max(settings.exposure_ms, 42)
                self.camera_manager._current_camera.ExposureTime.SetValue(min_exposure)
                
                # Apply gain
                self.camera_manager._current_camera.GainAuto.SetValue(settings.gain_mode)
                
                # Apply white balance
                self.camera_manager._current_camera.BalanceWhiteAuto.SetValue(settings.white_balance_mode)
                
                # Apply framerate
                self.camera_manager._current_camera.AcquisitionFrameRateEnable.SetValue(True)
                self.camera_manager._current_camera.AcquisitionFrameRate.SetValue(settings.framerate_fps)
                
                self.current_settings = settings
                return True
            return False
        except Exception as e:
            publish_error("camera_settings", f"Failed to apply settings: {str(e)}")
            return False
    
    def update_exposure(self, exposure_ms: float) -> bool:
        """Update exposure setting."""
        self.current_settings.exposure_ms = exposure_ms
        return self.apply_settings(self.current_settings)
    
    def update_gain_mode(self, gain_mode: str) -> bool:
        """Update gain mode setting."""
        self.current_settings.gain_mode = gain_mode
        return self.apply_settings(self.current_settings)
    
    def update_white_balance_mode(self, wb_mode: str) -> bool:
        """Update white balance mode setting."""
        self.current_settings.white_balance_mode = wb_mode
        return self.apply_settings(self.current_settings)
    
    def update_framerate(self, framerate_fps: float) -> bool:
        """Update framerate setting."""
        self.current_settings.framerate_fps = framerate_fps
        return self.apply_settings(self.current_settings)
    
    def get_current_frame(self) -> Optional[Any]:
        """Get the current frame from the camera."""
        try:
            if not self.camera_manager.connected:
                return None
            return self.camera_manager.current_frame
        except Exception as e:
            publish_error("frame_access", f"Failed to get current frame: {str(e)}")
            return None
    
    def is_camera_connected(self) -> bool:
        """Check if camera is connected."""
        return self.camera_manager.connected
    
    def shutdown(self) -> None:
        """Shutdown camera controller."""
        try:
            self.camera_manager.cleanup()
        except Exception:
            pass
