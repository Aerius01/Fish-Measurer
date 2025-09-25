"""Configuration management for the Fish Measurer application."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import os


# Constants
VALID_CAMERA_MODES = {"Once", "Continuous", "Off"}
VALID_OUTPUT_FORMATS = {".jpeg", ".png", ".tiff"}


@dataclass
class CameraSettings:
    """Camera configuration settings."""
    exposure_ms: float = 2000.0
    gain_mode: str = "Once"  # Once, Continuous, Off
    white_balance_mode: str = "Off"  # Once, Continuous, Off
    framerate_fps: float = 30.0
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if self.exposure_ms < 42:
            self.exposure_ms = 42
        
        if self.gain_mode not in VALID_CAMERA_MODES:
            self.gain_mode = "Once"
        if self.white_balance_mode not in VALID_CAMERA_MODES:
            self.white_balance_mode = "Off"


@dataclass
class OutputSettings:
    """Output configuration settings."""
    folder_path: Path = field(default_factory=lambda: Path.cwd())
    file_format: str = ".jpeg"  # .jpeg, .png, .tiff
    fish_id: str = ""
    additional_text: str = ""
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if self.file_format not in VALID_OUTPUT_FORMATS:
            self.file_format = ".jpeg"
        
        # Ensure folder_path is a Path object
        if isinstance(self.folder_path, str):
            self.folder_path = Path(self.folder_path)


@dataclass
class ProcessingSettings:
    """Processing configuration settings."""
    include_shadows: bool = True  # True = threshold 100, False = threshold 200
    number_of_frames: int = 10
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if self.number_of_frames < 3:
            self.number_of_frames = 3
    
    @property
    def threshold(self) -> float:
        """Get threshold value based on shadow setting."""
        return 100.0 if self.include_shadows else 200.0


@dataclass
class AppConfig:
    """Main application configuration."""
    camera: CameraSettings = field(default_factory=CameraSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    
    # UI state
    current_state: int = 0  # 0: base, 1: background trained, 2: running analysis
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'camera': {
                'exposure_ms': self.camera.exposure_ms,
                'gain_mode': self.camera.gain_mode,
                'white_balance_mode': self.camera.white_balance_mode,
                'framerate_fps': self.camera.framerate_fps,
            },
            'output': {
                'folder_path': str(self.output.folder_path),
                'file_format': self.output.file_format,
                'fish_id': self.output.fish_id,
                'additional_text': self.output.additional_text,
            },
            'processing': {
                'include_shadows': self.processing.include_shadows,
                'number_of_frames': self.processing.number_of_frames,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'camera' in data:
            camera_data = data['camera']
            config.camera = CameraSettings(
                exposure_ms=camera_data.get('exposure_ms', 2000.0),
                gain_mode=camera_data.get('gain_mode', 'Once'),
                white_balance_mode=camera_data.get('white_balance_mode', 'Off'),
                framerate_fps=camera_data.get('framerate_fps', 30.0)
            )
        
        if 'output' in data:
            output_data = data['output']
            config.output = OutputSettings(
                folder_path=Path(output_data.get('folder_path', os.getcwd())),
                file_format=output_data.get('file_format', '.jpeg'),
                fish_id=output_data.get('fish_id', ''),
                additional_text=output_data.get('additional_text', '')
            )
        
        if 'processing' in data:
            processing_data = data['processing']
            config.processing = ProcessingSettings(
                include_shadows=processing_data.get('include_shadows', True),
                number_of_frames=processing_data.get('number_of_frames', 10)
            )
        
        return config
