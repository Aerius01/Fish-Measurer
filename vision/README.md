# Vision Module - Fish Measurer

## Overview

This refactored vision module provides modern, modular computer vision functionality for fish measurement. The code has been completely restructured from the original 2020 implementation to follow current best practices, improve maintainability, and enhance performance.

## Key Improvements

### 🔧 **Modern Architecture**
- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Type Safety**: Full type hints throughout the codebase
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Resource Management**: Proper cleanup and context managers
- **Thread Safety**: Safe concurrent operations where needed

### 📦 **Modular Design**
The monolithic `ProcessingInstance.py` (723 lines) has been broken down into focused modules:

- `camera_manager.py` - Camera hardware interface and frame capture
- `image_processor.py` - Core image processing operations  
- `filament_analyzer.py` - FilFinder-based skeletal analysis
- `path_analyzer.py` - Longest path construction and analysis
- `aruco_detector.py` - ArUco marker detection and calibration
- `display_manager.py` - UI display and annotation management
- `fish_processor.py` - Main coordinator for complete measurement pipeline

### 🚀 **Updated Dependencies**
- **OpenCV 4.5+**: Updated ArUco API usage with fallbacks for compatibility
- **Modern Python**: Uses `dataclasses`, `pathlib`, `typing`, and other modern features
- **Better Logging**: Structured logging throughout the application

## Module Structure

```
vision/
├── __init__.py              # Module interface and public API
├── README.md               # This documentation
├── camera_manager.py       # Camera hardware management
├── image_processor.py      # Core image processing
├── filament_analyzer.py    # FilFinder operations
├── path_analyzer.py        # Path construction algorithms
├── aruco_detector.py       # ArUco marker detection
├── display_manager.py      # Display and visualization
├── fish_processor.py       # Main processing coordinator
├── aruco_generator.py      # ArUco marker generation utility
└── example_usage.py        # Usage examples and demonstrations
```

## Quick Start

### Basic Usage

```python
from vision import FishProcessor, CameraManager

# Initialize components
camera = CameraManager()
processor = FishProcessor(output_folder="results")

# Process a fish measurement
if camera.connected:
    raw_frames, binary_frames = camera.get_frames(3)
    
    for i, (raw, binary) in enumerate(zip(raw_frames, binary_frames)):
        result = processor.process_fish(f"fish_{i}", raw, binary)
        
        if result.success:
            print(f"Total length: {result.total_length_pixels:.2f} pixels")
            processor.save_results(result)
        else:
            print(f"Processing failed: {result.error_message}")
```

### Advanced Usage

```python
from vision import (
    CameraManager, CameraConfig, 
    ArUcoDetector, DisplayManager,
    FishProcessor
)

# Configure camera
config = CameraConfig(framerate=60, number_of_frames=5)
camera = CameraManager(config)

# Setup ArUco detection
aruco = ArUcoDetector()
display = DisplayManager()

# Setup processing pipeline
processor = FishProcessor()

# Process with custom frame processing
def process_frame(raw_frame):
    # Custom background subtraction logic
    return processed_frame

camera.set_frame_processor(process_frame)

# Get calibrated measurements
frame = camera.current_frame
if frame is not None:
    markers = aruco.detect_markers(frame)
    if markers:
        calibration = aruco.calculate_calibration(markers)
        length_cm = aruco.convert_pixels_to_length(100)  # Convert 100 pixels to cm
```

## API Reference

### FishProcessor

Main coordinator for the measurement pipeline.

```python
class FishProcessor:
    def __init__(self, output_folder: Optional[Path] = None)
    def process_fish(self, fish_id: str, raw_frame: np.ndarray, 
                    binarized_frame: np.ndarray) -> FishMeasurementResult
    def process_multiple_frames(self, fish_id: str, raw_frames: List[np.ndarray], 
                               binarized_frames: List[np.ndarray]) -> List[FishMeasurementResult]
    def save_results(self, result: FishMeasurementResult, save_images: bool = True) -> bool
```

### CameraManager

Modern camera interface with thread-safe operations.

```python
class CameraManager:
    def __init__(self, config: Optional[CameraConfig] = None)
    def start_grabbing(self) -> None
    def stop_grabbing(self) -> bool
    def get_frames(self, count: int) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]
    def set_frame_processor(self, processor: Callable) -> None
    def cleanup(self) -> None
```

### ArUcoDetector

ArUco marker detection with modern OpenCV API.

```python
class ArUcoDetector:
    def __init__(self, dictionary_type: int = cv2.aruco.DICT_4X4_50)
    def detect_markers(self, image: np.ndarray) -> List[ArUcoMarker]
    def calculate_calibration(self, markers: List[ArUcoMarker]) -> Optional[CalibrationData]
    def convert_pixels_to_length(self, pixels: float) -> Optional[float]
    def draw_markers(self, image: np.ndarray, markers: List[ArUcoMarker]) -> np.ndarray
```

## Migration Guide

### Modern Usage Examples

**Camera Management:**
```python
from vision import CameraManager

camera = CameraManager()
camera.set_frame_processor(your_processing_function)
raw_frames, binary_frames = camera.get_frames(3)
```

**Fish Processing:**
```python
from vision import FishProcessor

processor = FishProcessor(output_folder=output)
result = processor.process_fish(fish_id, raw_frame, binary_frame)
if result.success:
    print(f"Length: {result.total_length_pixels}")
```

## Error Handling

The refactored code includes comprehensive error handling:

```python
from vision import CameraError, FishProcessor

try:
    processor = FishProcessor()
    result = processor.process_fish("fish_1", raw_frame, binary_frame)
    
    if not result.success:
        print(f"Processing failed: {result.error_message}")
        for log_entry in result.processing_log:
            print(f"  {log_entry}")
            
except CameraError as e:
    print(f"Camera error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Improvements

- **Reduced Memory Usage**: Better memory management and cleanup
- **Faster Processing**: Optimized algorithms and reduced redundancy
- **Concurrent Operations**: Thread-safe frame capture and processing
- **Efficient Data Structures**: Use of NumPy arrays and efficient algorithms

## Testing

```python
# Basic functionality test
from vision import get_version_info, create_fish_processor

print(get_version_info())

processor = create_fish_processor("test_output")
# Test with sample data...
```


## Dependencies

- OpenCV 4.5+ (with ArUco support)
- NumPy 1.20+
- Pillow 8.0+
- scikit-image 0.19+
- FilFinder 1.7+
- Astropy 5.0+
- pypylon 1.8+ (for Basler cameras)

## Contributing

When contributing to this module:

1. Follow type hints for all functions
2. Include comprehensive docstrings
3. Add appropriate error handling
4. Write unit tests for new functionality
5. Update this README for API changes

## License

This module is part of the Fish Measurer project. See the main project LICENSE file for details.
