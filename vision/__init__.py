"""
Vision module for Fish Measurer application.

This module provides modern, modular computer vision functionality for fish measurement
including camera management, image processing, filament analysis, and measurement extraction.

The module is organized into specialized components:
- camera_manager: Camera hardware interface and frame capture
- image_processor: Core image processing operations
- filament_analyzer: FilFinder-based skeletal analysis
- path_analyzer: Longest path construction and analysis
- aruco_detector: ArUco marker detection and calibration
- display_manager: UI display and annotation management
- fish_processor: Main coordinator for complete measurement pipeline

"""

from __future__ import annotations

import logging

# Configure logging for the vision module
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Main processing components
from .fish_processor import FishProcessor, FishMeasurementResult
from .camera_manager import CameraManager, CameraConfig, CameraError
from .image_processor import ImageProcessor, ProcessingResult
from .filament_analyzer import FilamentAnalyzer, FilamentProperties, FilamentAnalysisResult
from .path_analyzer import PathAnalyzer, PathAnalysisResult
from .aruco_detector import ArUcoDetector, ArUcoMarker, CalibrationData
from .display_manager import DisplayManager

# Utility functions
from .aruco_generator import generate_aruco_marker, generate_marker_set

# Version information
__version__ = "2.0.0"
__author__ = "Fish Measurer Team"

# Public API
__all__ = [
    # Main classes
    'FishProcessor',
    'CameraManager', 
    'ImageProcessor',
    'FilamentAnalyzer',
    'PathAnalyzer',
    'ArUcoDetector',
    'DisplayManager',
    
    # Result classes
    'FishMeasurementResult',
    'ProcessingResult',
    'FilamentAnalysisResult',
    'PathAnalysisResult',
    'FilamentProperties',
    'ArUcoMarker',
    'CalibrationData',
    'CameraConfig',
    
    # Exceptions
    'CameraError',
    
    # Utility functions
    'generate_aruco_marker',
    'generate_marker_set',
]


def create_fish_processor(output_folder=None):
    """
    Create a configured FishProcessor instance.
    
    Args:
        output_folder: Optional output folder path
        
    Returns:
        Configured FishProcessor instance
    """
    return FishProcessor(output_folder=output_folder)


def create_camera_manager(config=None):
    """
    Create a configured CameraManager instance.
    
    Args:
        config: Optional CameraConfig instance
        
    Returns:
        Configured CameraManager instance
    """
    return CameraManager(config=config)


def get_version_info():
    """
    Get version and component information.
    
    Returns:
        Dictionary with version information
    """
    return {
        'version': __version__,
        'author': __author__,
        'components': [
            'FishProcessor - Main measurement coordinator',
            'CameraManager - Camera hardware interface',
            'ImageProcessor - Core image processing',
            'FilamentAnalyzer - FilFinder-based analysis',
            'PathAnalyzer - Longest path construction',
            'ArUcoDetector - Marker detection and calibration',
            'DisplayManager - UI and visualization',
        ]
    }


# Module initialization
def _setup_logging():
    """Setup default logging configuration for the vision module."""
    logger = logging.getLogger(__name__)
    
    # Only add handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


# Initialize logging when module is imported
_setup_logging()
