"""
Fish measurement logic module.

This package contains the core measurement logic for fish length analysis,
including point utilities, path elements, measurement instances, and support modules.

Modern Python 3.13 compatible implementation with proper type hints,
error handling, and modular design following Pythonic conventions.
"""

from .point_utils import (
    add_thick_binary_dots,
    is_point_in_neighborhood,
    contains_mutual_points,
    calculate_euclidean_distance,
    optimize_path,
    calculate_path_length,
)

from .long_path_element import LongPathElement
from .measurement_config import MeasurementConfig, get_global_config, reset_global_config
from .folder_manager import FolderManager

# Orchestration and state management
from .measurement_state_machine import (
    MeasurementStateMachine,
    MeasurementState,
    StateTransition
)
from .measurement_orchestrator import (
    MeasurementOrchestrator,
    CalibrationResult,
    TrainingResult,
    ProcessingResult
)

# Processing coordination (NEW - splitting God Object)
from .processing_coordinator import ProcessingCoordinator
from .statistics_collector import StatisticsCollector

# Protocol interfaces for dependency injection (NEW)
from . import protocols

# Vision-dependent modules (now re-enabled with proper integration)
from .measurer_instance import MeasurerInstance
from .measurement_statistics import MeasurementStatistics
from .image_watermarker import ImageWatermarker

# Version information
__version__ = "2.0.0"
__author__ = "Fish Measurer Development Team"
__description__ = "Modern fish measurement logic with Python 3.13 compatibility"

# Export all public classes and functions
__all__ = [
    # Core classes (vision-independent)
    "LongPathElement",
    "MeasurementConfig",
    "FolderManager",

    # Orchestration and state management (NEW)
    "MeasurementStateMachine",
    "MeasurementState",
    "StateTransition",
    "MeasurementOrchestrator",
    "CalibrationResult",
    "TrainingResult",
    "ProcessingResult",

    # Processing coordination (NEW)
    "ProcessingCoordinator",
    "StatisticsCollector",
    "protocols",

    # Vision-dependent classes (now integrated)
    "MeasurerInstance",
    "MeasurementStatistics",
    "ImageWatermarker",

    # Point utilities
    "add_thick_binary_dots",
    "is_point_in_neighborhood",
    "contains_mutual_points",
    "calculate_euclidean_distance",
    "optimize_path",
    "calculate_path_length",

    # Configuration utilities
    "get_global_config",
    "reset_global_config",

    # Module metadata
    "__version__",
    "__author__",
    "__description__",
]