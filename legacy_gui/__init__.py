"""Modern GUI package for Fish Measurer application.

This package provides a modular, event-driven GUI architecture with:
- Configuration management with dataclasses
- Event-driven component communication
- Validated input widgets with type safety
- Camera controller abstraction layer
- Separation of concerns following MVC pattern
"""

from .app import FishMeasurerApplication
from .config import AppConfig, CameraSettings, OutputSettings, ProcessingSettings
from .events import EventType, Event, event_bus
from .widgets import ValidatedEntry, NumericValidator

__all__ = [
    'FishMeasurerApplication',
    'AppConfig',
    'CameraSettings',
    'OutputSettings', 
    'ProcessingSettings',
    'EventType',
    'Event',
    'event_bus',
    'ValidatedEntry',
    'NumericValidator'
]
