"""Event system for loose coupling between GUI components."""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum, auto
import weakref


class EventType(Enum):
    """Enumeration of application events."""
    
    # Camera events
    CAMERA_CHANGED = auto()
    CAMERA_SETTINGS_CHANGED = auto()
    CAMERA_DISCONNECTED = auto()
    
    # Processing events
    BACKGROUND_TRAINING_STARTED = auto()
    BACKGROUND_TRAINING_COMPLETED = auto()
    BACKGROUND_TRAINING_FAILED = auto()
    
    ANALYSIS_STARTED = auto()
    ANALYSIS_PROGRESS = auto()
    ANALYSIS_COMPLETED = auto()
    ANALYSIS_FAILED = auto()
    ANALYSIS_CANCELLED = auto()
    
    # Configuration events
    OUTPUT_SETTINGS_CHANGED = auto()
    PROCESSING_SETTINGS_CHANGED = auto()
    
    # UI state events
    APP_STATE_CHANGED = auto()
    SETTINGS_LOCKED = auto()
    SETTINGS_UNLOCKED = auto()
    
    # Error events
    ERROR_OCCURRED = auto()


@dataclass
class Event:
    """Represents an application event."""
    event_type: EventType
    data: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


EventHandler = Callable[[Event], None]


class EventBus:
    """Simple event bus for application-wide communication."""
    
    def __init__(self):
        # Use WeakSet to automatically remove dead references
        self._handlers: Dict[EventType, List[weakref.ReferenceType]] = {}
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        # Use weak reference to prevent memory leaks
        self._handlers[event_type].append(weakref.ref(handler))
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe from an event type."""
        if event_type not in self._handlers:
            return
        
        # Remove the handler (compare the actual objects, not weak references)
        self._handlers[event_type] = [
            ref for ref in self._handlers[event_type]
            if ref() is not None and ref() is not handler
        ]
    
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        if event.event_type not in self._handlers:
            return
        
        # Clean up dead references and call active handlers
        active_handlers = []
        for handler_ref in self._handlers[event.event_type]:
            handler = handler_ref()
            if handler is not None:
                active_handlers.append(handler_ref)
                try:
                    handler(event)
                except Exception as e:
                    # Log error but don't let one handler break others
                    print(f"Error in event handler: {e}")
        
        # Update the list to remove dead references
        self._handlers[event.event_type] = active_handlers
    
    def publish_event(self, 
                     event_type: EventType, 
                     data: Optional[Dict[str, Any]] = None,
                     source: Optional[str] = None) -> None:
        """Convenience method to publish an event."""
        event = Event(event_type=event_type, data=data, source=source)
        self.publish(event)


# Global event bus instance
event_bus = EventBus()


# Convenience functions for common events
def publish_camera_changed(camera_id: str, camera_info: Dict[str, Any]) -> None:
    """Publish camera changed event."""
    event_bus.publish_event(
        EventType.CAMERA_CHANGED,
        data={"camera_id": camera_id, "camera_info": camera_info},
        source="camera_controller"
    )


def publish_settings_changed(setting_type: str, settings: Dict[str, Any]) -> None:
    """Publish settings changed event."""
    if setting_type == "camera":
        event_type = EventType.CAMERA_SETTINGS_CHANGED
    elif setting_type == "output":
        event_type = EventType.OUTPUT_SETTINGS_CHANGED
    elif setting_type == "processing":
        event_type = EventType.PROCESSING_SETTINGS_CHANGED
    else:
        return
    
    event_bus.publish_event(
        event_type,
        data=settings,
        source="settings_panel"
    )


def publish_app_state_changed(old_state: int, new_state: int) -> None:
    """Publish application state change."""
    event_bus.publish_event(
        EventType.APP_STATE_CHANGED,
        data={"old_state": old_state, "new_state": new_state},
        source="app_controller"
    )


def publish_error(error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Publish error event."""
    data = {"error_type": error_type, "message": message}
    if details:
        data.update(details)
    
    event_bus.publish_event(
        EventType.ERROR_OCCURRED,
        data=data,
        source="error_handler"
    )


def publish_analysis_progress(current: int, total: int, stage: str = "") -> None:
    """Publish analysis progress event."""
    event_bus.publish_event(
        EventType.ANALYSIS_PROGRESS,
        data={"current": current, "total": total, "stage": stage},
        source="analysis_controller"
    )
