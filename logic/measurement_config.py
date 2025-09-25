"""
Configuration management for fish measurement system.

This module handles all configuration state for the measurement system,
replacing the previous class variable approach with a cleaner singleton pattern.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import threading
from pathlib import Path


@dataclass
class MeasurementConfig:
    """
    Thread-safe configuration manager for measurement parameters.
    
    This class centralizes all measurement configuration and state management,
    replacing the previous approach of using class variables.
    """
    
    # Measurement parameters
    threshold: Optional[int] = None
    fish_id: Optional[str] = None
    additional_text: Optional[str] = None
    
    # Error tracking
    errors: Dict[str, List[str]] = field(default_factory=lambda: {"interrupt": []})
    
    # Process control
    stop_requested: bool = False
    
    # Threading state
    processing_frame: Optional[int] = None
    trial_count: int = 0
    completed_threads: int = 0
    
    # Thread lock for concurrent access
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    
    def get_threshold(self) -> int:
        """Get the current threshold value."""
        with self._lock:
            return self.threshold or 50  # Default threshold
    
    def set_threshold(self, value: int) -> None:
        """Set the threshold value."""
        if not isinstance(value, int) or value < 0 or value > 255:
            raise ValueError("Threshold must be an integer between 0 and 255")
        
        with self._lock:
            self.threshold = value
    
    def get_fish_id(self) -> Optional[str]:
        """Get the current fish ID."""
        with self._lock:
            return self.fish_id
    
    def set_fish_id(self, fish_id: Optional[str]) -> None:
        """Set the fish ID."""
        with self._lock:
            self.fish_id = fish_id.strip() if fish_id else None
    
    def get_additional_text(self) -> Optional[str]:
        """Get additional text for watermarking."""
        with self._lock:
            return self.additional_text
    
    def set_additional_text(self, text: Optional[str]) -> None:
        """Set additional text for watermarking."""
        with self._lock:
            self.additional_text = text.strip() if text else None
    
    def should_stop(self) -> bool:
        """Check if processing should stop."""
        with self._lock:
            return self.stop_requested
    
    def request_stop(self) -> None:
        """Request processing to stop."""
        with self._lock:
            self.stop_requested = True
    
    def reset_stop(self) -> None:
        """Reset the stop request."""
        with self._lock:
            self.stop_requested = False
    
    def get_processing_frame(self) -> Optional[int]:
        """Get the currently processing frame number."""
        with self._lock:
            return self.processing_frame
    
    def set_processing_frame(self, frame_num: Optional[int]) -> None:
        """Set the currently processing frame number."""
        with self._lock:
            self.processing_frame = frame_num
    
    def get_trial_count(self) -> int:
        """Get the number of successful trials."""
        with self._lock:
            return self.trial_count
    
    def set_trial_count(self, count: int) -> None:
        """Set the trial count."""
        with self._lock:
            self.trial_count = max(0, count)
    
    def get_completed_threads(self) -> int:
        """Get the number of completed threads."""
        with self._lock:
            return self.completed_threads
    
    def increment_completed_threads(self) -> int:
        """Increment and return the completed thread count."""
        with self._lock:
            self.completed_threads += 1
            return self.completed_threads
    
    def reset_completed_threads(self) -> None:
        """Reset the completed thread counter."""
        with self._lock:
            self.completed_threads = 0
    
    def add_error(self, category: str, message: str) -> None:
        """Add an error message to the specified category."""
        with self._lock:
            if category not in self.errors:
                self.errors[category] = []
            
            if message not in self.errors[category]:
                self.errors[category].append(message)
    
    def get_errors(self, category: str) -> List[str]:
        """Get errors for a specific category."""
        with self._lock:
            return self.errors.get(category, []).copy()
    
    def clear_errors(self, category: Optional[str] = None) -> None:
        """Clear errors for a category or all categories."""
        with self._lock:
            if category is None:
                self.errors = {"interrupt": []}
            else:
                self.errors[category] = []
    
    def has_errors(self, category: Optional[str] = None) -> bool:
        """Check if there are any errors."""
        with self._lock:
            if category is None:
                return any(errors for errors in self.errors.values())
            return bool(self.errors.get(category, []))
    
    def reset(self) -> None:
        """Reset all configuration to defaults."""
        with self._lock:
            self.threshold = None
            self.fish_id = None
            self.additional_text = None
            self.errors = {"interrupt": []}
            self.stop_requested = False
            self.processing_frame = None
            self.trial_count = 0
            self.completed_threads = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        with self._lock:
            return {
                "threshold": self.threshold,
                "fish_id": self.fish_id,
                "additional_text": self.additional_text,
                "trial_count": self.trial_count,
                "has_errors": self.has_errors(),
                "error_count": sum(len(errors) for errors in self.errors.values())
            }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"MeasurementConfig(threshold={self.threshold}, "
            f"fish_id={self.fish_id!r}, trial_count={self.trial_count})"
        )


# Singleton instance for global access (if needed for legacy compatibility)
_global_config: Optional[MeasurementConfig] = None
_config_lock = threading.Lock()


def get_global_config() -> MeasurementConfig:
    """Get or create the global configuration instance."""
    global _global_config
    
    with _config_lock:
        if _global_config is None:
            _global_config = MeasurementConfig()
        return _global_config


def reset_global_config() -> None:
    """Reset the global configuration instance."""
    global _global_config
    
    with _config_lock:
        if _global_config is not None:
            _global_config.reset()
        else:
            _global_config = MeasurementConfig()
