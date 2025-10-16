"""
Thread-safe state machine for the fish measurement application.

This module provides a unified state machine that replaces the scattered state
management across multiple modules. It implements the State pattern with
Observer pattern for state change notifications.
"""

from enum import Enum, auto
from typing import Set, Callable, Optional, Dict, Any
from dataclasses import dataclass
import threading
import logging

logger = logging.getLogger(__name__)


class MeasurementState(Enum):
    """States in the measurement workflow."""
    NOCAM = auto()         # No camera connected
    UNCALIBRATED = auto()  # Camera connected, no calibration data
    CALIBRATING = auto()   # Collecting calibration samples
    CALIBRATED = auto()    # Calibration complete, ready for training
    TRAINING = auto()      # Training background subtractor
    TRAINED = auto()       # Background trained, ready for measurement
    PROCESSING = auto()    # Capturing and analyzing frames
    ERROR = auto()         # Error state

    def __str__(self) -> str:
        """String representation for logging."""
        return self.name


@dataclass
class StateTransition:
    """Information about a state transition."""
    from_state: MeasurementState
    to_state: MeasurementState
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# Type alias for state change observers
StateObserver = Callable[[StateTransition], None]


class MeasurementStateMachine:
    """
    Thread-safe state machine for measurement workflow.

    This class manages the application state transitions and notifies
    observers when states change. It ensures valid state transitions
    and provides thread-safe access to the current state.

    Valid state transitions:
        UNCALIBRATED -> CALIBRATING -> CALIBRATED
        CALIBRATED -> TRAINING -> TRAINED
        TRAINED -> PROCESSING -> TRAINED (cycle for multiple measurements)
        Any state -> ERROR
        ERROR -> UNCALIBRATED (reset)
    """

    # Define valid state transitions
    VALID_TRANSITIONS: Dict[MeasurementState, Set[MeasurementState]] = {
        MeasurementState.NOCAM: {
            MeasurementState.UNCALIBRATED,  # Camera detected
            MeasurementState.ERROR
        },
        MeasurementState.UNCALIBRATED: {
            MeasurementState.CALIBRATING,
            MeasurementState.NOCAM,  # Camera disconnected
            MeasurementState.ERROR
        },
        MeasurementState.CALIBRATING: {
            MeasurementState.CALIBRATED,
            MeasurementState.UNCALIBRATED,  # Cancel calibration
            MeasurementState.NOCAM,  # Camera disconnected
            MeasurementState.ERROR
        },
        MeasurementState.CALIBRATED: {
            MeasurementState.TRAINING,
            MeasurementState.CALIBRATING,  # Recalibrate
            MeasurementState.UNCALIBRATED,  # Reset
            MeasurementState.NOCAM,  # Camera disconnected
            MeasurementState.ERROR
        },
        MeasurementState.TRAINING: {
            MeasurementState.TRAINED,
            MeasurementState.CALIBRATED,  # Cancel training
            MeasurementState.NOCAM,  # Camera disconnected
            MeasurementState.ERROR
        },
        MeasurementState.TRAINED: {
            MeasurementState.PROCESSING,
            MeasurementState.TRAINING,  # Retrain background
            MeasurementState.CALIBRATED,  # Reset to calibrated
            MeasurementState.UNCALIBRATED,  # Full reset
            MeasurementState.NOCAM,  # Camera disconnected
            MeasurementState.ERROR
        },
        MeasurementState.PROCESSING: {
            MeasurementState.TRAINED,  # Normal completion or cancel
            MeasurementState.NOCAM,  # Camera disconnected
            MeasurementState.ERROR
        },
        MeasurementState.ERROR: {
            MeasurementState.NOCAM,  # No camera after error
            MeasurementState.UNCALIBRATED,  # Reset from error
            MeasurementState.CALIBRATED,    # Resume if calibration is still valid
            MeasurementState.TRAINED        # Resume if training is still valid
        }
    }

    def __init__(self, initial_state: MeasurementState = MeasurementState.NOCAM):
        """
        Initialize the state machine.

        Args:
            initial_state: Starting state (default: NOCAM)
        """
        self._current_state = initial_state
        self._previous_state: Optional[MeasurementState] = None
        self._lock = threading.RLock()
        self._observers: Set[StateObserver] = set()
        self._state_metadata: Dict[str, Any] = {}

        logger.info(f"State machine initialized in state: {initial_state}")

    @property
    def current_state(self) -> MeasurementState:
        """Get the current state (thread-safe)."""
        with self._lock:
            return self._current_state

    @property
    def previous_state(self) -> Optional[MeasurementState]:
        """Get the previous state (thread-safe)."""
        with self._lock:
            return self._previous_state

    def is_in_state(self, state: MeasurementState) -> bool:
        """
        Check if currently in a specific state.

        Args:
            state: State to check

        Returns:
            True if in the specified state
        """
        with self._lock:
            return self._current_state == state

    def is_in_any_state(self, *states: MeasurementState) -> bool:
        """
        Check if currently in any of the specified states.

        Args:
            *states: States to check

        Returns:
            True if in any of the specified states
        """
        with self._lock:
            return self._current_state in states

    def can_transition_to(self, new_state: MeasurementState) -> bool:
        """
        Check if transition to new state is valid.

        Args:
            new_state: Desired state

        Returns:
            True if transition is valid
        """
        with self._lock:
            valid_next_states = self.VALID_TRANSITIONS.get(self._current_state, set())
            return new_state in valid_next_states

    def transition_to(
        self,
        new_state: MeasurementState,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition to a new state.

        Args:
            new_state: State to transition to
            reason: Optional reason for transition
            metadata: Optional additional data about the transition

        Returns:
            True if transition was successful, False otherwise

        Raises:
            ValueError: If transition is not valid
        """
        with self._lock:
            if not self.can_transition_to(new_state):
                error_msg = (
                    f"Invalid state transition: {self._current_state} -> {new_state}. "
                    f"Valid transitions from {self._current_state}: "
                    f"{self.VALID_TRANSITIONS.get(self._current_state, set())}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Store old state
            old_state = self._current_state
            self._previous_state = old_state

            # Update state
            self._current_state = new_state

            # Update metadata if provided
            if metadata:
                self._state_metadata.update(metadata)

            # Create transition object
            transition = StateTransition(
                from_state=old_state,
                to_state=new_state,
                reason=reason,
                metadata=metadata
            )

            logger.info(
                f"State transition: {old_state} -> {new_state}"
                f"{f' (reason: {reason})' if reason else ''}"
            )

            # Notify observers (outside lock to avoid deadlock)
            observers_copy = self._observers.copy()

        # Call observers outside the lock
        self._notify_observers(transition, observers_copy)

        return True

    def _notify_observers(
        self,
        transition: StateTransition,
        observers: Set[StateObserver]
    ) -> None:
        """
        Notify all observers of state transition.

        Args:
            transition: The state transition that occurred
            observers: Set of observer callbacks to notify
        """
        for observer in observers:
            try:
                observer(transition)
            except Exception as e:
                logger.error(f"Error in state observer: {e}", exc_info=True)

    def add_observer(self, observer: StateObserver) -> None:
        """
        Add a state change observer.

        Args:
            observer: Callback function to be called on state transitions
        """
        with self._lock:
            self._observers.add(observer)
            logger.debug(f"Added state observer: {observer.__name__}")

    def remove_observer(self, observer: StateObserver) -> None:
        """
        Remove a state change observer.

        Args:
            observer: Callback function to remove
        """
        with self._lock:
            self._observers.discard(observer)
            logger.debug(f"Removed state observer: {observer.__name__}")

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Store metadata associated with current state.

        Args:
            key: Metadata key
            value: Metadata value
        """
        with self._lock:
            self._state_metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata associated with current state.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        with self._lock:
            return self._state_metadata.get(key, default)

    def clear_metadata(self) -> None:
        """Clear all state metadata."""
        with self._lock:
            self._state_metadata.clear()

    def reset(self, has_camera: bool = False) -> None:
        """
        Reset state machine to initial state.

        Args:
            has_camera: If True, reset to UNCALIBRATED; if False, reset to NOCAM
        """
        with self._lock:
            old_state = self._current_state
            new_state = MeasurementState.UNCALIBRATED if has_camera else MeasurementState.NOCAM
            self._current_state = new_state
            self._previous_state = old_state
            self._state_metadata.clear()

            logger.info(f"State machine reset: {old_state} -> {new_state}")

            # Create reset transition
            transition = StateTransition(
                from_state=old_state,
                to_state=new_state,
                reason="reset",
                metadata=None
            )

            observers_copy = self._observers.copy()

        # Notify observers outside lock
        self._notify_observers(transition, observers_copy)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state machine to dictionary for serialization.

        Returns:
            Dictionary representation of state machine
        """
        with self._lock:
            return {
                "current_state": self._current_state.name,
                "previous_state": self._previous_state.name if self._previous_state else None,
                "metadata": self._state_metadata.copy()
            }

    def __repr__(self) -> str:
        """String representation of state machine."""
        with self._lock:
            return (
                f"MeasurementStateMachine(current={self._current_state}, "
                f"previous={self._previous_state})"
            )

    # Convenience methods for common state checks

    def is_calibrated(self) -> bool:
        """Check if calibration is complete (CALIBRATED or later states)."""
        return self.is_in_any_state(
            MeasurementState.CALIBRATED,
            MeasurementState.TRAINING,
            MeasurementState.TRAINED,
            MeasurementState.PROCESSING
        )

    def is_trained(self) -> bool:
        """Check if background training is complete (TRAINED or PROCESSING)."""
        return self.is_in_any_state(
            MeasurementState.TRAINED,
            MeasurementState.PROCESSING
        )

    def is_ready_for_measurement(self) -> bool:
        """Check if ready to start measurement (TRAINED state)."""
        return self.is_in_state(MeasurementState.TRAINED)

    def is_busy(self) -> bool:
        """Check if system is busy (TRAINING or PROCESSING)."""
        return self.is_in_any_state(
            MeasurementState.TRAINING,
            MeasurementState.PROCESSING
        )
