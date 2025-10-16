"""
Unit tests for MeasurementStateMachine.

Tests the state machine behavior, transitions, and observer pattern.
"""

import pytest
from logic import MeasurementStateMachine, MeasurementState, StateTransition


class TestMeasurementStateMachine:
    """Test suite for MeasurementStateMachine."""

    @pytest.fixture
    def state_machine(self):
        """Create a fresh state machine for each test."""
        return MeasurementStateMachine()

    def test_initial_state(self, state_machine):
        """Test that state machine starts in UNCALIBRATED state."""
        assert state_machine.current_state == MeasurementState.UNCALIBRATED
        assert state_machine.previous_state is None

    def test_valid_transition_uncalibrated_to_calibrating(self, state_machine):
        """Test valid transition from UNCALIBRATED to CALIBRATING."""
        result = state_machine.transition_to(MeasurementState.CALIBRATING)
        assert result is True
        assert state_machine.current_state == MeasurementState.CALIBRATING
        assert state_machine.previous_state == MeasurementState.UNCALIBRATED

    def test_invalid_transition_raises_error(self, state_machine):
        """Test that invalid transitions raise ValueError."""
        with pytest.raises(ValueError, match="Invalid state transition"):
            state_machine.transition_to(MeasurementState.TRAINED)

    def test_transition_to_error_from_any_state(self, state_machine):
        """Test that ERROR state can be reached from any state."""
        # From UNCALIBRATED
        result = state_machine.transition_to(MeasurementState.ERROR)
        assert result is True
        assert state_machine.current_state == MeasurementState.ERROR

        # Reset and try from CALIBRATED
        state_machine.reset()
        state_machine.transition_to(MeasurementState.CALIBRATING)
        state_machine.transition_to(MeasurementState.CALIBRATED)
        result = state_machine.transition_to(MeasurementState.ERROR)
        assert result is True

    def test_complete_workflow_success(self, state_machine):
        """Test a complete successful workflow through all states."""
        # UNCALIBRATED -> CALIBRATING
        assert state_machine.transition_to(MeasurementState.CALIBRATING)
        assert state_machine.is_in_state(MeasurementState.CALIBRATING)

        # CALIBRATING -> CALIBRATED
        assert state_machine.transition_to(MeasurementState.CALIBRATED)
        assert state_machine.is_in_state(MeasurementState.CALIBRATED)

        # CALIBRATED -> TRAINING
        assert state_machine.transition_to(MeasurementState.TRAINING)
        assert state_machine.is_in_state(MeasurementState.TRAINING)

        # TRAINING -> TRAINED
        assert state_machine.transition_to(MeasurementState.TRAINED)
        assert state_machine.is_in_state(MeasurementState.TRAINED)

        # TRAINED -> PROCESSING
        assert state_machine.transition_to(MeasurementState.PROCESSING)
        assert state_machine.is_in_state(MeasurementState.PROCESSING)

        # PROCESSING -> TRAINED (cycle back)
        assert state_machine.transition_to(MeasurementState.TRAINED)
        assert state_machine.is_in_state(MeasurementState.TRAINED)

    def test_observer_notified_on_transition(self, state_machine):
        """Test that observers are notified of state transitions."""
        transitions = []

        def observer(transition: StateTransition):
            transitions.append(transition)

        state_machine.add_observer(observer)
        state_machine.transition_to(MeasurementState.CALIBRATING)

        assert len(transitions) == 1
        assert transitions[0].from_state == MeasurementState.UNCALIBRATED
        assert transitions[0].to_state == MeasurementState.CALIBRATING

    def test_multiple_observers(self, state_machine):
        """Test that multiple observers all get notified."""
        call_count = [0, 0]

        def observer1(transition):
            call_count[0] += 1

        def observer2(transition):
            call_count[1] += 1

        state_machine.add_observer(observer1)
        state_machine.add_observer(observer2)

        state_machine.transition_to(MeasurementState.CALIBRATING)

        assert call_count[0] == 1
        assert call_count[1] == 1

    def test_observer_removal(self, state_machine):
        """Test that removed observers are not notified."""
        call_count = [0]

        def observer(transition):
            call_count[0] += 1

        state_machine.add_observer(observer)
        state_machine.transition_to(MeasurementState.CALIBRATING)
        assert call_count[0] == 1

        state_machine.remove_observer(observer)
        state_machine.transition_to(MeasurementState.CALIBRATED)
        assert call_count[0] == 1  # Should not increase

    def test_metadata_storage(self, state_machine):
        """Test metadata storage and retrieval."""
        state_machine.set_metadata("test_key", "test_value")
        assert state_machine.get_metadata("test_key") == "test_value"
        assert state_machine.get_metadata("missing_key") is None
        assert state_machine.get_metadata("missing_key", "default") == "default"

    def test_metadata_in_transition(self, state_machine):
        """Test that metadata is passed with transitions."""
        transitions = []

        def observer(transition: StateTransition):
            transitions.append(transition)

        state_machine.add_observer(observer)

        metadata = {"frame_count": 10, "duration": 5}
        state_machine.transition_to(
            MeasurementState.CALIBRATING,
            reason="test_reason",
            metadata=metadata
        )

        assert len(transitions) == 1
        assert transitions[0].reason == "test_reason"
        assert transitions[0].metadata == metadata

    def test_reset(self, state_machine):
        """Test that reset returns to UNCALIBRATED state."""
        state_machine.transition_to(MeasurementState.CALIBRATING)
        state_machine.transition_to(MeasurementState.CALIBRATED)
        state_machine.set_metadata("key", "value")

        state_machine.reset()

        assert state_machine.current_state == MeasurementState.UNCALIBRATED
        assert state_machine.get_metadata("key") is None

    def test_is_calibrated(self, state_machine):
        """Test is_calibrated convenience method."""
        assert not state_machine.is_calibrated()

        state_machine.transition_to(MeasurementState.CALIBRATING)
        assert not state_machine.is_calibrated()

        state_machine.transition_to(MeasurementState.CALIBRATED)
        assert state_machine.is_calibrated()

        state_machine.transition_to(MeasurementState.TRAINING)
        assert state_machine.is_calibrated()

    def test_is_trained(self, state_machine):
        """Test is_trained convenience method."""
        # Navigate to TRAINED state
        state_machine.transition_to(MeasurementState.CALIBRATING)
        state_machine.transition_to(MeasurementState.CALIBRATED)
        state_machine.transition_to(MeasurementState.TRAINING)

        assert not state_machine.is_trained()

        state_machine.transition_to(MeasurementState.TRAINED)
        assert state_machine.is_trained()

    def test_is_busy(self, state_machine):
        """Test is_busy convenience method."""
        assert not state_machine.is_busy()

        # Navigate to TRAINING
        state_machine.transition_to(MeasurementState.CALIBRATING)
        state_machine.transition_to(MeasurementState.CALIBRATED)
        state_machine.transition_to(MeasurementState.TRAINING)

        assert state_machine.is_busy()

        state_machine.transition_to(MeasurementState.TRAINED)
        assert not state_machine.is_busy()

        state_machine.transition_to(MeasurementState.PROCESSING)
        assert state_machine.is_busy()

    def test_can_transition_to(self, state_machine):
        """Test can_transition_to validation."""
        assert state_machine.can_transition_to(MeasurementState.CALIBRATING)
        assert not state_machine.can_transition_to(MeasurementState.TRAINED)

        state_machine.transition_to(MeasurementState.CALIBRATING)
        assert state_machine.can_transition_to(MeasurementState.CALIBRATED)
        assert not state_machine.can_transition_to(MeasurementState.PROCESSING)

    def test_is_in_any_state(self, state_machine):
        """Test is_in_any_state method."""
        assert state_machine.is_in_any_state(
            MeasurementState.UNCALIBRATED,
            MeasurementState.CALIBRATING
        )

        assert not state_machine.is_in_any_state(
            MeasurementState.CALIBRATED,
            MeasurementState.TRAINED
        )

    def test_to_dict(self, state_machine):
        """Test state machine serialization."""
        state_machine.transition_to(MeasurementState.CALIBRATING)
        state_machine.set_metadata("key", "value")

        state_dict = state_machine.to_dict()

        assert state_dict["current_state"] == "CALIBRATING"
        assert state_dict["previous_state"] == "UNCALIBRATED"
        assert "key" in state_dict["metadata"]
        assert state_dict["metadata"]["key"] == "value"

    def test_thread_safety(self, state_machine):
        """Test that state machine handles concurrent access safely."""
        import threading
        import time

        transitions_completed = []
        errors = []

        def transition_worker(target_states):
            try:
                for state in target_states:
                    if state_machine.can_transition_to(state):
                        result = state_machine.transition_to(state)
                        if result:
                            transitions_completed.append(state)
                        time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(str(e))

        # Create threads that will try to transition through states
        thread1 = threading.Thread(
            target=transition_worker,
            args=([MeasurementState.CALIBRATING, MeasurementState.CALIBRATED],)
        )
        thread2 = threading.Thread(
            target=transition_worker,
            args=([MeasurementState.ERROR],)
        )

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Should not have any errors from concurrent access
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
