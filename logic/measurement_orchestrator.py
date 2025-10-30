"""
GUI-agnostic measurement orchestrator for fish measurement system.

This module provides a centralized orchestrator that manages the entire
measurement workflow, replacing duplicated logic in both GUI implementations.
"""

from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
import threading
import time
import logging
import concurrent.futures as cf

import cv2
import numpy as np

from .measurement_state_machine import MeasurementStateMachine, MeasurementState, StateTransition
from .measurement_config import MeasurementConfig
from .measurement_statistics import MeasurementStatistics
from .image_watermarker import ImageWatermarker
from .folder_manager import FolderManager
from vision import ArUcoDetector, FishProcessor

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of calibration check."""
    is_calibrated: bool
    marker_count: int
    message: str


@dataclass
class TrainingResult:
    """Result of background training."""
    success: bool
    frames_captured: int
    message: str


@dataclass
class ProcessingResult:
    """Result of frame processing."""
    success: bool
    measurements_count: int
    trial_count: int
    message: str
    errors: List[str]


# Type alias for progress callbacks
ProgressCallback = Callable[[int, int, str], None]  # (current, total, message)


class MeasurementOrchestrator:
    """
    Central orchestrator for fish measurement workflow.

    This class coordinates all measurement operations in a GUI-agnostic way,
    providing a clean interface that both tkinter and PySide6 GUIs can use.

    The orchestrator manages:
    - State transitions via MeasurementStateMachine
    - Calibration workflow
    - Background training workflow
    - Frame capture and processing workflow
    - Statistical analysis and export
    """

    def __init__(
        self,
        camera_manager,
        output_folder: Optional[str] = None,
        image_format: Optional[str] = None
    ):
        """
        Initialize the measurement orchestrator.

        Args:
            camera_manager: Camera manager instance (CameraManager or compatible interface)
            output_folder: Output folder for results
            image_format: Image file format (.jpeg, .png, .tiff)
        """
        # Core components
        self.camera_manager = camera_manager
        self.state_machine = MeasurementStateMachine()
        self.config = MeasurementConfig()

        # Output settings
        self.output_folder = output_folder
        self.image_format = image_format

        # Folder management
        self.folder_manager = FolderManager()

        # Vision components
        self.aruco_detector = ArUcoDetector()
        self.fish_processor = FishProcessor(output_folder=output_folder)

        # Background subtraction
        self.background_subtractor: Optional[cv2.BackgroundSubtractor] = None

        # Processing state
        self.measurements: List = []
        self._processing_thread: Optional[threading.Thread] = None
        self._training_thread: Optional[threading.Thread] = None

        # Progress callbacks
        self._progress_callback: Optional[ProgressCallback] = None

        logger.info("MeasurementOrchestrator initialized")

    def set_progress_callback(self, callback: Optional[ProgressCallback]) -> None:
        """
        Set a callback for progress updates.

        Args:
            callback: Function called with (current, total, message) during processing
        """
        self._progress_callback = callback

    # ========================
    # Camera utilities
    # ========================

    def _get_camera_framerate(self) -> float:
        """Return camera framerate from HAL (.settings) or legacy manager (.config)."""
        try:
            cam = self.camera_manager
            if cam is None:
                return 30.0
            # Preferred: HAL settings
            if hasattr(cam, "settings") and hasattr(cam.settings, "framerate"):
                return float(cam.settings.framerate)
            # Legacy: CameraManager config
            if hasattr(cam, "config") and hasattr(cam.config, "framerate"):
                return float(cam.config.framerate)
        except Exception:
            pass
        return 30.0

    def _report_progress(self, current: int, total: int, message: str) -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            try:
                self._progress_callback(current, total, message)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    # ========================
    # Calibration Workflow
    # ========================

    def start_calibration(self) -> bool:
        """
        Start calibration mode.

        Returns:
            True if calibration mode started successfully
        """
        try:
            if not self.state_machine.can_transition_to(MeasurementState.CALIBRATING):
                logger.warning(f"Cannot start calibration from state: {self.state_machine.current_state}")
                return False

            self.state_machine.transition_to(
                MeasurementState.CALIBRATING,
                reason="user_initiated"
            )

            # Clear previous calibration data
            self.aruco_detector.clear_calibration()

            logger.info("Calibration mode started")
            return True

        except Exception as e:
            logger.error(f"Failed to start calibration: {e}")
            return False

    def update_calibration(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Update calibration with current frame.

        This should be called continuously while in CALIBRATING state to
        detect ArUco markers and build calibration data.

        Args:
            frame: Camera frame to process

        Returns:
            Tuple of (markers_detected, marker_count)
        """
        try:
            # Detect ArUco markers
            markers = self.aruco_detector.detect_markers(frame)

            if markers:
                # Calculate calibration from detected markers
                self.aruco_detector.calculate_calibration(markers)

                # Get current calibration status
                calibration = self.aruco_detector.get_average_calibration()
                marker_count = calibration.marker_count if calibration else 0

                # Auto-complete calibration when enough samples collected
                if marker_count >= 15:
                    self.complete_calibration()

                return True, marker_count

            return False, 0

        except Exception as e:
            logger.error(f"Error updating calibration: {e}")
            return False, 0

    def complete_calibration(self) -> CalibrationResult:
        """
        Complete calibration and transition to CALIBRATED state.

        Returns:
            CalibrationResult with status and details
        """
        try:
            calibration = self.aruco_detector.get_average_calibration()

            if calibration is None:
                return CalibrationResult(
                    is_calibrated=False,
                    marker_count=0,
                    message="No calibration data available"
                )

            if calibration.marker_count < 3:
                return CalibrationResult(
                    is_calibrated=False,
                    marker_count=calibration.marker_count,
                    message=f"Insufficient calibration samples: {calibration.marker_count}/3 minimum"
                )

            # Transition to calibrated state
            self.state_machine.transition_to(
                MeasurementState.CALIBRATED,
                reason="calibration_complete",
                metadata={"marker_count": calibration.marker_count}
            )

            logger.info(f"Calibration completed with {calibration.marker_count} samples")

            return CalibrationResult(
                is_calibrated=True,
                marker_count=calibration.marker_count,
                message=f"Calibration successful with {calibration.marker_count} samples"
            )

        except Exception as e:
            logger.error(f"Error completing calibration: {e}")
            return CalibrationResult(
                is_calibrated=False,
                marker_count=0,
                message=f"Calibration failed: {str(e)}"
            )

    def check_calibration_status(self) -> CalibrationResult:
        """
        Check current calibration status.

        Returns:
            CalibrationResult with current status
        """
        calibration = self.aruco_detector.get_average_calibration()

        if calibration is None:
            return CalibrationResult(
                is_calibrated=False,
                marker_count=0,
                message="No calibration data"
            )

        is_calibrated = calibration.marker_count >= 3

        return CalibrationResult(
            is_calibrated=is_calibrated,
            marker_count=calibration.marker_count,
            message=f"Calibration: {calibration.marker_count} samples"
        )

    # ========================
    # Background Training Workflow
    # ========================

    def start_background_training(
        self,
        duration_seconds: int = 1,
        async_mode: bool = True
    ) -> bool:
        """
        Start background training process.

        Args:
            duration_seconds: Duration to capture background frames
            async_mode: If True, runs in background thread; if False, blocks

        Returns:
            True if training started successfully
        """
        try:
            # Validate state transition
            if not self.state_machine.can_transition_to(MeasurementState.TRAINING):
                logger.warning(f"Cannot start training from state: {self.state_machine.current_state}")
                return False

            # Ensure previous cancellations don't carry over
            self.config.reset_stop()
            self.config.clear_errors("interrupt")

            # Transition to training state
            self.state_machine.transition_to(
                MeasurementState.TRAINING,
                reason="user_initiated"
            )

            if async_mode:
                # Run in background thread
                self._training_thread = threading.Thread(
                    target=self._train_background_impl,
                    args=(duration_seconds,),
                    daemon=True
                )
                self._training_thread.start()
            else:
                # Run synchronously
                self._train_background_impl(duration_seconds)

            return True

        except Exception as e:
            logger.error(f"Failed to start background training: {e}")
            self.state_machine.transition_to(
                MeasurementState.ERROR,
                reason=f"training_start_failed: {str(e)}"
            )
            return False

    def _train_background_impl(self, duration_seconds: int) -> None:
        """
        Internal implementation of background training.

        Args:
            duration_seconds: Duration to capture background frames
        """
        try:
            logger.info("Capturing background frames for training")
            self._report_progress(0, 100, "Capturing background...")

            # Calculate number of frames to capture
            num_frames = int(self._get_camera_framerate() * duration_seconds)
            background_images = []

            # Capture frames
            for i in range(num_frames):
                if self.config.should_stop():
                    logger.info("Background training cancelled")
                    self.state_machine.transition_to(
                        MeasurementState.CALIBRATED,
                        reason="training_cancelled"
                    )
                    return

                frame = self.camera_manager.current_frame
                if frame is not None:
                    background_images.append(frame)

                # Report progress
                progress = int((i + 1) / num_frames * 100)
                self._report_progress(i + 1, num_frames, f"Capturing frame {i+1}/{num_frames}")

                time.sleep(1.0 / self._get_camera_framerate())

            if not background_images:
                logger.error("No background frames captured")
                self.config.add_error("interrupt", "Failed to capture background frames")
                self.state_machine.transition_to(
                    MeasurementState.ERROR,
                    reason="no_background_frames"
                )
                return

            # If cancellation requested after capture, honor it
            if self.config.should_stop():
                logger.info("Background training cancelled after capture phase")
                self.state_machine.transition_to(
                    MeasurementState.CALIBRATED,
                    reason="training_cancelled"
                )
                return

            # Initialize background subtractor
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2()

            # Train with captured frames
            for i, image in enumerate(background_images):
                if self.config.should_stop():
                    logger.info("Background training cancelled during training phase")
                    self.state_machine.transition_to(
                        MeasurementState.CALIBRATED,
                        reason="training_cancelled"
                    )
                    return
                self.background_subtractor.apply(image)
                progress = int((i + 1) / len(background_images) * 100)
                self._report_progress(i + 1, len(background_images), f"Training {i+1}/{len(background_images)}")

            # Transition to trained state
            self.state_machine.transition_to(
                MeasurementState.TRAINED,
                reason="training_complete",
                metadata={"frames_trained": len(background_images)}
            )

            logger.info(f"Background training completed with {len(background_images)} frames")

        except Exception as e:
            logger.error(f"Background training failed: {e}", exc_info=True)
            self.config.add_error("interrupt", f"Background training failed: {str(e)}")
            self.state_machine.transition_to(
                MeasurementState.ERROR,
                reason=f"training_failed: {str(e)}"
            )

    def is_training_complete(self) -> bool:
        """
        Check if background training is complete.

        Returns:
            True if training thread has finished
        """
        if self._training_thread is None:
            return True
        return not self._training_thread.is_alive()

    def cancel_training(self) -> None:
        """Cancel ongoing background training."""
        self.config.request_stop()
        logger.info("Training cancellation requested")

    def reset_background(self) -> None:
        """Reset background training."""
        try:
            self.background_subtractor = None
            self.config.reset_stop()

            # Transition back to calibrated state
            if self.state_machine.is_calibrated():
                self.state_machine.transition_to(
                    MeasurementState.CALIBRATED,
                    reason="background_reset"
                )

            logger.info("Background reset")

        except Exception as e:
            logger.error(f"Failed to reset background: {e}")

    # ========================
    # Frame Processing Workflow
    # ========================

    def start_processing(
        self,
        num_frames: int,
        async_mode: bool = True
    ) -> bool:
        """
        Start frame capture and processing.

        Args:
            num_frames: Number of frames to capture and process
            async_mode: If True, runs in background thread

        Returns:
            True if processing started successfully
        """
        try:
            # Validate state
            if not self.state_machine.can_transition_to(MeasurementState.PROCESSING):
                logger.warning(f"Cannot start processing from state: {self.state_machine.current_state}")
                return False

            # Validate settings
            if not self.output_folder or not self.image_format:
                logger.error("Output folder and image format must be set")
                return False

            if num_frames < 3:
                logger.warning(f"Frame count too low: {num_frames}, using minimum of 3")
                num_frames = 3

            # Reset processing state
            self.measurements = []
            self.config.reset_stop()
            self.config.reset_completed_threads()

            # Transition to processing state
            self.state_machine.transition_to(
                MeasurementState.PROCESSING,
                reason="user_initiated",
                metadata={"num_frames": num_frames}
            )

            if async_mode:
                # Run in background thread
                self._processing_thread = threading.Thread(
                    target=self._process_frames_impl,
                    args=(num_frames,),
                    daemon=True
                )
                self._processing_thread.start()
            else:
                # Run synchronously
                self._process_frames_impl(num_frames)

            return True

        except Exception as e:
            logger.error(f"Failed to start processing: {e}")
            self.state_machine.transition_to(
                MeasurementState.ERROR,
                reason=f"processing_start_failed: {str(e)}"
            )
            return False

    def _process_frames_impl(self, num_frames: int) -> None:
        """
        Internal implementation of frame processing.

        Args:
            num_frames: Number of frames to capture and process
        """
        try:
            # Step 1: Capture frames
            logger.info(f"Capturing {num_frames} frames...")
            self._report_progress(0, num_frames, "Capturing frames...")

            raw_frames = []
            binarized_frames = []

            for i in range(num_frames):
                if self.config.should_stop():
                    logger.info("Processing cancelled during capture")
                    self._handle_processing_cancelled()
                    return

                frame = self.camera_manager.current_frame
                if frame is None:
                    logger.warning(f"Frame {i}: camera frame is None")
                    continue

                raw_frames.append(frame.copy())

                # Apply background subtraction
                binary_mask = self._subtract_background(frame)
                if binary_mask is not None:
                    binarized_frames.append(binary_mask)
                else:
                    logger.warning(f"Frame {i}: background subtraction failed")
                    # Still add a placeholder to keep lists aligned
                    binarized_frames.append(None)

                self._report_progress(i + 1, num_frames, f"Capturing frame {i+1}/{num_frames}")
                time.sleep(1.0 / self._get_camera_framerate())

            logger.info(f"Captured {len(raw_frames)} raw frames and {len(binarized_frames)} binarized frames")

            if not raw_frames:
                self.config.add_error("interrupt", "No frames captured")
                self._handle_processing_error()
                return

            # Step 2: Analyze frames
            logger.info(f"Analyzing {len(raw_frames)} frames...")
            self._analyze_frames(raw_frames, binarized_frames)

        except Exception as e:
            logger.error(f"Frame processing failed: {e}", exc_info=True)
            self.config.add_error("interrupt", f"Processing failed: {str(e)}")
            self._handle_processing_error()

    def _subtract_background(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Subtract background and create binary mask.

        Args:
            frame: Input frame

        Returns:
            Binary mask or None if processing fails
        """
        if self.background_subtractor is None:
            return None

        try:
            # Apply background subtraction (learningRate=0 means no adaptation)
            foreground_mask = self.background_subtractor.apply(frame, learningRate=0)

            # Threshold the mask
            threshold = self.config.get_threshold()
            _, binary_image = cv2.threshold(
                foreground_mask, threshold, 255, cv2.THRESH_BINARY
            )

            # Find and fill the largest contour
            contours, _ = cv2.findContours(
                binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            if not contours:
                return None

            largest_contour = max(contours, key=cv2.contourArea)

            # Create binary mask with largest contour
            binary_mask = np.zeros_like(binary_image, dtype=np.uint8)
            cv2.drawContours(
                binary_mask, [largest_contour], -1, 255, thickness=cv2.FILLED
            )

            return binary_mask

        except Exception as e:
            logger.error(f"Background subtraction failed: {e}")
            return None

    def _analyze_frames(
        self,
        raw_frames: List[np.ndarray],
        binarized_frames: List[np.ndarray]
    ) -> None:
        """
        Analyze captured frames to measure fish length.

        Args:
            raw_frames: List of raw frames
            binarized_frames: List of binarized frames
        """
        self.measurements = []

        if not raw_frames:
            logger.warning("No frames to analyze")
            return

        try:
            # Setup folder structure
            self.folder_manager.setup_folders(
                self.output_folder,
                self.config.get_fish_id()
            )

            # Create processing instances
            instances = self._create_processing_instances(raw_frames, binarized_frames)

            if not instances:
                self.config.add_error(
                    "interrupt",
                    "No length values could be obtained. Please check image quality and try again."
                )
                self._handle_processing_error()
                return

            # Process instances (simplified - in real implementation would use ThreadPoolExecutor)
            successful_measurements = instances  # Simplified for now

            if not successful_measurements:
                self.config.add_error(
                    "interrupt",
                    "No successful measurements. Please try again."
                )
                self._handle_processing_error()
                return

            # Perform statistical analysis and export
            self._finalize_analysis(successful_measurements)

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self.config.add_error("interrupt", f"Analysis failed: {str(e)}")
            self._handle_processing_error()

    def _create_processing_instances(
        self,
        raw_frames: List[np.ndarray],
        binarized_frames: List[np.ndarray]
    ) -> List:
        """Create processing instances for each frame."""
        instances = []

        for i, (raw_frame, binarized_frame) in enumerate(zip(raw_frames, binarized_frames)):
            if self.config.should_stop():
                break

            # Skip frames with no binarized data
            if binarized_frame is None:
                logger.warning(f"Skipping frame {i}: no binarized data")
                continue

            self.config.set_processing_frame(i)
            self._report_progress(i + 1, len(raw_frames), f"Analyzing frame {i+1}/{len(raw_frames)}")

            try:
                # Use FishProcessor to process the frame
                fish_result = self.fish_processor.process_fish(
                    raw_frame=raw_frame,
                    binary_mask=binarized_frame,
                    fish_id=str(i)
                )

                if fish_result.success:
                    # Attach raw frames for later saving
                    fish_result.raw_frame = raw_frame
                    fish_result.binarized_frame = binarized_frame
                    fish_result.frame_index = i

                    instances.append(fish_result)

                    # Save images immediately
                    self._save_frame_images(fish_result)
                    logger.info(f"Frame {i} processed successfully: {fish_result.total_length_pixels:.2f} pixels")
                else:
                    logger.warning(f"Frame {i} processing failed: {fish_result.error_message}")

            except Exception as e:
                logger.error(f"Failed to process frame {i}: {e}", exc_info=True)
                continue

        return instances

    def _save_frame_images(self, result) -> None:
        """
        Save images for a measurement result.

        Args:
            result: FishMeasurementResult with image data
        """
        try:
            frame_idx = getattr(result, 'frame_index', 0)

            # Save raw frame
            if hasattr(result, 'raw_frame') and result.raw_frame is not None:
                raw_path = self.folder_manager.get_raw_path(frame_idx, self.image_format)
                cv2.imwrite(str(raw_path), result.raw_frame)
                logger.debug(f"Saved raw frame {frame_idx} to {raw_path}")

            # Save visualization (skeleton + longest path overlay)
            if result.visualization_data and 'processed_frame' in result.visualization_data:
                skeleton_path = self.folder_manager.get_skeleton_path(frame_idx, self.image_format)
                cv2.imwrite(str(skeleton_path), result.visualization_data['processed_frame'])
                logger.debug(f"Saved skeleton frame {frame_idx} to {skeleton_path}")

        except Exception as e:
            logger.warning(f"Failed to save images for frame {frame_idx}: {e}")

    def _finalize_analysis(self, measurements: List) -> None:
        """Finalize analysis with statistics and export."""
        self.measurements = measurements

        if not self.config.should_stop():
            # Filter outliers and calculate statistics (simplified)
            self.config.set_trial_count(len(measurements))

            if len(measurements) < 2:
                self.config.add_error(
                    "interrupt",
                    "Too few measurements to calculate statistics. Please try again."
                )
                self._handle_processing_error()
            else:
                # Complete successfully
                self.state_machine.transition_to(
                    MeasurementState.TRAINED,
                    reason="processing_complete",
                    metadata={"measurements": len(measurements)}
                )
                logger.info(f"Processing completed with {len(measurements)} measurements")

    def _handle_processing_cancelled(self) -> None:
        """Handle cancelled processing."""
        self.state_machine.transition_to(
            MeasurementState.TRAINED,
            reason="processing_cancelled"
        )
        self.config.set_processing_frame(None)
        self.config.reset_completed_threads()

    def _handle_processing_error(self) -> None:
        """Handle processing error."""
        self.state_machine.transition_to(
            MeasurementState.ERROR,
            reason="processing_failed"
        )
        self.config.set_processing_frame(None)
        self.config.reset_completed_threads()

    def cancel_processing(self) -> None:
        """Cancel ongoing processing."""
        self.config.request_stop()
        logger.info("Processing cancellation requested")

    def is_processing_complete(self) -> bool:
        """
        Check if processing is complete.

        Returns:
            True if processing thread has finished
        """
        if self._processing_thread is None:
            return True
        return not self._processing_thread.is_alive()

    # ========================
    # Live Preview (TRAINED/PROCESSING)
    # ========================

    def create_binarized_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return binary mask of largest foreground object on black background.

        Requires a trained background subtractor (TRAINED/PROCESSING states).
        Returns None if not available.
        """
        if self.background_subtractor is None or frame is None:
            return None
        return self._subtract_background(frame)

    # ========================
    # State Queries
    # ========================

    def get_current_state(self) -> MeasurementState:
        """Get current state."""
        return self.state_machine.current_state

    def is_ready_for_measurement(self) -> bool:
        """Check if ready to start measurement."""
        return self.state_machine.is_ready_for_measurement()

    def is_busy(self) -> bool:
        """Check if system is busy (training or processing)."""
        return self.state_machine.is_busy()

    # ========================
    # Configuration
    # ========================

    def set_output_folder(self, folder: str) -> None:
        """Set output folder for results."""
        self.output_folder = folder
        self.fish_processor.output_folder = folder

    def set_image_format(self, format: str) -> None:
        """Set image format (.jpeg, .png, .tiff)."""
        self.image_format = format

    def set_threshold(self, threshold: int) -> None:
        """Set background subtraction threshold."""
        self.config.set_threshold(threshold)

    def set_fish_id(self, fish_id: str) -> None:
        """Set fish ID for current measurement."""
        self.config.set_fish_id(fish_id)

    def set_additional_text(self, text: str) -> None:
        """Set additional text for watermarking."""
        self.config.set_additional_text(text)

    # ========================
    # Reset and Cleanup
    # ========================

    def reset(self) -> None:
        """Reset orchestrator to initial state."""
        self.measurements = []
        self.background_subtractor = None
        self.config.reset()
        self.folder_manager.reset()
        self.state_machine.reset()
        logger.info("Orchestrator reset")

    def cleanup(self) -> None:
        """Cleanup resources."""
        # Cancel any ongoing operations
        self.config.request_stop()

        # Wait for threads to complete
        if self._training_thread and self._training_thread.is_alive():
            self._training_thread.join(timeout=2.0)

        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        logger.info("Orchestrator cleanup complete")
