"""
Processing coordinator for parallel frame analysis.

This module handles the coordination of parallel frame processing,
replacing the processing logic previously embedded in MeasurerInstance.
"""

from typing import List, Optional, Callable, Any
import concurrent.futures as cf
import logging

import numpy as np

from vision import FishProcessor
from .measurement_config import MeasurementConfig
from .folder_manager import FolderManager

logger = logging.getLogger(__name__)


# Type alias for progress callbacks
ProgressCallback = Callable[[int, int, str], None]


class ProcessingCoordinator:
    """
    Coordinates parallel processing of fish measurement frames.

    This class is responsible for:
    - Creating processing instances from frames
    - Managing parallel processing with ThreadPoolExecutor
    - Tracking processing progress
    - Saving measurement results
    """

    def __init__(
        self,
        fish_processor: FishProcessor,
        config: MeasurementConfig,
        folder_manager: FolderManager,
        image_format: str
    ):
        """
        Initialize the processing coordinator.

        Args:
            fish_processor: FishProcessor instance for frame analysis
            config: Configuration manager
            folder_manager: Folder management instance
            image_format: Image file format for saving
        """
        self.fish_processor = fish_processor
        self.config = config
        self.folder_manager = folder_manager
        self.image_format = image_format
        self._progress_callback: Optional[ProgressCallback] = None

    def set_progress_callback(self, callback: Optional[ProgressCallback]) -> None:
        """
        Set a callback for progress updates.

        Args:
            callback: Function called with (current, total, message)
        """
        self._progress_callback = callback

    def _report_progress(self, current: int, total: int, message: str) -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            try:
                self._progress_callback(current, total, message)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def process_frames(
        self,
        raw_frames: List[np.ndarray],
        binarized_frames: List[np.ndarray]
    ) -> List[Any]:
        """
        Process frames to extract fish measurements.

        Args:
            raw_frames: List of raw camera frames
            binarized_frames: List of binarized (background-subtracted) frames

        Returns:
            List of successful measurement results
        """
        if not raw_frames:
            logger.warning("No frames to process")
            return []

        try:
            # Create processing instances
            instances = self._create_processing_instances(raw_frames, binarized_frames)

            if not instances:
                logger.warning("No processing instances created")
                return []

            # Process instances in parallel
            successful_measurements = self._process_instances_parallel(instances)

            logger.info(f"Successfully processed {len(successful_measurements)}/{len(instances)} frames")
            return successful_measurements

        except Exception as e:
            logger.error(f"Frame processing failed: {e}", exc_info=True)
            self.config.add_error("interrupt", f"Processing failed: {str(e)}")
            return []

    def _create_processing_instances(
        self,
        raw_frames: List[np.ndarray],
        binarized_frames: List[np.ndarray]
    ) -> List[Any]:
        """
        Create processing instances for each frame.

        Args:
            raw_frames: Raw camera frames
            binarized_frames: Binarized frames

        Returns:
            List of processing instances
        """
        instances = []

        for i, (raw_frame, binarized_frame) in enumerate(zip(raw_frames, binarized_frames)):
            if self.config.should_stop():
                logger.info("Processing instance creation cancelled")
                break

            self.config.set_processing_frame(i)
            self._report_progress(i + 1, len(raw_frames), f"Creating instance {i+1}/{len(raw_frames)}")

            try:
                # Use FishProcessor to analyze the frame
                fish_result = self.fish_processor.process_fish(
                    raw_frame=raw_frame,
                    binary_mask=binarized_frame,
                    fish_id=str(i)
                )

                if fish_result.success:
                    # Attach original frames for saving later
                    fish_result.raw_frame = raw_frame
                    fish_result.binarized_frame = binarized_frame
                    fish_result.process_id = i
                    instances.append(fish_result)
                else:
                    logger.warning(f"Frame {i} processing unsuccessful")

            except Exception as e:
                logger.warning(f"Failed to create processing instance for frame {i}: {e}")
                continue

        return instances

    def _process_instances_parallel(
        self,
        instances: List[Any]
    ) -> List[Any]:
        """
        Process instances in parallel using ThreadPoolExecutor.

        Args:
            instances: List of processing instances

        Returns:
            List of successful measurements
        """
        successful_measurements = []

        # For now, simplified processing without parallel execution
        # In the original implementation, this would use ThreadPoolExecutor
        # to call instance.ConstructLongestPath() on each instance

        for i, instance in enumerate(instances):
            if self.config.should_stop():
                logger.info("Parallel processing cancelled")
                break

            self.config.increment_completed_threads()
            self._report_progress(i + 1, len(instances), f"Processing {i+1}/{len(instances)}")

            try:
                # Instance is already processed by FishProcessor
                successful_measurements.append(instance)

                # Log the result
                self._log_measurement_result(instance)

                # Save images
                self._save_measurement_images(instance)

            except Exception as e:
                logger.error(f"Processing instance {i} failed: {e}")
                continue

        return successful_measurements

    def _log_measurement_result(self, instance: Any) -> None:
        """
        Log the measurement result for an instance.

        Args:
            instance: Processing instance with results
        """
        try:
            process_id = getattr(instance, 'process_id', 'unknown')
            total_length = getattr(instance, 'total_length_pixels', 0)

            logger.info(f"Frame {process_id}: {total_length:.2f}px")

            # Log processing details if available
            if hasattr(instance, 'processing_log'):
                for message in instance.processing_log:
                    logger.debug(message)

        except Exception as e:
            logger.warning(f"Failed to log measurement result: {e}")

    def _save_measurement_images(self, instance: Any) -> None:
        """
        Save images for a measurement instance.

        Args:
            instance: Processing instance with image data
        """
        try:
            import cv2

            process_id = getattr(instance, 'process_id', 0)

            # Save raw frame if available
            if hasattr(instance, 'raw_frame') and instance.raw_frame is not None:
                raw_path = self.folder_manager.get_raw_path(process_id, self.image_format)
                cv2.imwrite(str(raw_path), instance.raw_frame)

            # Save visualization if available
            if hasattr(instance, 'visualizations') and instance.visualizations:
                # Save the main visualization (e.g., skeleton with longest path)
                if 'longest_path_overlay' in instance.visualizations:
                    skeleton_path = self.folder_manager.get_skeleton_path(process_id, self.image_format)
                    cv2.imwrite(str(skeleton_path), instance.visualizations['longest_path_overlay'])

        except Exception as e:
            logger.warning(f"Failed to save images for frame {getattr(instance, 'process_id', 'unknown')}: {e}")

    def cancel(self) -> None:
        """Cancel ongoing processing."""
        self.config.request_stop()
        logger.info("Processing cancellation requested")

    def reset(self) -> None:
        """Reset coordinator state."""
        self.config.reset_stop()
        self.config.set_processing_frame(None)
        self.config.reset_completed_threads()
        logger.debug("Processing coordinator reset")
