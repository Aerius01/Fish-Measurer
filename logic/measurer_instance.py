"""
Main measurement instance for fish length analysis.

This module contains the core measurement logic, orchestrating the entire
fish measurement process from image capture to data export.
"""

from typing import List, Tuple, Optional, Any
import os
import time
import logging
import concurrent.futures as cf

import cv2
import numpy as np

from vision import CameraManager, CameraConfig, FishProcessor, ArUcoDetector
from .measurement_config import MeasurementConfig
from .measurement_statistics import MeasurementStatistics
from .image_watermarker import ImageWatermarker
from .folder_manager import FolderManager


logger = logging.getLogger(__name__)

# Type alias for backwards compatibility
ProcessingInstance = Any


class MeasurerInstance:
    """
    Main measurement instance that orchestrates the fish measurement process.
    
    This class handles background subtraction, frame analysis, statistical processing,
    and data export for fish length measurements.
    """
    
    def __init__(self, camera_manager: Optional['CameraManager'] = None):
        """Initialize the MeasurerInstance."""
        # Instance-specific variables
        self.output_folder: Optional[str] = None
        self.image_format: Optional[str] = None
        
        # Folder structure
        self.folder_manager = FolderManager()
        
        # Processing state
        self.background_is_trained: bool = False
        self.block_tkinter_start_button: bool = False
        self.pulling_background: bool = False
        
        # Background subtractor
        self.background_subtractor: Optional[cv2.BackgroundSubtractor] = None
        self.binary_mask: Optional[np.ndarray] = None
        
        # Measurement results
        self.measurements: List[ProcessingInstance] = []
        
        # Configuration
        self.config = MeasurementConfig()
        
        # Initialize vision components
        try:
            # Use provided camera manager or create a new one
            if camera_manager is not None:
                self.camera_manager = camera_manager
            else:
                self.camera_manager = CameraManager()
            
            self.fish_processor = FishProcessor(output_folder=self.output_folder)
            self.aruco_detector = ArUcoDetector()
            
            # Connect ArUco detector to config for calibration access
            self.config._aruco_detector = self.aruco_detector
            
        except Exception as e:
            logger.error(f"Failed to initialize vision components: {e}")
            raise
    
    def subtract_background(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Subtract background and create binary mask of the largest contour.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Binary mask with largest contour or None if processing fails
        """
        if self.background_subtractor is None:
            logger.error("Background subtractor not initialized")
            self.block_tkinter_start_button = True
            return None
        
        try:
            # Apply background subtraction
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
                raise ValueError("No contours found")
            
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create binary mask with largest contour
            self.binary_mask = np.zeros_like(binary_image, dtype=np.uint8)
            cv2.drawContours(
                self.binary_mask, [largest_contour], -1, 255, thickness=cv2.FILLED
            )
            
            self.block_tkinter_start_button = False
            return self.binary_mask
            
        except (ValueError, cv2.error) as e:
            self.block_tkinter_start_button = True
            logger.error(f"Background subtraction failed: {e}")
            return None
    
    def train_background(self, duration_seconds: int = 1) -> bool:
        """
        Train the background subtractor using captured frames.
        
        Args:
            duration_seconds: Number of seconds to capture background frames
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Capturing background frames for training")
            self.pulling_background = True
            
            # Use camera manager to capture frames
            num_frames = self.camera_manager.config.framerate * duration_seconds
            background_images = []
            
            # Capture frames for background subtraction
            for _ in range(int(num_frames)):
                frame = self.camera_manager.current_frame
                if frame is not None:
                    background_images.append(frame)
                time.sleep(1.0 / self.camera_manager.config.framerate)
            
            if not background_images:
                logger.error("No background frames captured")
                return False
            
            # Initialize background subtractor
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
            
            # Train with captured frames
            for image in background_images:
                self.background_subtractor.apply(image)
            
            self.background_is_trained = True
            self.pulling_background = False
            
            logger.info(f"Background training completed with {len(background_images)} frames")
            return True
            
        except Exception as e:
            logger.error(f"Background training failed: {e}")
            self.pulling_background = False
            return False
    
    def analyze_frames(self, frames: Tuple[List[np.ndarray], List[np.ndarray]]) -> None:
        """
        Analyze captured frames to measure fish length.
        
        Args:
            frames: Tuple of (raw_frames, binarized_frames)
        """
        raw_frames, binarized_frames = frames
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
            
            # Process frames and create processing instances
            instances = self._create_processing_instances(raw_frames, binarized_frames)
            
            if not instances:
                self._handle_no_instances_error()
                return
            
            # Process instances in parallel
            successful_measurements = self._process_instances_parallel(instances)
            
            if not successful_measurements:
                self._handle_no_measurements_error()
                return
            
            # Perform statistical analysis and export
            self._finalize_analysis(successful_measurements)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self.config.add_error("interrupt", f"Analysis failed: {str(e)}")
        finally:
            # Reset processing state
            self.config.set_processing_frame(None)
            self.config.reset_completed_threads()
    
    def _create_processing_instances(
        self, 
        raw_frames: List[np.ndarray], 
        binarized_frames: List[np.ndarray]
    ) -> List[ProcessingInstance]:
        """Create processing instances for each frame."""
        instances = []
        
        for i, (raw_frame, binarized_frame) in enumerate(zip(raw_frames, binarized_frames)):
            if self.config.should_stop():
                break
            
            self.config.set_processing_frame(i)
            
            try:
                # Use FishProcessor to process the frame
                fish_result = self.fish_processor.process_fish(
                    raw_frame=raw_frame,
                    binary_mask=binarized_frame,
                    fish_id=str(i)
                )
                
                if fish_result.success:
                    # Create a compatible instance object for backwards compatibility
                    processing_instance = type('ProcessingInstance', (), {
                        'process_id': i,
                        'fil_length_pixels': fish_result.total_length_pixels,
                        'successful_init': True,
                        'output_log': fish_result.processing_log,
                        'raw_frame': raw_frame,
                        'binarized_frame': binarized_frame
                    })()
                    instances.append(processing_instance)
                    
            except Exception as e:
                logger.warning(f"Failed to create processing instance for frame {i}: {e}")
                continue
        
        return instances
    
    def _process_instances_parallel(
        self, 
        instances: List[ProcessingInstance]
    ) -> List[ProcessingInstance]:
        """Process instances in parallel using ThreadPoolExecutor."""
        successful_measurements = []
        
        with cf.ThreadPoolExecutor() as executor:
            if self.config.should_stop():
                return successful_measurements
            
            # Submit all tasks
            futures = [
                executor.submit(instance.ConstructLongestPath) 
                for instance in instances
            ]
            
            # Process results as they complete
            for future in cf.as_completed(futures):
                self.config.increment_completed_threads()
                
                try:
                    instance = future.result()
                    
                    if isinstance(instance, ProcessingInstance) and instance.successful_pathing:
                        successful_measurements.append(instance)
                        self._log_measurement_result(instance)
                        self._save_measurement_images(instance)
                        
                except Exception as e:
                    logger.error(f"Processing instance failed: {e}")
                    continue
        
        return successful_measurements
    
    def _log_measurement_result(self, instance: ProcessingInstance) -> None:
        """Log the measurement result for an instance."""
        try:
            length_cm = self.aruco_detector.convert_pixels_to_length(instance.fil_length_pixels)
            if length_cm:
                logger.info(f"Frame {instance.process_id}: {instance.fil_length_pixels:.2f}px, {length_cm:.2f}cm")
            else:
                logger.info(f"Frame {instance.process_id}: {instance.fil_length_pixels:.2f}px (no calibration)")
            
            if hasattr(instance, 'output_log'):
                for message in instance.output_log:
                    logger.debug(message)
                    
        except Exception as e:
            logger.warning(f"Failed to log measurement result: {e}")
    
    def _save_measurement_images(self, instance: ProcessingInstance) -> None:
        """Save images for a measurement instance."""
        try:
            # Save raw frame
            raw_path = self.folder_manager.get_raw_path(instance.process_id, self.image_format)
            cv2.imwrite(str(raw_path), instance.raw_frame)
            
            # Save skeleton/longpath frame
            skeleton_path = self.folder_manager.get_skeleton_path(instance.process_id, self.image_format)
            cv2.imwrite(str(skeleton_path), instance.skeleton_contour)
            
        except Exception as e:
            logger.warning(f"Failed to save images for frame {instance.process_id}: {e}")
    
    def _finalize_analysis(self, measurements: List[ProcessingInstance]) -> None:
        """Finalize the analysis with statistics and export."""
        self.measurements = measurements
        
        if not self.config.should_stop():
            # Filter outliers and calculate statistics
            refined_measurements = MeasurementStatistics.filter_outliers(measurements)
            self.config.set_trial_count(len(refined_measurements))
            
            if len(refined_measurements) < 2:
                self._handle_insufficient_data_error()
            else:
                # Export data and images
                watermarker = ImageWatermarker(self.config)
                MeasurementStatistics.export_data(
                    measurements=refined_measurements,
                    folder_manager=self.folder_manager,
                    image_format=self.image_format,
                    watermarker=watermarker
                )
    
    def _handle_no_instances_error(self) -> None:
        """Handle case where no processing instances were created."""
        error_msg = (
            "No length values could be obtained from the collected images. "
            "Either the blob was too small and filtered out, or the "
            "skeletonization process was too complex and failed. Please try again."
        )
        self.config.add_error("interrupt", error_msg)
        logger.error(error_msg)
    
    def _handle_no_measurements_error(self) -> None:
        """Handle case where no successful measurements were obtained."""
        error_msg = (
            "No successful measurements could be processed. "
            "Please check image quality and try again."
        )
        self.config.add_error("interrupt", error_msg)
        logger.error(error_msg)
    
    def _handle_insufficient_data_error(self) -> None:
        """Handle case where measurements are too variant."""
        error_msg = (
            "The lengths obtained are too variant to be consolidated "
            "(the variance is too high). The data is unreliable and will "
            "not be saved, please re-measure."
        )
        self.config.add_error("interrupt", error_msg)
        logger.error(error_msg)
    
    @property
    def is_ready_for_measurement(self) -> bool:
        """Check if the instance is ready to perform measurements."""
        return (
            self.background_is_trained and
            not self.block_tkinter_start_button and
            not self.pulling_background and
            self.output_folder is not None and
            self.image_format is not None
        )
    
    def reset(self) -> None:
        """Reset the instance for a new measurement session."""
        self.measurements = []
        self.background_is_trained = False
        self.block_tkinter_start_button = False
        self.pulling_background = False
        self.background_subtractor = None
        self.binary_mask = None
        self.config.reset()
        self.folder_manager.reset()