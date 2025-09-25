"""
Image watermarking utilities for fish measurement results.

This module handles the creation of watermarked images with measurement data,
timestamps, and other metadata overlays using modern OpenCV features.
"""

from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from vision import ArUcoDetector
from datetime import datetime
import logging

import cv2
import numpy as np
from .measurement_config import MeasurementConfig


logger = logging.getLogger(__name__)


class ImageWatermarker:
    """
    Creates watermarked images with measurement data and metadata.
    
    This class handles the overlay of text information on measurement images
    including lengths, timestamps, fish IDs, and statistical data.
    """
    
    def __init__(self, config: MeasurementConfig):
        """
        Initialize the watermarker with configuration.
        
        Args:
            config: Measurement configuration instance
        """
        self.config = config
        
        # Text styling parameters - optimized for modern OpenCV
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale_large = 2.0
        self.font_scale_medium = 1.5
        self.font_scale_small = 1.0
        self.color_white = (255, 255, 255)
        self.color_yellow = (0, 255, 255)  # BGR format
        self.thickness_thick = 3
        self.thickness_medium = 2
        self.thickness_thin = 1
        self.line_type = cv2.LINE_AA
        
        # Layout parameters
        self.margin = 15
        self.line_height = 70
        self.section_gap = 20
    
    def create_watermarked_image(
        self,
        measurement_instance: Any,
        statistics: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Create a watermarked image with measurement data.
        
        Args:
            measurement_instance: ProcessingInstance with measurement data
            statistics: Optional statistical data for detailed watermark
            
        Returns:
            Watermarked image as numpy array
            
        Raises:
            ValueError: If measurement instance is invalid
        """
        if not hasattr(measurement_instance, 'processed_frame'):
            raise ValueError("Measurement instance must have processed_frame attribute")
        
        try:
            # Start with processed frame
            watermarked = measurement_instance.processed_frame.copy()
            
            if watermarked is None or watermarked.size == 0:
                raise ValueError("Processed frame is empty")
            
            # Ensure image is in BGR format for color text
            if len(watermarked.shape) == 2:
                watermarked = cv2.cvtColor(watermarked, cv2.COLOR_GRAY2BGR)
            elif watermarked.shape[2] == 1:
                watermarked = cv2.cvtColor(watermarked, cv2.COLOR_GRAY2BGR)
            
            # Add watermark elements
            y_position = self._add_timestamp(watermarked)
            y_position = self._add_length_info(watermarked, measurement_instance, statistics, y_position)
            y_position = self._add_trial_count(watermarked, statistics, y_position)
            y_position = self._add_fish_id(watermarked, y_position)
            y_position = self._add_additional_text(watermarked, y_position)
            
            # Add statistics if provided
            if statistics and len(statistics) > 3:  # Only if we have meaningful stats
                self._add_statistics_overlay(watermarked, statistics)
            
            return watermarked
            
        except Exception as e:
            logger.error(f"Watermarking failed: {e}")
            # Return original image if watermarking fails
            return measurement_instance.processed_frame.copy()
    
    def _add_timestamp(self, image: np.ndarray) -> int:
        """Add timestamp to image."""
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        
        position = (self.margin, self.line_height)
        
        cv2.putText(
            image, timestamp, position,
            self.font, self.font_scale_large, self.color_white,
            self.thickness_medium, self.line_type
        )
        
        return position[1] + self.line_height
    
    def _add_length_info(
        self,
        image: np.ndarray,
        measurement_instance: Any,
        statistics: Optional[Dict[str, float]],
        y_start: int
    ) -> int:
        """Add length information to image."""
        try:
            # Get calibration from global ArUco detector instance
            aruco_detector = getattr(self.config, '_aruco_detector', None)
            if aruco_detector:
                length_cm = aruco_detector.convert_pixels_to_length(measurement_instance.fil_length_pixels)
            else:
                length_cm = None
            
            if statistics and "mean_cm" in statistics:
                # Detailed format with statistics
                mean_cm = statistics["mean_cm"]
                std_cm = statistics.get("std_cm", 0)
                
                length_text = (
                    f"Avg: {mean_cm:.2f}cm ± {std_cm:.2f}cm "
                    f"(This: {length_cm:.2f}cm)"
                )
                color = self.color_yellow
                font_scale = self.font_scale_medium
            else:
                # Simple format
                length_text = f"Length: {length_cm:.2f} cm"
                color = self.color_white
                font_scale = self.font_scale_large
            
            position = (self.margin, y_start + self.line_height)
            
            cv2.putText(
                image, length_text, position,
                self.font, font_scale, color,
                self.thickness_medium, self.line_type
            )
            
            return position[1]
            
        except Exception as e:
            logger.warning(f"Failed to add length info: {e}")
            return y_start
    
    def _add_trial_count(
        self,
        image: np.ndarray,
        statistics: Optional[Dict[str, float]],
        y_start: int
    ) -> int:
        """Add trial count information."""
        if not statistics or "count" not in statistics:
            trial_count = self.config.get_trial_count()
            if trial_count <= 0:
                return y_start
        else:
            trial_count = int(statistics["count"])
        
        trial_text = f"{trial_count} image{'s' if trial_count != 1 else ''} analyzed"
        position = (self.margin, image.shape[0] - 120)  # Near bottom
        
        cv2.putText(
            image, trial_text, position,
            self.font, self.font_scale_large, self.color_white,
            self.thickness_medium, self.line_type
        )
        
        return position[1]
    
    def _add_fish_id(self, image: np.ndarray, y_start: int) -> int:
        """Add fish ID if available."""
        fish_id = self.config.get_fish_id()
        
        if not fish_id:
            return y_start
        
        fish_text = f"Fish ID: {fish_id}"
        position = (self.margin, y_start + self.line_height)
        
        cv2.putText(
            image, fish_text, position,
            self.font, self.font_scale_large, self.color_white,
            self.thickness_medium, self.line_type
        )
        
        return position[1]
    
    def _add_additional_text(self, image: np.ndarray, y_start: int) -> int:
        """Add user-specified additional text."""
        additional_text = self.config.get_additional_text()
        
        if not additional_text:
            return y_start
        
        # Handle multi-line text
        lines = additional_text.split('\n')
        current_y = y_start + self.line_height + self.section_gap
        
        for line in lines:
            if line.strip():  # Skip empty lines
                position = (self.margin, current_y)
                
                cv2.putText(
                    image, line.strip(), position,
                    self.font, self.font_scale_large, self.color_white,
                    self.thickness_medium, self.line_type
                )
                
                current_y += self.line_height
        
        return current_y
    
    def _add_statistics_overlay(
        self,
        image: np.ndarray,
        statistics: Dict[str, float]
    ) -> None:
        """Add statistical overlay in corner."""
        try:
            # Create semi-transparent overlay
            overlay = image.copy()
            
            # Statistics text
            stats_lines = [
                f"Mean: {statistics.get('mean_cm', 0):.2f}cm",
                f"Std: {statistics.get('std_cm', 0):.2f}cm",
                f"CV: {statistics.get('cv_percent', 0):.1f}%",
                f"N: {statistics.get('count', 0)}"
            ]
            
            # Calculate overlay size
            max_width = 0
            total_height = 0
            
            for line in stats_lines:
                (text_width, text_height), baseline = cv2.getTextSize(
                    line, self.font, self.font_scale_small, self.thickness_thin
                )
                max_width = max(max_width, text_width)
                total_height += text_height + 10
            
            # Position in top-right corner
            overlay_x = image.shape[1] - max_width - 20
            overlay_y = 20
            
            # Draw semi-transparent background
            cv2.rectangle(
                overlay,
                (overlay_x - 10, overlay_y - 10),
                (overlay_x + max_width + 10, overlay_y + total_height + 10),
                (0, 0, 0), -1
            )
            
            # Blend overlay
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            
            # Add statistics text
            current_y = overlay_y + 25
            for line in stats_lines:
                cv2.putText(
                    image, line, (overlay_x, current_y),
                    self.font, self.font_scale_small, self.color_white,
                    self.thickness_thin, self.line_type
                )
                current_y += 35
                
        except Exception as e:
            logger.warning(f"Failed to add statistics overlay: {e}")
    
    def get_text_size(self, text: str, font_scale: float) -> Tuple[int, int]:
        """
        Get the size of text when rendered.
        
        Args:
            text: Text to measure
            font_scale: Font scale to use
            
        Returns:
            Tuple of (width, height) in pixels
        """
        try:
            (width, height), _ = cv2.getTextSize(
                text, self.font, font_scale, self.thickness_medium
            )
            return width, height
        except Exception:
            return 0, 0
    
    def create_info_panel(
        self,
        width: int,
        height: int,
        measurement_instance: Any,
        statistics: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Create a separate information panel image.
        
        Args:
            width: Panel width
            height: Panel height  
            measurement_instance: Measurement data
            statistics: Optional statistics
            
        Returns:
            Information panel as numpy array
        """
        # Create black panel
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        try:
            y_pos = 30
            
            # Title
            cv2.putText(
                panel, "Measurement Results", (20, y_pos),
                self.font, self.font_scale_large, self.color_white,
                self.thickness_medium, self.line_type
            )
            y_pos += 80
            
            # Basic info
            aruco_detector = getattr(self.config, '_aruco_detector', None)
            if aruco_detector:
                length_cm = aruco_detector.convert_pixels_to_length(measurement_instance.fil_length_pixels)
            else:
                length_cm = None
            info_lines = [
                f"Frame: {measurement_instance.process_id}",
                f"Length: {length_cm:.3f} cm",
                f"Pixels: {measurement_instance.fil_length_pixels:.1f} px",
            ]
            
            if statistics:
                info_lines.extend([
                    "",
                    "Statistics:",
                    f"Mean: {statistics.get('mean_cm', 0):.3f} cm",
                    f"Std Dev: {statistics.get('std_cm', 0):.3f} cm",
                    f"Count: {statistics.get('count', 0)}",
                    f"CV: {statistics.get('cv_percent', 0):.1f}%"
                ])
            
            for line in info_lines:
                if line:  # Skip empty lines
                    cv2.putText(
                        panel, line, (20, y_pos),
                        self.font, self.font_scale_medium, self.color_white,
                        self.thickness_thin, self.line_type
                    )
                y_pos += 50
                
        except Exception as e:
            logger.error(f"Info panel creation failed: {e}")
        
        return panel