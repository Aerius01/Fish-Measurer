"""
Display and annotation management module.

This module handles image display, annotations, and UI-related image processing
with separation of concerns from camera and processing logic.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple, List, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from .aruco_detector import ArUcoDetector, ArUcoMarker


class DisplayManager:
    """
    Manages image display and annotations.
    
    This class handles all display-related operations including text overlays,
    ArUco marker visualization, and image scaling for UI display.
    """
    
    def __init__(self, scale_factor: float = 0.39):
        self.logger = logging.getLogger(__name__)
        self.scale_factor = scale_factor
        self.aruco_detector = ArUcoDetector()
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.7
        self.font_color = (255, 255, 255)
        self.font_thickness = 1
        self.line_type = cv2.LINE_AA
    
    def create_display_image(self, 
                           frame: np.ndarray,
                           fish_id: Optional[str] = None,
                           additional_text: Optional[str] = None,
                           show_aruco: bool = True,
                           show_timestamp: bool = True) -> Optional[ImageTk.PhotoImage]:
        """
        Create a display-ready image with annotations.
        
        Args:
            frame: Input frame
            fish_id: Fish ID to display
            additional_text: Additional text to overlay
            show_aruco: Whether to detect and show ArUco markers
            show_timestamp: Whether to show timestamp
            
        Returns:
            PIL ImageTk.PhotoImage ready for display, or None if failed
        """
        if frame is None:
            return None
        
        try:
            display_frame = frame.copy()
            
            # Detect and draw ArUco markers if requested
            if show_aruco:
                markers = self.aruco_detector.detect_markers(display_frame)
                if markers:
                    display_frame = self.aruco_detector.draw_markers(display_frame, markers)
                    
                    # Update calibration
                    calibration = self.aruco_detector.calculate_calibration(markers)
                    if calibration:
                        self.logger.debug(f"Updated calibration: slope={calibration.slope:.4f}, "
                                        f"intercept={calibration.intercept:.4f}")
            
            # Scale image for display
            display_frame = cv2.resize(display_frame, None, 
                                     fx=self.scale_factor, fy=self.scale_factor)
            
            # Add text overlays
            display_frame = self._add_text_overlays(
                display_frame, fish_id, additional_text, show_timestamp
            )
            
            # Convert to PIL Image for Tkinter
            pil_image = Image.fromarray(display_frame)
            return ImageTk.PhotoImage(pil_image)
            
        except Exception as e:
            self.logger.error(f"Failed to create display image: {e}")
            return None
    
    def _add_text_overlays(self, 
                          image: np.ndarray,
                          fish_id: Optional[str],
                          additional_text: Optional[str],
                          show_timestamp: bool) -> np.ndarray:
        """Add text overlays to the image."""
        y_offset = 25
        
        # Timestamp
        if show_timestamp:
            timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            cv2.putText(image, timestamp, (15, y_offset), 
                       self.font, self.font_scale, self.font_color, 
                       self.font_thickness, self.line_type)
            y_offset += 35
        
        # Fish ID
        if fish_id and fish_id.strip():
            cv2.putText(image, f"Fish ID: {fish_id}", (15, y_offset), 
                       self.font, self.font_scale, self.font_color, 
                       self.font_thickness, self.line_type)
            y_offset += 35
        
        # Additional text (multi-line support)
        if additional_text and additional_text.strip():
            lines = additional_text.split('\n')
            for line in lines:
                if line.strip():
                    cv2.putText(image, line, (15, y_offset), 
                               self.font, self.font_scale, self.font_color, 
                               self.font_thickness, self.line_type)
                    y_offset += 25
        
        return image
    
    def get_calibration_info(self) -> dict:
        """Get current calibration information."""
        return self.aruco_detector.get_calibration_stats()
    
    def convert_pixels_to_length(self, pixels: float) -> Optional[float]:
        """Convert pixels to length using current calibration."""
        return self.aruco_detector.convert_pixels_to_length(pixels)
    
    def convert_length_to_pixels(self, length: float) -> Optional[float]:
        """Convert length to pixels using current calibration."""
        return self.aruco_detector.convert_length_to_pixels(length)
    
    def clear_calibration(self) -> None:
        """Clear calibration history."""
        self.aruco_detector.clear_calibration_history()
    
    def create_processing_visualization(self, 
                                      raw_frame: np.ndarray,
                                      processed_frame: np.ndarray,
                                      skeleton: Optional[np.ndarray] = None,
                                      contour: Optional[np.ndarray] = None,
                                      long_path: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a visualization combining multiple processing stages.
        
        Args:
            raw_frame: Original raw frame
            processed_frame: Processed/binary frame
            skeleton: Skeleton image
            contour: Contour image
            long_path: Long path visualization
            
        Returns:
            Combined visualization image
        """
        try:
            if raw_frame is None or processed_frame is None:
                return raw_frame if raw_frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Convert raw frame to grayscale if needed
            if len(raw_frame.shape) == 3:
                raw_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            else:
                raw_gray = raw_frame.copy()
            
            # Create base visualization
            if skeleton is not None and contour is not None:
                # Combine skeleton and contour
                skeleton_contour = (skeleton + contour).astype(np.uint8)
                
                if long_path is not None:
                    # Add long path with dilation for visibility
                    long_path_dilated = cv2.dilate(
                        np.rint(long_path * 255).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                        iterations=1
                    )
                    result = cv2.addWeighted(long_path_dilated, 0.3, skeleton_contour, 0.7, 0)
                else:
                    result = skeleton_contour
                
                # Blend with raw frame
                result = cv2.addWeighted(result, 0.65, raw_gray, 0.35, 0)
            else:
                # Simple blend of processed and raw
                if len(processed_frame.shape) == 3:
                    processed_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                else:
                    processed_gray = processed_frame
                
                result = cv2.addWeighted(processed_gray, 0.6, raw_gray, 0.4, 0)
            
            # Convert back to color if needed for display
            if len(result.shape) == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create processing visualization: {e}")
            return raw_frame if raw_frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)
    
    def save_annotated_image(self, 
                           frame: np.ndarray,
                           filepath: str,
                           annotations: Optional[dict] = None) -> bool:
        """
        Save an annotated image to file.
        
        Args:
            frame: Image frame to save
            filepath: Output file path
            annotations: Dictionary of annotations to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if frame is None:
                return False
            
            save_frame = frame.copy()
            
            # Add annotations if provided
            if annotations:
                fish_id = annotations.get('fish_id')
                additional_text = annotations.get('additional_text')
                show_timestamp = annotations.get('show_timestamp', True)
                show_aruco = annotations.get('show_aruco', True)
                
                if show_aruco:
                    markers = self.aruco_detector.detect_markers(save_frame)
                    if markers:
                        save_frame = self.aruco_detector.draw_markers(save_frame, markers)
                
                save_frame = self._add_text_overlays(
                    save_frame, fish_id, additional_text, show_timestamp
                )
            
            # Save image
            success = cv2.imwrite(filepath, save_frame)
            if success:
                self.logger.info(f"Saved annotated image to {filepath}")
            else:
                self.logger.error(f"Failed to save image to {filepath}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving annotated image: {e}")
            return False
