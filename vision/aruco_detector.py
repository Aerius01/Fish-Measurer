"""
Modern ArUco marker detection module.

This module provides ArUco marker detection functionality using the updated OpenCV API
with proper error handling and type safety.
"""

from __future__ import annotations

import logging
import statistics
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ArUcoMarker:
    """Represents a detected ArUco marker."""
    id: int
    corners: np.ndarray
    center: Tuple[int, int]
    size_pixels: float


@dataclass
class CalibrationData:
    """Calibration data derived from ArUco markers."""
    slope: float
    intercept: float
    marker_count: int


class ArUcoDetector:
    """
    Modern ArUco marker detector with updated OpenCV API.
    
    This class handles ArUco marker detection and calibration parameter calculation
    using the modern OpenCV ArUco API.
    """
    
    def __init__(self, dictionary_type: int = cv2.aruco.DICT_4X4_50):
        self.logger = logging.getLogger(__name__)
        
        # Initialize ArUco detector with modern API
        try:
            self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        except AttributeError:
            # Fallback for older OpenCV versions
            self.logger.warning("Using legacy ArUco API")
            self.dictionary = cv2.aruco.Dictionary_get(dictionary_type)
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.detector = None
        
        # Calibration data storage
        self._slope_history: List[float] = []
        self._intercept_history: List[float] = []
        self._max_history_size = 50  # Keep last 50 measurements
    
    def detect_markers(self, image: np.ndarray) -> List[ArUcoMarker]:
        """
        Detect ArUco markers in the given image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected ArUco markers
        """
        try:
            if self.detector is not None:
                # Modern API
                corners, ids, rejected = self.detector.detectMarkers(image)
            else:
                # Legacy API fallback
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    image, self.dictionary, parameters=self.parameters
                )
            
            markers = []
            if ids is not None and len(corners) > 0:
                ids = ids.flatten()
                
                for corner_set, marker_id in zip(corners, ids):
                    marker = self._create_marker_from_corners(corner_set, marker_id)
                    markers.append(marker)
            
            return markers
            
        except Exception as e:
            self.logger.error(f"ArUco detection failed: {e}")
            return []
    
    def _create_marker_from_corners(self, corners: np.ndarray, marker_id: int) -> ArUcoMarker:
        """Create ArUcoMarker object from detected corners."""
        # Reshape corners and convert to integers
        corner_points = corners.reshape((4, 2))
        top_left, top_right, bottom_right, bottom_left = corner_points
        
        # Convert to integers
        top_left = (int(top_left[0]), int(top_left[1]))
        top_right = (int(top_right[0]), int(top_right[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        
        # Calculate center
        center_x = int(np.mean([p[0] for p in [top_left, top_right, bottom_right, bottom_left]]))
        center_y = int(np.mean([p[1] for p in [top_left, top_right, bottom_right, bottom_left]]))
        center = (center_x, center_y)
        
        # Calculate size (distance between top corners)
        size_pixels = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
        
        return ArUcoMarker(
            id=int(marker_id),
            corners=corner_points,
            center=center,
            size_pixels=float(size_pixels)
        )
    
    def draw_markers(self, image: np.ndarray, markers: List[ArUcoMarker], 
                    color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw detected markers on the image.
        
        Args:
            image: Input image
            markers: List of detected markers
            color: Color for drawing (BGR format)
            thickness: Line thickness
            
        Returns:
            Image with drawn markers
        """
        result_image = image.copy()
        
        for marker in markers:
            corners = marker.corners.astype(int)
            
            # Draw bounding box
            cv2.line(result_image, tuple(corners[0]), tuple(corners[1]), color, thickness)
            cv2.line(result_image, tuple(corners[1]), tuple(corners[2]), color, thickness)
            cv2.line(result_image, tuple(corners[2]), tuple(corners[3]), color, thickness)
            cv2.line(result_image, tuple(corners[3]), tuple(corners[0]), color, thickness)
            
            # Draw marker ID
            cv2.putText(result_image, str(marker.id), marker.center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        return result_image
    
    def calculate_calibration(self, markers: List[ArUcoMarker]) -> Optional[CalibrationData]:
        """
        Calculate calibration parameters from detected markers.
        
        Args:
            markers: List of detected markers
            
        Returns:
            CalibrationData if successful, None otherwise
        """
        if len(markers) < 2:
            return None
        
        try:
            # Extract marker data
            marker_ids = [m.id for m in markers]
            marker_sizes = [m.size_pixels for m in markers]
            
            # Find min and max
            max_id = max(marker_ids)
            min_id = min(marker_ids)
            max_size_idx = marker_ids.index(max_id)
            min_size_idx = marker_ids.index(min_id)
            max_pixel_dist = marker_sizes[max_size_idx]
            min_pixel_dist = marker_sizes[min_size_idx]
            
            # Calculate calibration parameters
            slope = (max_id - min_id) / (max_pixel_dist - min_pixel_dist)
            intercept = max_id - slope * max_pixel_dist
            
            # Store in history (with size limit)
            self._slope_history.append(slope)
            self._intercept_history.append(intercept)
            
            if len(self._slope_history) > self._max_history_size:
                self._slope_history.pop(0)
            if len(self._intercept_history) > self._max_history_size:
                self._intercept_history.pop(0)
            
            return CalibrationData(
                slope=slope,
                intercept=intercept,
                marker_count=len(markers)
            )
            
        except Exception as e:
            self.logger.error(f"Calibration calculation failed: {e}")
            return None
    
    def get_average_calibration(self) -> Optional[CalibrationData]:
        """Get average calibration parameters from history."""
        if not self._slope_history or not self._intercept_history:
            return None
        
        return CalibrationData(
            slope=statistics.mean(self._slope_history),
            intercept=statistics.mean(self._intercept_history),
            marker_count=len(self._slope_history)
        )
    
    def convert_pixels_to_length(self, pixels: float) -> Optional[float]:
        """Convert pixels to length using current calibration."""
        calibration = self.get_average_calibration()
        if calibration is None:
            return None
        
        return calibration.slope * pixels + calibration.intercept
    
    def convert_length_to_pixels(self, length: float) -> Optional[float]:
        """Convert length to pixels using current calibration."""
        calibration = self.get_average_calibration()
        if calibration is None:
            return None
        
        return (length - calibration.intercept) / calibration.slope
    
    # Backwards-compatible alias expected by orchestrator
    def clear_calibration(self) -> None:
        """Clear stored calibration samples/history (alias)."""
        self.clear_calibration_history()
    
    def clear_calibration_history(self) -> None:
        """Clear calibration history."""
        self._slope_history.clear()
        self._intercept_history.clear()
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get statistics about current calibration."""
        if not self._slope_history:
            return {}
        
        return {
            'slope_mean': statistics.mean(self._slope_history),
            'slope_stdev': statistics.stdev(self._slope_history) if len(self._slope_history) > 1 else 0,
            'intercept_mean': statistics.mean(self._intercept_history),
            'intercept_stdev': statistics.stdev(self._intercept_history) if len(self._intercept_history) > 1 else 0,
            'sample_count': len(self._slope_history)
        }


def generate_aruco_marker(marker_id: int, size: int, 
                         dictionary_type: int = cv2.aruco.DICT_4X4_50) -> np.ndarray:
    """
    Generate an ArUco marker image.
    
    Args:
        marker_id: ID of the marker to generate
        size: Size of the marker in pixels
        dictionary_type: ArUco dictionary type
        
    Returns:
        Generated marker as numpy array
    """
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
    except AttributeError:
        # Fallback for older OpenCV versions
        dictionary = cv2.aruco.Dictionary_get(dictionary_type)
    
    marker_image = np.zeros((size, size, 1), dtype="uint8")
    cv2.aruco.generateImageMarker(dictionary, marker_id, size, marker_image, 1)
    
    return marker_image
