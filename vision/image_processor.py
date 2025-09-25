"""
Core image processing functionality.

This module handles the fundamental image processing operations including
morphological operations, skeletonization, and contour detection.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

import cv2
import numpy as np
from skimage import img_as_bool
from skimage.morphology import medial_axis, binary_closing, binary_opening


@dataclass
class ProcessingResult:
    """Result of image processing operations."""
    skeleton: np.ndarray
    distance_transform: np.ndarray
    contour: np.ndarray
    dimensions: Tuple[int, int]
    success: bool
    error_message: Optional[str] = None


class ImageProcessor:
    """
    Core image processing operations for fish measurement.
    
    This class handles morphological operations, skeletonization,
    and contour detection with proper error handling.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_binary_image(self, binary_frame: np.ndarray) -> ProcessingResult:
        """
        Process a binary image to extract skeleton and contours.
        
        Args:
            binary_frame: Input binary image
            
        Returns:
            ProcessingResult with skeleton, distance transform, and contour
        """
        try:
            if binary_frame is None:
                return ProcessingResult(
                    skeleton=np.array([]),
                    distance_transform=np.array([]),
                    contour=np.array([]),
                    dimensions=(0, 0),
                    success=False,
                    error_message="Input frame is None"
                )
            
            dimensions = np.shape(binary_frame)
            contour = np.zeros(dimensions, dtype=np.uint8)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(
                opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(contour, contours, -1, 255)
            
            # Skeletonize
            bool_image = img_as_bool(binary_opening(opening))
            skeleton, distance_transform = medial_axis(bool_image, return_distance=True)
            skeleton = ((binary_closing(skeleton)) * 255).astype('uint8')
            
            self.logger.debug(f"Processed image with dimensions {dimensions}")
            
            return ProcessingResult(
                skeleton=skeleton,
                distance_transform=distance_transform,
                contour=contour,
                dimensions=dimensions,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return ProcessingResult(
                skeleton=np.array([]),
                distance_transform=np.array([]),
                contour=np.array([]),
                dimensions=(0, 0),
                success=False,
                error_message=str(e)
            )
    
    def enhance_skeleton(self, skeleton: np.ndarray, 
                        iterations: int = 1) -> np.ndarray:
        """
        Enhance skeleton connectivity using morphological operations.
        
        Args:
            skeleton: Input skeleton image
            iterations: Number of dilation iterations
            
        Returns:
            Enhanced skeleton
        """
        try:
            if skeleton is None or skeleton.size == 0:
                return skeleton
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            enhanced = cv2.dilate(skeleton, kernel, iterations=iterations)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Skeleton enhancement failed: {e}")
            return skeleton
    
    def create_visualization_overlay(self, 
                                   raw_frame: np.ndarray,
                                   skeleton: np.ndarray,
                                   contour: np.ndarray,
                                   long_path: Optional[np.ndarray] = None,
                                   alpha: float = 0.7) -> np.ndarray:
        """
        Create a visualization overlay combining multiple processing elements.
        
        Args:
            raw_frame: Original raw frame
            skeleton: Skeleton image
            contour: Contour image  
            long_path: Optional long path visualization
            alpha: Blending factor
            
        Returns:
            Combined visualization image
        """
        try:
            if raw_frame is None or skeleton is None or contour is None:
                return raw_frame if raw_frame is not None else np.zeros((100, 100, 3))
            
            # Ensure raw frame is grayscale
            if len(raw_frame.shape) == 3:
                raw_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            else:
                raw_gray = raw_frame.copy()
            
            # Combine skeleton and contour
            skeleton_contour = (skeleton + contour).astype('uint8')
            
            # Add long path if provided
            if long_path is not None:
                long_path_dilated = cv2.dilate(
                    np.rint(long_path * 255).astype('uint8'),
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=1
                )
                overlay = cv2.addWeighted(long_path_dilated, 0.3, skeleton_contour, 0.7, 0)
            else:
                overlay = skeleton_contour
            
            # Blend with raw frame
            result = cv2.addWeighted(overlay, alpha, raw_gray, 1.0 - alpha, 0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Visualization overlay creation failed: {e}")
            return raw_frame if raw_frame is not None else np.zeros((100, 100, 3))
    
    def validate_processing_result(self, result: ProcessingResult) -> bool:
        """
        Validate that processing result contains meaningful data.
        
        Args:
            result: Processing result to validate
            
        Returns:
            True if result is valid, False otherwise
        """
        if not result.success:
            return False
        
        if result.skeleton is None or result.skeleton.size == 0:
            self.logger.warning("Skeleton is empty")
            return False
        
        if result.distance_transform is None or result.distance_transform.size == 0:
            self.logger.warning("Distance transform is empty")
            return False
        
        # Check if skeleton has reasonable content
        skeleton_pixels = np.sum(result.skeleton > 0)
        if skeleton_pixels < 10:  # Minimum threshold
            self.logger.warning(f"Skeleton has too few pixels: {skeleton_pixels}")
            return False
        
        return True
    
    def get_processing_stats(self, result: ProcessingResult) -> dict:
        """
        Get statistics about the processing result.
        
        Args:
            result: Processing result
            
        Returns:
            Dictionary with processing statistics
        """
        if not result.success:
            return {'success': False, 'error': result.error_message}
        
        try:
            skeleton_pixels = np.sum(result.skeleton > 0)
            contour_pixels = np.sum(result.contour > 0)
            avg_distance = np.mean(result.distance_transform[result.skeleton > 0])
            max_distance = np.max(result.distance_transform)
            
            return {
                'success': True,
                'dimensions': result.dimensions,
                'skeleton_pixels': int(skeleton_pixels),
                'contour_pixels': int(contour_pixels),
                'average_distance_to_boundary': float(avg_distance),
                'max_distance_to_boundary': float(max_distance)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compute processing stats: {e}")
            return {'success': False, 'error': str(e)}
