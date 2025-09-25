"""
Long path element for fish skeleton measurement processing.

This module defines the LongPathElement class which represents a segment 
of a fish skeleton path with intersection and endpoint information.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from . import point_utils


class LongPathElement:
    """
    Represents a branch element in a fish skeleton's longest path.
    
    This class handles the processing and measurement of individual skeleton branches,
    including ordering points, trimming based on head/tail positions, and calculating lengths.
    """
    
    def __init__(
        self,
        branch_index: int,
        intersection_points: np.ndarray,
        intersection_point_indices: np.ndarray,
        end_points: Optional[np.ndarray] = None,
        end_point_indices: Optional[np.ndarray] = None
    ):
        """
        Initialize a LongPathElement.
        
        Args:
            branch_index: Index identifying this branch
            intersection_points: Array of intersection point coordinates
            intersection_point_indices: Indices of intersection points
            end_points: Array of endpoint coordinates (optional)
            end_point_indices: Indices of endpoints (optional)
        """
        self.branch_index = branch_index
        self.intersection_points = np.asarray(intersection_points) if intersection_points is not None else np.array([])
        self.intersection_point_indices = np.asarray(intersection_point_indices) if intersection_point_indices is not None else np.array([])
        self.end_points = np.asarray(end_points) if end_points is not None else np.array([])
        self.end_point_indices = np.asarray(end_point_indices) if end_point_indices is not None else np.array([])
        
        # Connection points (set externally by the Measurement class)
        self.head_point: Optional[Tuple[int, int]] = None
        self.tail_point: Optional[Tuple[int, int]] = None
        
        # Processing results (set by process_element method)
        self.ordered_branch_points_full: Optional[np.ndarray] = None
        self.ordered_branch_points_adjusted: Optional[np.ndarray] = None
        self.total_pixel_length: float = 0.0
        self.total_adjusted_length: float = 0.0
        self.unit_length: float = 0.0
        
        # Processing messages for debugging/logging
        self.return_messages: List[str] = []
    
    def process_element(
        self, 
        base_branch_points: np.ndarray, 
        base_branch_length: float
    ) -> bool:
        """
        Process the branch element by ordering and trimming based on head/tail points.
        
        This method orders the branch points and trims them based on the head and tail
        points which serve as connectivity points with adjacent branches.
        
        Args:
            base_branch_points: Array of (y, x) pixel coordinates making up the branch
            base_branch_length: Branch length as calculated by FilFinder
            
        Returns:
            True if processing completed successfully, False otherwise
            
        Raises:
            ValueError: If required points are not set or invalid
        """
        if self.head_point is None or self.tail_point is None:
            self.return_messages.append("Error: Head or tail point not set")
            return False
        
        try:
            # Validate input
            base_points = np.asarray(base_branch_points)
            if base_points.size == 0:
                self.return_messages.append("Error: Empty branch points array")
                return False
            
            if base_points.ndim == 1:
                base_points = base_points.reshape(1, -1)
            
            # Optimize path ordering starting from head point
            optimized_coords = point_utils.optimize_path(
                base_points.tolist(), 
                start_point=list(self.head_point)
            )
            ordered_points = np.asarray(optimized_coords)
            
            # Initialize attributes
            self.ordered_branch_points_full = ordered_points
            self.ordered_branch_points_adjusted = ordered_points.copy()
            self.total_pixel_length = float(base_branch_length)
            self.total_adjusted_length = float(base_branch_length)
            
            # Calculate unit length (length per point)
            num_points = len(self.ordered_branch_points_full)
            self.unit_length = self.total_pixel_length / max(num_points, 1)
            
            self.return_messages.append(
                f"Branch contains {num_points} points and is {self.total_pixel_length:.2f} pixels long"
            )
            
            # Trim based on tail point position
            if not self._trim_at_tail():
                return False
                
            # Trim based on head point position  
            if not self._trim_at_head():
                return False
            
            return True
            
        except Exception as e:
            self.return_messages.append(f"Error processing element: {str(e)}")
            return False
    
    def _trim_at_tail(self) -> bool:
        """
        Trim the branch from the tail point if it's not at the end.
        
        Returns:
            True if trimming successful or not needed, False on error
        """
        try:
            if len(self.ordered_branch_points_adjusted) == 0:
                return False
                
            last_point = self.ordered_branch_points_adjusted[-1]
            
            if not point_utils.is_point_in_neighborhood(self.tail_point, tuple(last_point)):
                # Tail point is mid-branch, need to trim
                tail_kernel_coords = self._get_neighborhood_coords(self.tail_point, (5, 5))
                tail_matches = point_utils.contains_mutual_points(
                    tail_kernel_coords, 
                    self.ordered_branch_points_adjusted
                )
                
                if not np.any(tail_matches):
                    self.return_messages.append("Error: Could not find tail point in branch")
                    return False
                
                # Find the last matching index
                tail_index = np.where(tail_matches)[0][-1]
                
                # Calculate trimmed length
                points_to_trim = len(self.ordered_branch_points_adjusted) - tail_index - 1
                trimmed_length = points_to_trim * self.unit_length
                
                # Apply trimming
                self.total_adjusted_length -= trimmed_length
                self.ordered_branch_points_adjusted = self.ordered_branch_points_adjusted[:tail_index + 1]
                
                self.return_messages.append(f"Tail is mid-branch, trimmed {trimmed_length:.2f} pixels")
            else:
                self.return_messages.append("Tail point is at the end of the branch")
                
            return True
            
        except Exception as e:
            self.return_messages.append(f"Error trimming at tail: {str(e)}")
            return False
    
    def _trim_at_head(self) -> bool:
        """
        Trim the branch from the head point if it's not at the beginning.
        
        Returns:
            True if trimming successful or not needed, False on error
        """
        try:
            if len(self.ordered_branch_points_adjusted) == 0:
                return False
                
            first_point = self.ordered_branch_points_adjusted[0]
            
            if not point_utils.is_point_in_neighborhood(self.head_point, tuple(first_point)):
                # Head point is mid-branch, need to trim
                head_kernel_coords = self._get_neighborhood_coords(self.head_point, (5, 5))
                head_matches = point_utils.contains_mutual_points(
                    head_kernel_coords,
                    self.ordered_branch_points_adjusted
                )
                
                if not np.any(head_matches):
                    self.return_messages.append("Error: Could not find head point in branch")
                    return False
                
                # Find the first matching index
                head_index = np.where(head_matches)[0][0]
                
                # Calculate trimmed length
                trimmed_length = head_index * self.unit_length
                
                # Apply trimming
                self.total_adjusted_length -= trimmed_length
                self.ordered_branch_points_adjusted = self.ordered_branch_points_adjusted[head_index:]
                
                self.return_messages.append(f"Head is mid-branch, trimmed {trimmed_length:.2f} pixels")
            else:
                self.return_messages.append("Head point is at the beginning of the branch")
                
            return True
            
        except Exception as e:
            self.return_messages.append(f"Error trimming at head: {str(e)}")
            return False
    
    def _get_neighborhood_coords(
        self, 
        center_point: Tuple[int, int], 
        kernel_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Get coordinates of pixels in the neighborhood of a center point.
        
        Args:
            center_point: Center point as (y, x) tuple
            kernel_size: Size of neighborhood kernel as (height, width)
            
        Returns:
            Array of neighborhood coordinate points
        """
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            kernel_indices = np.argwhere(kernel)
            offset = ((np.asarray(kernel_size) - 1) / 2).astype(np.int32)
            neighborhood_coords = kernel_indices + np.array(center_point) - offset
            return neighborhood_coords
            
        except Exception:
            # Return empty array on error
            return np.array([]).reshape(0, 2)
    
    def get_adjusted_length_cm(self, pixels_to_cm_ratio: float) -> float:
        """
        Get the adjusted length converted to centimeters.
        
        Args:
            pixels_to_cm_ratio: Conversion ratio from pixels to centimeters
            
        Returns:
            Adjusted length in centimeters
        """
        return self.total_adjusted_length * pixels_to_cm_ratio
    
    def get_point_count(self) -> int:
        """
        Get the number of points in the adjusted branch.
        
        Returns:
            Number of points in ordered_branch_points_adjusted
        """
        if self.ordered_branch_points_adjusted is not None:
            return len(self.ordered_branch_points_adjusted)
        return 0
    
    def is_processed(self) -> bool:
        """
        Check if the element has been successfully processed.
        
        Returns:
            True if element has been processed, False otherwise
        """
        return (
            self.ordered_branch_points_full is not None and
            self.ordered_branch_points_adjusted is not None and
            self.total_pixel_length > 0
        )
    
    def __repr__(self) -> str:
        """String representation of the LongPathElement."""
        return (
            f"LongPathElement(branch_index={self.branch_index}, "
            f"points={self.get_point_count()}, "
            f"length={self.total_adjusted_length:.2f}px)"
        )