"""
Path analysis and longest path construction.

This module handles the construction of the longest path through a filament
network using branch connectivity and orientation analysis.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

# Import utilities - avoid circular import by importing directly at module level
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.long_path_element import PathElement

@dataclass
class PathElement:
    """
    Path element representing a branch segment in the longest path.

    This class handles ordering and trimming of branch points based on
    head/tail connectivity points, using the original algorithm.
    """
    branch_index: int
    intersection_pts: List[Tuple[int, int]]
    intersection_indices: List[int]
    end_pts: List[Tuple[int, int]]
    end_indices: List[int]
    head_point: Optional[Tuple[int, int]] = None
    tail_point: Optional[Tuple[int, int]] = None
    ordered_branch_points_adjusted: Optional[List] = None
    ordered_branch_points_full: Optional[np.ndarray] = None
    total_pixel_length: float = 0.0
    total_adjusted_length: float = 0.0
    unit_length: float = 0.0

    def ProcessElement(self, branch_points: np.ndarray, branch_length: float) -> bool:
        """
        Order and trim branch points based on head and tail connectivity points.

        Uses greedy nearest-neighbor optimization from the head point, then trims
        any points that fall outside the head-tail segment (for mid-branch connections).

        Args:
            branch_points: Array of (y, x) coordinates for this branch
            branch_length: FilFinder-calculated branch length

        Returns:
            True if processing succeeded, False otherwise
        """
        if self.head_point is None or self.tail_point is None:
            return False

        try:
            from .path_utils import optimize_path, point_in_neighborhood, create_neighborhood_kernel, contains_mutual_points
            import cv2

            # 1. Order points using greedy nearest-neighbor from head
            ordered_points = optimize_path(branch_points.tolist(), start=self.head_point)
            self.ordered_branch_points_full = np.array(ordered_points)
            self.ordered_branch_points_adjusted = self.ordered_branch_points_full.copy()

            # 2. Calculate unit length
            self.total_pixel_length = branch_length
            self.total_adjusted_length = branch_length
            self.unit_length = self.total_pixel_length / len(self.ordered_branch_points_full)

            # 3. Trim tail if it occurs mid-branch
            if not point_in_neighborhood(self.tail_point, tuple(self.ordered_branch_points_adjusted[-1])):
                # Create kernel around tail point
                tail_kernel = create_neighborhood_kernel(self.tail_point, size=(5, 5))

                # Find where tail point matches in ordered points
                tail_bool_array = contains_mutual_points(tail_kernel, self.ordered_branch_points_adjusted)

                if np.any(tail_bool_array):
                    # Get the last matching index (furthest along the branch)
                    tail_indices = np.where(tail_bool_array)[0]
                    tail_index = tail_indices[-1]

                    # Trim everything after tail point
                    trimmed_length = (len(self.ordered_branch_points_adjusted) - tail_index) * self.unit_length
                    self.total_adjusted_length -= trimmed_length
                    self.ordered_branch_points_adjusted = self.ordered_branch_points_adjusted[:tail_index+1, :]
                else:
                    # Could not find tail point in branch - should not happen
                    return False

            # 4. Trim head if it occurs mid-branch
            if not point_in_neighborhood(self.head_point, tuple(self.ordered_branch_points_adjusted[0])):
                # Create kernel around head point
                head_kernel = create_neighborhood_kernel(self.head_point, size=(5, 5))

                # Find where head point matches in ordered points
                head_bool_array = contains_mutual_points(head_kernel, self.ordered_branch_points_adjusted)

                if np.any(head_bool_array):
                    # Get the first matching index (closest to start)
                    head_indices = np.where(head_bool_array)[0]
                    head_index = head_indices[0]

                    # Trim everything before head point
                    trimmed_length = head_index * self.unit_length
                    self.total_adjusted_length -= trimmed_length
                    self.ordered_branch_points_adjusted = self.ordered_branch_points_adjusted[head_index:, :]
                else:
                    # Could not find head point in branch - should not happen
                    return False

            # Convert back to list of tuples for compatibility
            self.ordered_branch_points_adjusted = [tuple(map(int, pt)) for pt in self.ordered_branch_points_adjusted]

            return True

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"ProcessElement failed: {e}", exc_info=True)
            return False

# Define utility functions locally to avoid circular import
def calculate_path_length(coordinates: np.ndarray) -> float:
    """Calculate path length from coordinates."""
    if len(coordinates) < 2:
        return 0.0
    distances = np.sqrt(np.sum(np.diff(coordinates, axis=0)**2, axis=1))
    return float(np.sum(distances))

def is_point_in_neighborhood(pt1: np.ndarray, pt2: np.ndarray, threshold: float = 4.0) -> bool:
    """Check if two points are within neighborhood threshold."""
    distance = np.sqrt(np.sum((pt1 - pt2)**2))
    return distance <= threshold

def contains_mutual_points(points1: np.ndarray, points2: np.ndarray, threshold: float = 4.0) -> bool:
    """Check if any points from points1 are close to any points in points2."""
    for pt1 in points1:
        for pt2 in points2:
            if is_point_in_neighborhood(pt1, pt2, threshold):
                return True
    return False


@dataclass
class PathAnalysisResult:
    """Result of path analysis."""
    success: bool
    path_coordinates: Optional[np.ndarray] = None
    path_elements: Optional[List[PathElement]] = None
    head_point: Optional[Tuple[int, int]] = None
    tail_point: Optional[Tuple[int, int]] = None
    total_length: float = 0.0
    error_message: Optional[str] = None


class PathAnalyzer:
    """
    Analyzes filament networks to construct longest paths.
    
    This class implements the algorithm for finding the longest continuous
    path through a branched filament network.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def construct_longest_path(self, 
                             filament: Any,
                             head_point: Tuple[int, int],
                             distance_transform: np.ndarray) -> PathAnalysisResult:
        """
        Construct the longest path through a filament network.
        
        Args:
            filament: FilFinder filament object
            head_point: Starting point (head) of the path
            distance_transform: Distance transform for standard length point detection
            
        Returns:
            PathAnalysisResult with path construction results
        """
        try:
            # Initialize path construction
            path_elements = []
            covered_intersection_indices = []
            
            # Find the starting branch
            base_branch_index = self._get_branch_from_point(head_point, filament)
            if not base_branch_index:
                return PathAnalysisResult(
                    success=False,
                    error_message="Could not find starting branch for head point"
                )
            
            base_branch_index = base_branch_index[0]
            self.logger.debug(f"Starting with branch {base_branch_index}")
            
            # Create first path element
            intersection_points, intersection_indices = self._get_branch_intersections(
                base_branch_index, filament, also_indices=True
            )
            end_points, end_indices = self._get_branch_endpoints(
                base_branch_index, filament, also_indices=True
            )
            
            first_element = PathElement(
                base_branch_index, intersection_points, intersection_indices,
                end_points, end_indices, head_point=head_point
            )
            path_elements.append(first_element)
            
            # Recursively build the path
            tail_point = self._recursive_path_construction(
                base_branch_index, filament, path_elements, covered_intersection_indices
            )
            
            if tail_point is None:
                return PathAnalysisResult(
                    success=False,
                    error_message="Failed to construct complete path"
                )
            
            # Order and assemble path coordinates
            path_coordinates = self._construct_ordered_coordinates(
                path_elements, filament, head_point, tail_point
            )
            
            if path_coordinates is None:
                return PathAnalysisResult(
                    success=False,
                    error_message="Failed to construct ordered coordinates"
                )
            
            # Calculate total length
            total_length = calculate_path_length(path_coordinates)
            
            self.logger.info(f"Constructed path with {len(path_coordinates)} points, "
                           f"length: {total_length:.2f} pixels")
            
            return PathAnalysisResult(
                success=True,
                path_coordinates=path_coordinates,
                path_elements=path_elements,
                head_point=head_point,
                tail_point=tail_point,
                total_length=total_length
            )
            
        except Exception as e:
            self.logger.error(f"Path construction failed: {e}")
            return PathAnalysisResult(
                success=False,
                error_message=str(e)
            )
    
    def find_standard_length_point(self, 
                                 path_coordinates: np.ndarray,
                                 distance_transform: np.ndarray,
                                 start_ratio: float = 0.6,
                                 end_ratio: float = 0.9) -> Optional[Tuple[int, int]]:
        """
        Find the standard length point along the path.
        
        Args:
            path_coordinates: Ordered path coordinates
            distance_transform: Distance transform array
            start_ratio: Start of search range (as ratio of path length)
            end_ratio: End of search range (as ratio of path length)
            
        Returns:
            Standard length point coordinates or None
        """
        try:
            if path_coordinates is None or len(path_coordinates) == 0:
                return None
            
            # Get distance values along the path
            distance_array = distance_transform[
                path_coordinates[:, 0], path_coordinates[:, 1]
            ]
            
            # Find local minima
            average_distance = np.average(distance_array)
            local_minima_indices, _ = find_peaks(
                -distance_array, height=-average_distance, prominence=1
            )
            
            if len(local_minima_indices) == 0:
                self.logger.warning("No standard length point candidates found")
                return None
            
            # Filter by position along path
            path_length = len(path_coordinates)
            lb = round(start_ratio * path_length)
            ub = round(end_ratio * path_length)
            
            feasible_indices = local_minima_indices[
                (local_minima_indices >= lb) & (local_minima_indices <= ub)
            ]
            
            if len(feasible_indices) == 0:
                self.logger.warning("No feasible standard length point found in range")
                return None
            
            # Select point with minimum distance value
            min_distance_idx = np.argmin(distance_array[feasible_indices])
            global_index = feasible_indices[min_distance_idx]
            
            slp = tuple(map(int, path_coordinates[global_index]))
            self.logger.debug(f"Found standard length point at {slp}, index {global_index}")
            
            return slp
            
        except Exception as e:
            self.logger.error(f"Failed to find standard length point: {e}")
            return None
    
    def _recursive_path_construction(self, 
                                   base_branch_index: int,
                                   filament: Any,
                                   path_elements: List[PathElement],
                                   covered_intersection_indices: List[int]) -> Optional[Tuple[int, int]]:
        """Recursively construct the longest path."""
        try:
            # Get intersection points on current branch
            _, intersection_indices = self._get_branch_intersections(
                base_branch_index, filament, also_indices=True
            )
            
            # Find uncovered intersections
            uncovered_indices = [
                idx for idx in intersection_indices 
                if idx not in covered_intersection_indices
            ]
            covered_intersection_indices.extend(uncovered_indices)
            
            if not uncovered_indices:
                # No more intersections - we're at the end
                current_element = path_elements[-1]
                if current_element.end_pts:
                    # Find tail point (not the head point)
                    for end_pt in current_element.end_pts:
                        if end_pt != path_elements[0].head_point:
                            return tuple(map(int, end_pt))
                return None
            
            # Find connected branches
            connected_branches = []
            for idx in uncovered_indices:
                intersec_pts = filament.intersec_pts[idx]
                if not isinstance(intersec_pts, list):
                    intersec_pts = [intersec_pts]
                
                for pt in intersec_pts:
                    branches = self._get_branch_from_point(tuple(map(int, pt)), filament)
                    for branch in branches:
                        if (branch not in connected_branches and 
                            branch not in [elem.branch_index for elem in path_elements]):
                            connected_branches.append(branch)
            
            if not connected_branches:
                self.logger.warning("No connected branches found")
                return None
            
            # Select best aligned branch
            best_branch = self._select_best_aligned_branch(
                base_branch_index, connected_branches, filament
            )
            
            if best_branch is None:
                return None
            
            # Add new branch to path
            intersection_points, intersection_indices = self._get_branch_intersections(
                best_branch, filament, also_indices=True
            )
            end_points, end_indices = self._get_branch_endpoints(
                best_branch, filament, also_indices=True
            )
            
            new_element = PathElement(
                best_branch, intersection_points, intersection_indices,
                end_points, end_indices
            )
            path_elements.append(new_element)
            
            # Continue recursively
            return self._recursive_path_construction(
                best_branch, filament, path_elements, covered_intersection_indices
            )
            
        except Exception as e:
            self.logger.error(f"Recursive path construction failed: {e}")
            return None
    
    def _select_best_aligned_branch(self, 
                                  base_branch_index: int,
                                  candidate_branches: List[int],
                                  filament: Any) -> Optional[int]:
        """Select the branch with best orientation alignment."""
        try:
            if not hasattr(filament, 'orientation_branches'):
                # If no orientation data, just return first candidate
                return candidate_branches[0] if candidate_branches else None
            
            base_orientation = filament.orientation_branches[base_branch_index]
            
            best_branch = None
            min_distance = float('inf')
            
            for branch_idx in candidate_branches:
                branch_orientation = filament.orientation_branches[branch_idx]
                distance = abs(base_orientation - branch_orientation)
                
                if distance < min_distance:
                    min_distance = distance
                    best_branch = branch_idx
            
            self.logger.debug(f"Selected branch {best_branch} with orientation distance {min_distance}")
            return best_branch
            
        except Exception as e:
            self.logger.error(f"Branch selection failed: {e}")
            return candidate_branches[0] if candidate_branches else None
    
    def _construct_ordered_coordinates(self, 
                                     path_elements: List[PathElement],
                                     filament: Any,
                                     head_point: Tuple[int, int],
                                     tail_point: Tuple[int, int]) -> Optional[np.ndarray]:
        """Construct ordered path coordinates from path elements."""
        try:
            all_coordinates = []
            
            if len(path_elements) == 1:
                # Simple case - single element
                element = path_elements[0]
                element.head_point = head_point
                element.tail_point = tail_point
                
                if not self._process_element(element, filament):
                    return None
                
                all_coordinates.extend(element.ordered_branch_points_adjusted)
                
            else:
                # Complex case - multiple elements
                for i, element in enumerate(path_elements):
                    if i == 0:
                        element.head_point = head_point
                    else:
                        # Find shared intersection with previous element
                        prev_element = path_elements[i-1]
                        shared_intersection = self._find_shared_intersection(
                            prev_element, element
                        )
                        
                        if shared_intersection is None:
                            return None
                        
                        prev_element.tail_point = shared_intersection
                        element.head_point = shared_intersection
                        
                        # Process previous element now that both ends are known
                        if not self._process_element(prev_element, filament):
                            return None
                        
                        all_coordinates.extend(prev_element.ordered_branch_points_adjusted)
                    
                    # Handle final element
                    if i == len(path_elements) - 1:
                        element.tail_point = tail_point
                        if not self._process_element(element, filament):
                            return None
                        all_coordinates.extend(element.ordered_branch_points_adjusted)
            
            return np.array(all_coordinates) if all_coordinates else None
            
        except Exception as e:
            self.logger.error(f"Failed to construct ordered coordinates: {e}")
            return None
    
    def _find_shared_intersection(self, 
                                element1: PathElement,
                                element2: PathElement) -> Optional[Tuple[int, int]]:
        """Find shared intersection point between two path elements."""
        try:
            for pt1 in element1.intersection_pts:
                for pt2 in element2.intersection_pts:
                    if is_point_in_neighborhood(pt1, pt2):
                        return tuple(map(int, pt1))
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find shared intersection: {e}")
            return None
    
    def _process_element(self, element: PathElement, filament: Any) -> bool:
        """Process a path element to generate ordered coordinates."""
        try:
            branch_points = filament.branch_pts(True)[element.branch_index]
            branch_length = filament.branch_properties["length"][element.branch_index].value
            
            return element.ProcessElement(branch_points, branch_length)
            
        except Exception as e:
            self.logger.error(f"Failed to process element: {e}")
            return False
    
    def _get_branch_from_point(self, point: Tuple[int, int], filament: Any) -> List[int]:
        """Get branch indices containing a point."""
        try:
            branches = []
            if not hasattr(filament, 'branch_pts'):
                return branches
            
            branch_pts = filament.branch_pts(True)
            for i, branch_points in enumerate(branch_pts):
                if branch_points is not None:
                    for branch_pt in branch_points:
                        if contains_mutual_points(np.array([point]), np.array([branch_pt])):
                            branches.append(i)
                            break
            
            return branches
            
        except Exception as e:
            self.logger.error(f"Failed to get branch from point: {e}")
            return []
    
    def _get_branch_intersections(self, branch_index: int, filament: Any, 
                                also_indices: bool = False) -> Tuple[List, List]:
        """Get intersection points on a branch."""
        try:
            intersec_pts = []
            intersec_indices = []
            
            if not hasattr(filament, 'intersec_pts'):
                return (intersec_pts, intersec_indices) if also_indices else intersec_pts
            
            for i, intersec_pt in enumerate(filament.intersec_pts):
                pts = intersec_pt if isinstance(intersec_pt, list) else [intersec_pt]
                for pt in pts:
                    if self._point_on_branch(tuple(map(int, pt)), branch_index, filament):
                        intersec_pts.append(tuple(map(int, pt)))
                        intersec_indices.append(i)
                        break
            
            return (intersec_pts, intersec_indices) if also_indices else intersec_pts
            
        except Exception as e:
            self.logger.error(f"Failed to get branch intersections: {e}")
            return ([], []) if also_indices else []
    
    def _get_branch_endpoints(self, branch_index: int, filament: Any, 
                            also_indices: bool = False) -> Tuple[List, List]:
        """Get end points on a branch."""
        try:
            end_pts = []
            end_indices = []
            
            if not hasattr(filament, 'end_pts'):
                return (end_pts, end_indices) if also_indices else end_pts
            
            for i, end_pt in enumerate(filament.end_pts):
                pt = end_pt if not isinstance(end_pt, list) else end_pt[0]
                if self._point_on_branch(tuple(map(int, pt)), branch_index, filament):
                    end_pts.append(tuple(map(int, pt)))
                    end_indices.append(i)
            
            return (end_pts, end_indices) if also_indices else end_pts
            
        except Exception as e:
            self.logger.error(f"Failed to get branch endpoints: {e}")
            return ([], []) if also_indices else []
    
    def _point_on_branch(self, point: Tuple[int, int], branch_index: int, filament: Any) -> bool:
        """Check if a point is on a specific branch."""
        try:
            if not hasattr(filament, 'branch_pts'):
                return False
            
            branch_pts = filament.branch_pts(True)
            if branch_index >= len(branch_pts):
                return False
            
            branch_points = branch_pts[branch_index]
            if branch_points is None:
                return False
            
            return any(contains_mutual_points(
                np.array([point]), np.array([branch_pt])
            ) for branch_pt in branch_points)
            
        except Exception as e:
            self.logger.error(f"Failed to check point on branch: {e}")
            return False
