"""
Filament analysis using FilFinder.

This module handles FilFinder operations for skeletal analysis and
filament property extraction with proper error handling.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass

import numpy as np
import astropy.units as u
from fil_finder import FilFinder2D


@dataclass
class FilamentProperties:
    """Properties of a detected filament."""
    length_pixels: float
    branch_count: int
    end_points: List[Tuple[int, int]]
    intersection_points: List[Tuple[int, int]]
    branch_lengths: List[float]
    orientations: List[float]


@dataclass
class FilamentAnalysisResult:
    """Result of filament analysis."""
    success: bool
    filament: Optional[Any] = None
    properties: Optional[FilamentProperties] = None
    error_message: Optional[str] = None


class FilamentAnalyzer:
    """
    Filament analysis using FilFinder2D.
    
    This class provides a clean interface to FilFinder operations
    with proper error handling and type safety.
    """
    
    def __init__(self, skeleton_threshold: float = 1.0):
        self.logger = logging.getLogger(__name__)
        self.skeleton_threshold = skeleton_threshold * u.pix
    
    def analyze_skeleton(self, skeleton: np.ndarray) -> FilamentAnalysisResult:
        """
        Analyze a skeleton image using FilFinder.
        
        Args:
            skeleton: Binary skeleton image
            
        Returns:
            FilamentAnalysisResult with analysis results
        """
        try:
            if skeleton is None or skeleton.size == 0:
                return FilamentAnalysisResult(
                    success=False,
                    error_message="Skeleton is empty or None"
                )
            
            # Initialize FilFinder
            fil_finder = FilFinder2D(skeleton, mask=skeleton)
            fil_finder.create_mask(verbose=False, use_existing_mask=True)
            fil_finder.medskel(verbose=False)
            
            # Analyze skeletons
            try:
                fil_finder.analyze_skeletons(skel_thresh=self.skeleton_threshold)
            except ValueError as e:
                return FilamentAnalysisResult(
                    success=False,
                    error_message=f"FilFinder analysis failed: {e}"
                )
            
            # Select the longest filament
            filament = self._select_longest_filament(fil_finder)
            if filament is None:
                return FilamentAnalysisResult(
                    success=False,
                    error_message="No valid filament found"
                )
            
            # Extract properties
            properties = self._extract_filament_properties(filament)
            
            self.logger.debug(f"Analyzed filament with {properties.branch_count} branches, "
                            f"length: {properties.length_pixels:.2f} pixels")
            
            return FilamentAnalysisResult(
                success=True,
                filament=filament,
                properties=properties
            )
            
        except Exception as e:
            self.logger.error(f"Filament analysis failed: {e}")
            return FilamentAnalysisResult(
                success=False,
                error_message=str(e)
            )
    
    def _select_longest_filament(self, fil_finder: FilFinder2D) -> Optional[Any]:
        """Select the longest filament from FilFinder results."""
        try:
            if not hasattr(fil_finder, 'filaments') or len(fil_finder.filaments) == 0:
                return None
            
            if len(fil_finder.filaments) == 1:
                return fil_finder.filaments[0]
            
            # Find filament with maximum length
            lengths = [q.value for q in fil_finder.lengths()]
            max_index = lengths.index(max(lengths))
            return fil_finder.filaments[max_index]
            
        except Exception as e:
            self.logger.error(f"Failed to select longest filament: {e}")
            return None
    
    def _extract_filament_properties(self, filament: Any) -> FilamentProperties:
        """Extract properties from a filament object."""
        try:
            # Basic properties
            length_pixels = filament.length(u.pix).value
            branch_count = len(filament.branch_properties["length"])
            
            # Convert points to tuples for consistency
            end_points = [tuple(map(int, pt)) if not isinstance(pt, list) else tuple(map(int, pt[0])) 
                         for pt in filament.end_pts]
            
            intersection_points = []
            for pt in filament.intersec_pts:
                if isinstance(pt, list):
                    intersection_points.extend([tuple(map(int, p)) for p in pt])
                else:
                    intersection_points.append(tuple(map(int, pt)))
            
            # Branch properties
            branch_lengths = [length.value for length in filament.branch_properties["length"]]
            
            # Orientations (if available)
            orientations = []
            if hasattr(filament, 'orientation_branches'):
                orientations = list(filament.orientation_branches)
            
            return FilamentProperties(
                length_pixels=length_pixels,
                branch_count=branch_count,
                end_points=end_points,
                intersection_points=intersection_points,
                branch_lengths=branch_lengths,
                orientations=orientations
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract filament properties: {e}")
            # Return minimal properties
            return FilamentProperties(
                length_pixels=0.0,
                branch_count=0,
                end_points=[],
                intersection_points=[],
                branch_lengths=[],
                orientations=[]
            )
    
    def get_branch_points(self, filament: Any, branch_index: int) -> Optional[np.ndarray]:
        """
        Get points for a specific branch.
        
        Args:
            filament: FilFinder filament object
            branch_index: Index of the branch
            
        Returns:
            Array of branch points or None if failed
        """
        try:
            if not hasattr(filament, 'branch_pts'):
                return None
            
            branch_pts = filament.branch_pts(True)
            if branch_index >= len(branch_pts):
                return None
            
            return branch_pts[branch_index]
            
        except Exception as e:
            self.logger.error(f"Failed to get branch points: {e}")
            return None
    
    def find_head_point(self, filament: Any, distance_transform: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find the head point (end point with maximum distance to boundary).
        
        Args:
            filament: FilFinder filament object
            distance_transform: Distance transform array
            
        Returns:
            Head point coordinates or None if failed
        """
        try:
            if not hasattr(filament, 'end_pts') or len(filament.end_pts) == 0:
                return None
            
            head_point = None
            max_distance = -1
            
            for end_pt in filament.end_pts:
                # Handle both single points and lists of points
                if isinstance(end_pt, list):
                    pt = end_pt[0]
                else:
                    pt = end_pt
                
                pt_tuple = tuple(map(int, pt))
                
                # Check bounds
                if (pt_tuple[0] < distance_transform.shape[0] and 
                    pt_tuple[1] < distance_transform.shape[1] and
                    pt_tuple[0] >= 0 and pt_tuple[1] >= 0):
                    
                    distance = distance_transform[pt_tuple]
                    if distance > max_distance:
                        max_distance = distance
                        head_point = pt_tuple
            
            self.logger.debug(f"Found head point at {head_point} with distance {max_distance}")
            return head_point
            
        except Exception as e:
            self.logger.error(f"Failed to find head point: {e}")
            return None
    
    def get_branch_from_point(self, point: Tuple[int, int], filament: Any, 
                             search_radius: int = 2) -> List[int]:
        """
        Find which branch(es) contain a given point.
        
        Args:
            point: Point coordinates (y, x)
            filament: FilFinder filament object
            search_radius: Search radius for point matching
            
        Returns:
            List of branch indices containing the point
        """
        try:
            if not hasattr(filament, 'branch_pts'):
                return []
            
            branch_pts = filament.branch_pts(True)
            matching_branches = []
            
            # Create search kernel around the point
            y, x = point
            search_points = []
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    search_points.append((y + dy, x + dx))
            
            # Check each branch
            for i, branch_points in enumerate(branch_pts):
                if branch_points is None or len(branch_points) == 0:
                    continue
                
                # Check if any search points match branch points
                for branch_pt in branch_points:
                    branch_tuple = tuple(map(int, branch_pt))
                    if branch_tuple in search_points:
                        matching_branches.append(i)
                        break
            
            return matching_branches
            
        except Exception as e:
            self.logger.error(f"Failed to find branch from point: {e}")
            return []
    
    def validate_filament(self, properties: FilamentProperties, 
                         min_length: float = 50.0,
                         min_branches: int = 1) -> bool:
        """
        Validate that a filament meets minimum requirements.
        
        Args:
            properties: Filament properties to validate
            min_length: Minimum length in pixels
            min_branches: Minimum number of branches
            
        Returns:
            True if filament is valid, False otherwise
        """
        if properties.length_pixels < min_length:
            self.logger.warning(f"Filament too short: {properties.length_pixels} < {min_length}")
            return False
        
        if properties.branch_count < min_branches:
            self.logger.warning(f"Too few branches: {properties.branch_count} < {min_branches}")
            return False
        
        if len(properties.end_points) == 0:
            self.logger.warning("No end points found")
            return False
        
        return True
