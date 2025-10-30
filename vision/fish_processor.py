"""
Main fish processing coordinator.

This module coordinates the entire fish measurement process by orchestrating
the various specialized modules (image processing, filament analysis, path analysis).
"""

from __future__ import annotations

import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_processor import ImageProcessor, ProcessingResult
from .filament_analyzer import FilamentAnalyzer, FilamentAnalysisResult
from .path_analyzer import PathAnalyzer, PathAnalysisResult
from .display_manager import DisplayManager


@dataclass
class FishMeasurementResult:
    """Complete result of fish measurement analysis."""
    success: bool
    fish_id: str
    total_length_pixels: float
    standard_length_pixels: Optional[float] = None
    head_point: Optional[Tuple[int, int]] = None
    tail_point: Optional[Tuple[int, int]] = None
    standard_length_point: Optional[Tuple[int, int]] = None
    processing_log: List[str] = None
    visualization_data: Optional[Dict[str, np.ndarray]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.processing_log is None:
            self.processing_log = []


class FishProcessor:
    """
    Main coordinator for fish measurement processing.
    
    This class orchestrates the entire measurement pipeline from raw images
    to final measurements, coordinating multiple specialized processors.
    """
    
    def __init__(self, output_folder: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.output_folder = Path(output_folder) if output_folder else Path.cwd()
        
        # Initialize processors
        self.image_processor = ImageProcessor()
        self.filament_analyzer = FilamentAnalyzer()
        self.path_analyzer = PathAnalyzer()
        self.display_manager = DisplayManager()
        
        # Processing parameters
        self.length_tolerance = 0.3  # 30% tolerance for FilFinder comparison
        self.min_path_length = 50.0  # Minimum path length in pixels
    
    def process_fish(self, 
                    fish_id: str,
                    raw_frame: np.ndarray,
                    binary_mask: np.ndarray) -> FishMeasurementResult:
        """
        Process a fish measurement from raw and binarized frames.
        
        Args:
            fish_id: Unique identifier for this fish
            raw_frame: Original raw camera frame
            binary_mask: Background-subtracted binary mask (0/255)
            
        Returns:
            FishMeasurementResult with complete analysis results
        """
        processing_log = []
        processing_log.append(f"Starting analysis for fish ID: {fish_id}")
        
        try:
            # Step 1: Basic image processing
            processing_log.append("Step 1: Image processing")
            image_result = self.image_processor.process_binary_image(binary_mask)
            
            if not image_result.success:
                return FishMeasurementResult(
                    success=False,
                    fish_id=fish_id,
                    total_length_pixels=0.0,
                    processing_log=processing_log,
                    error_message=f"Image processing failed: {image_result.error_message}"
                )
            
            processing_log.append(f"Image processing successful - dimensions: {image_result.dimensions}")
            
            # Step 2: Filament analysis
            processing_log.append("Step 2: Filament analysis")
            filament_result = self.filament_analyzer.analyze_skeleton(image_result.skeleton)
            
            if not filament_result.success:
                return FishMeasurementResult(
                    success=False,
                    fish_id=fish_id,
                    total_length_pixels=0.0,
                    processing_log=processing_log,
                    error_message=f"Filament analysis failed: {filament_result.error_message}"
                )
            
            processing_log.append(f"Found filament with {filament_result.properties.branch_count} branches, "
                                f"length: {filament_result.properties.length_pixels:.2f} pixels")
            
            # Step 3: Find head point
            processing_log.append("Step 3: Finding head point")
            head_point = self.filament_analyzer.find_head_point(
                filament_result.filament, image_result.distance_transform
            )
            
            if head_point is None:
                return FishMeasurementResult(
                    success=False,
                    fish_id=fish_id,
                    total_length_pixels=0.0,
                    processing_log=processing_log,
                    error_message="Could not find head point"
                )
            
            processing_log.append(f"Head point found at: {head_point}")
            
            # Step 4: Path analysis
            processing_log.append("Step 4: Longest path construction")
            path_result = self.path_analyzer.construct_longest_path(
                filament_result.filament, head_point, image_result.distance_transform
            )
            
            if not path_result.success:
                return FishMeasurementResult(
                    success=False,
                    fish_id=fish_id,
                    total_length_pixels=0.0,
                    processing_log=processing_log,
                    error_message=f"Path analysis failed: {path_result.error_message}"
                )
            
            processing_log.append(f"Constructed path with {len(path_result.path_coordinates)} points")
            
            # Step 5: Validate path length against FilFinder
            processing_log.append("Step 5: Validating path length")
            filfinder_length = filament_result.properties.length_pixels
            path_length = path_result.total_length
            
            if not self._validate_path_length(path_length, filfinder_length):
                return FishMeasurementResult(
                    success=False,
                    fish_id=fish_id,
                    total_length_pixels=path_length,
                    processing_log=processing_log,
                    error_message=f"Path length {path_length:.2f} outside expected range "
                                f"[{filfinder_length * (1-self.length_tolerance):.2f}, "
                                f"{filfinder_length * (1+self.length_tolerance):.2f}]"
                )
            
            processing_log.append(f"Path length {path_length:.2f} validated against "
                                f"FilFinder length {filfinder_length:.2f}")
            
            # Step 6: Find standard length point
            processing_log.append("Step 6: Finding standard length point")
            standard_length_point = self.path_analyzer.find_standard_length_point(
                path_result.path_coordinates, image_result.distance_transform
            )

            # Calculate standard length and check if SLP trimming is needed
            standard_length_pixels = None
            if standard_length_point is not None:
                # Check if SLP is distinct from tail (>40px away)
                slp_dist = np.linalg.norm(
                    np.array(standard_length_point) - np.array(path_result.tail_point)
                )
                slp_near_endpoint = slp_dist < 40.0

                processing_log.append(f"Standard length point found at {standard_length_point}, "
                                    f"distance from tail: {slp_dist:.2f} pixels")

                # Step 6a: Trim path at SLP if it's distinct (removes tail flaring)
                if not slp_near_endpoint:
                    processing_log.append("Step 6a: Trimming path at SLP (removing tail region)")
                    path_result = self._trim_path_at_slp(
                        path_result, standard_length_point, filament_result.filament
                    )
                    processing_log.append(f"Path trimmed at SLP: new length {path_result.total_length:.2f} pixels")
                else:
                    processing_log.append("SLP near tail endpoint, no trimming needed")

                # Calculate standard length from head to SLP
                slp_distances = [
                    np.linalg.norm(np.array(coord) - np.array(standard_length_point))
                    for coord in path_result.path_coordinates
                ]
                slp_index = np.argmin(slp_distances)

                standard_length_pixels = sum(
                    np.linalg.norm(np.array(path_result.path_coordinates[i+1]) -
                                 np.array(path_result.path_coordinates[i]))
                    for i in range(slp_index)
                )

                processing_log.append(f"Standard length: {standard_length_pixels:.2f} pixels")
            else:
                processing_log.append("No standard length point found")
            
            # Step 7: Add contour distances (head and tail extensions)
            processing_log.append("Step 7: Adding contour distances")
            extended_length = self._add_contour_distances(
                path_result, image_result, standard_length_point
            )
            
            if extended_length is None:
                processing_log.append("Warning: Could not add contour distances")
                extended_length = path_length
            else:
                processing_log.append(f"Added contour distances, total length: {extended_length:.2f}")
            
            # Step 8: Create visualization data
            processing_log.append("Step 8: Creating visualizations")
            visualization_data = self._create_visualizations(
                raw_frame, image_result, path_result, standard_length_point
            )
            
            # Success!
            processing_log.append("Analysis completed successfully")
            
            return FishMeasurementResult(
                success=True,
                fish_id=fish_id,
                total_length_pixels=extended_length,
                standard_length_pixels=standard_length_pixels,
                head_point=path_result.head_point,
                tail_point=path_result.tail_point,
                standard_length_point=standard_length_point,
                processing_log=processing_log,
                visualization_data=visualization_data
            )
            
        except Exception as e:
            self.logger.error(f"Fish processing failed: {e}")
            processing_log.append(f"ERROR: {e}")
            
            return FishMeasurementResult(
                success=False,
                fish_id=fish_id,
                total_length_pixels=0.0,
                processing_log=processing_log,
                error_message=str(e)
            )
    
    def process_multiple_frames(self,
                              fish_id: str,
                              raw_frames: List[np.ndarray],
                              binarized_frames: List[np.ndarray]) -> List[FishMeasurementResult]:
        """
        Process multiple frames of the same fish.
        
        Args:
            fish_id: Fish identifier
            raw_frames: List of raw frames
            binarized_frames: List of binarized frames
            
        Returns:
            List of measurement results
        """
        results = []
        
        for i, (raw_frame, binarized_frame) in enumerate(zip(raw_frames, binarized_frames)):
            frame_id = f"{fish_id}_frame_{i+1}"
            result = self.process_fish(frame_id, raw_frame, binarized_frame)
            results.append(result)
        
        return results
    
    def get_average_measurement(self, results: List[FishMeasurementResult]) -> Optional[Dict[str, float]]:
        """
        Calculate average measurements from multiple results.
        
        Args:
            results: List of measurement results
            
        Returns:
            Dictionary with average measurements or None if no valid results
        """
        valid_results = [r for r in results if r.success]
        
        if not valid_results:
            return None
        
        total_lengths = [r.total_length_pixels for r in valid_results]
        standard_lengths = [r.standard_length_pixels for r in valid_results 
                          if r.standard_length_pixels is not None]
        
        average_data = {
            'total_length_mean': np.mean(total_lengths),
            'total_length_std': np.std(total_lengths),
            'measurement_count': len(valid_results)
        }
        
        if standard_lengths:
            average_data.update({
                'standard_length_mean': np.mean(standard_lengths),
                'standard_length_std': np.std(standard_lengths),
                'standard_length_count': len(standard_lengths)
            })
        
        return average_data
    
    def _validate_path_length(self, path_length: float, filfinder_length: float) -> bool:
        """Validate path length against FilFinder reference."""
        if path_length < self.min_path_length:
            return False

        lower_bound = filfinder_length * (1 - self.length_tolerance)
        upper_bound = filfinder_length * (1 + self.length_tolerance)

        return lower_bound <= path_length <= upper_bound

    def _trim_path_at_slp(self,
                         path_result: PathAnalysisResult,
                         standard_length_point: Tuple[int, int],
                         filament: Any) -> PathAnalysisResult:
        """
        Trim path from SLP to tail when SLP is distinct (>40px from tail).

        This removes tail flaring from the measurement as per the original implementation.
        When an SLP is found that is meaningfully far from the tail, we:
        1. Find which path element contains the SLP
        2. Remove all elements after that element
        3. Trim points in the SLP element after the SLP
        4. Update tail point to be the SLP

        Args:
            path_result: Current path analysis result
            standard_length_point: The SLP coordinates
            filament: FilFinder filament object

        Returns:
            Updated PathAnalysisResult with trimmed path
        """
        try:
            from .path_utils import contains_mutual_points, calculate_path_length

            # Find which path element contains the SLP
            slp_branch_indices = self.filament_analyzer.get_branch_from_point(
                standard_length_point, filament
            )

            if not slp_branch_indices:
                self.logger.warning("Could not identify SLP branch, skipping trim")
                return path_result

            # Find the element in our path_elements list
            slp_element_index = None
            for idx, element in enumerate(path_result.path_elements):
                if element.branch_index == slp_branch_indices[0]:
                    slp_element_index = idx
                    break

            if slp_element_index is None:
                self.logger.warning("SLP branch not in path elements, skipping trim")
                return path_result

            self.logger.debug(f"SLP found on path element {slp_element_index}, "
                            f"branch index {slp_branch_indices[0]}")

            # Keep only elements up to and including the SLP element
            trimmed_elements = path_result.path_elements[:slp_element_index+1]

            # Trim points in the SLP element after the SLP
            slp_element = trimmed_elements[slp_element_index]
            ordered_pts = np.array(slp_element.ordered_branch_points_adjusted)

            # Find SLP index in ordered points
            bool_array = contains_mutual_points(
                np.array([standard_length_point]), ordered_pts
            )
            slp_indices = np.where(bool_array)[0]

            if len(slp_indices) > 0:
                slp_idx = slp_indices[0]
                points_to_remove = ordered_pts[slp_idx+1:]

                if len(points_to_remove) > 0:
                    # Update the element
                    length_to_remove = len(points_to_remove) * slp_element.unit_length
                    slp_element.ordered_branch_points_adjusted = [
                        tuple(map(int, pt)) for pt in ordered_pts[:slp_idx+1]
                    ]
                    slp_element.total_adjusted_length -= length_to_remove
                    slp_element.tail_point = standard_length_point

                    self.logger.debug(f"Trimmed {len(points_to_remove)} points, "
                                    f"removed {length_to_remove:.2f} pixels")

            # Reconstruct path coordinates from trimmed elements
            new_coords = []
            for element in trimmed_elements:
                new_coords.extend(element.ordered_branch_points_adjusted)

            new_coords_array = np.array(new_coords)
            new_length = calculate_path_length(new_coords_array)

            self.logger.info(f"Trimmed path at SLP: {len(new_coords)} points, "
                           f"length: {new_length:.2f} pixels")

            # Return updated result with SLP as new tail
            return PathAnalysisResult(
                success=True,
                path_coordinates=new_coords_array,
                path_elements=trimmed_elements,
                head_point=path_result.head_point,
                tail_point=standard_length_point,  # SLP becomes new tail
                total_length=new_length
            )

        except Exception as e:
            self.logger.error(f"Failed to trim path at SLP: {e}", exc_info=True)
            return path_result  # Return original on error
    
    def _add_contour_distances(self,
                             path_result: PathAnalysisResult,
                             image_result: ProcessingResult,
                             standard_length_point: Optional[Tuple[int, int]]) -> Optional[float]:
        """
        Add distances from endpoints to contour boundaries using pullback points.

        This implements the pullback-circle-line intersection method from the original code.
        """
        try:
            dimensions = image_result.dimensions
            total_added_distance = 0.0

            # Get pullback points
            pullback_pts = []

            # Head pullback: Use 20-pixel circle intersection
            head_pullback = self._circle_mask_intersection(
                path_result.head_point, path_result.path_coordinates, dimensions
            )
            if head_pullback is None:
                self.logger.warning("Head pullback does not intersect with long path")
                return None
            pullback_pts.append(head_pullback)

            # Tail pullback: Use SLP if distinct (>40px from tail), otherwise use circle
            slp_near_endpoint = False
            if standard_length_point is not None:
                # Check if SLP is meaningfully far from tail
                slp_dist = np.linalg.norm(
                    np.array(standard_length_point) - np.array(path_result.tail_point)
                )
                slp_near_endpoint = slp_dist < 40.0
            else:
                slp_near_endpoint = True

            if not slp_near_endpoint:
                pullback_pts.append(standard_length_point)
                self.logger.debug(f"Using SLP for tail pullback: {standard_length_point}")
            else:
                tail_pullback = self._circle_mask_intersection(
                    path_result.tail_point, path_result.path_coordinates, dimensions
                )
                if tail_pullback is None:
                    self.logger.warning("Tail pullback does not intersect with long path")
                    return None
                pullback_pts.append(tail_pullback)
                self.logger.debug(f"Using circle for tail pullback: {tail_pullback}")

            # Process head and tail extensions
            for i in range(2):
                pullback_pt = pullback_pts[i]
                end_pt = path_result.head_point if i == 0 else path_result.tail_point

                # Draw line from pullback through endpoint to contour
                line_mask = self._draw_line_mask(pullback_pt, end_pt, dimensions)

                # Find contour intersection
                dilated_contour = cv2.dilate(
                    image_result.contour,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=1
                )
                combined = np.add(line_mask, dilated_contour / 255.0)
                boundary_pts = list(zip(*np.where(combined > 1.5)))

                if not boundary_pts:
                    self.logger.warning(f"No boundary intersection found for {'head' if i == 0 else 'tail'}")
                    continue

                # Find closest boundary point to endpoint
                closest_pt = min(
                    boundary_pts,
                    key=lambda pt: np.linalg.norm(np.array(pt) - np.array(end_pt))
                )

                # Calculate distance
                if i == 0:
                    # Head: distance from endpoint to contour
                    distance = np.linalg.norm(np.array(end_pt) - np.array(closest_pt))
                    self.logger.debug(f"Head extension: {distance:.2f} pixels")
                else:
                    # Tail: depends on SLP
                    if not slp_near_endpoint:
                        # Distance from SLP to contour
                        distance = np.linalg.norm(np.array(pullback_pt) - np.array(closest_pt))
                        self.logger.debug(f"Tail extension (from SLP): {distance:.2f} pixels")
                    else:
                        # Distance from endpoint to contour
                        distance = np.linalg.norm(np.array(end_pt) - np.array(closest_pt))
                        self.logger.debug(f"Tail extension (from endpoint): {distance:.2f} pixels")

                total_added_distance += distance

            return path_result.total_length + total_added_distance

        except Exception as e:
            self.logger.error(f"Failed to add contour distances: {e}", exc_info=True)
            return None

    def _circle_mask_intersection(
        self,
        point: Tuple[int, int],
        path_coords: np.ndarray,
        dimensions: Tuple[int, int],
        radius: int = 20
    ) -> Optional[Tuple[int, int]]:
        """
        Find intersection between a circle around a point and the path coordinates.

        Args:
            point: Center point (y, x)
            path_coords: Path coordinates array
            dimensions: Image dimensions (height, width)
            radius: Circle radius in pixels

        Returns:
            Intersection point or None
        """
        try:
            # Vectorized approach: create grid of points in circle bounding box
            y_min = max(0, point[0] - radius)
            y_max = min(dimensions[0], point[0] + radius + 1)
            x_min = max(0, point[1] - radius)
            x_max = min(dimensions[1], point[1] + radius + 1)

            # Create coordinate grids
            yy, xx = np.meshgrid(
                np.arange(y_min, y_max),
                np.arange(x_min, x_max),
                indexing='ij'
            )

            # Calculate distances from center
            distances = np.sqrt((xx - point[1])**2 + (yy - point[0])**2)

            # Find points on circle perimeter (radius ± 0.7)
            circle_mask = (distances >= radius - 0.7) & (distances <= radius + 0.7)
            circle_pts = np.column_stack(np.where(circle_mask))
            circle_pts[:, 0] += y_min  # Adjust for offset
            circle_pts[:, 1] += x_min

            if len(circle_pts) == 0:
                return None

            # Find intersection with path using vectorized distance calculation
            # Compute all pairwise distances at once (more efficient than loop)
            if len(path_coords) > 0 and len(circle_pts) > 0:
                # Broadcast: path_coords (n, 2) -> (n, 1, 2), circle_pts (m, 2) -> (1, m, 2)
                # Result: (n, m) distances
                dists = np.linalg.norm(
                    path_coords[:, np.newaxis, :] - circle_pts[np.newaxis, :, :],
                    axis=2
                )
                # Find any path point within threshold
                min_dists = np.min(dists, axis=1)
                matches = np.where(min_dists < 2.0)[0]
                if len(matches) > 0:
                    return tuple(map(int, path_coords[matches[0]]))

            return None

        except Exception as e:
            self.logger.error(f"Circle mask intersection failed: {e}")
            return None

    def _draw_line_mask(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        dimensions: Tuple[int, int]
    ) -> np.ndarray:
        """
        Draw a line mask from pt1 through pt2 extending to image boundaries.

        Args:
            pt1: First point (y, x)
            pt2: Second point (y, x)
            dimensions: Image dimensions

        Returns:
            Binary line mask
        """
        line_mask = np.zeros(dimensions, dtype=np.uint8)

        try:
            # Calculate line direction and extend to image boundaries
            dy = pt2[0] - pt1[0]
            dx = pt2[1] - pt1[1]

            # Extend line far beyond image boundaries
            # Use a large multiplier to ensure line reaches edges
            scale = max(dimensions) * 2

            # Calculate extended endpoints
            if dx == 0:  # Vertical line
                start_pt = (0, pt1[1])
                end_pt = (dimensions[0] - 1, pt1[1])
            elif dy == 0:  # Horizontal line
                start_pt = (pt1[0], 0)
                end_pt = (pt1[0], dimensions[1] - 1)
            else:
                # Extend in both directions from pt1
                direction = np.array([dy, dx]) / np.linalg.norm([dy, dx])
                start_pt = tuple(map(int, np.array(pt1) - direction * scale))
                end_pt = tuple(map(int, np.array(pt1) + direction * scale))

            # Use OpenCV's line drawing (much faster than manual)
            # Note: cv2.line uses (x, y) convention, so we swap
            cv2.line(
                line_mask,
                (start_pt[1], start_pt[0]),  # (x, y)
                (end_pt[1], end_pt[0]),      # (x, y)
                1,
                thickness=1,
                lineType=cv2.LINE_AA  # Anti-aliased for better coverage
            )

            return line_mask

        except Exception as e:
            self.logger.error(f"Line mask drawing failed: {e}")
            return line_mask
    
    def _create_visualizations(self,
                             raw_frame: np.ndarray,
                             image_result: ProcessingResult,
                             path_result: PathAnalysisResult,
                             standard_length_point: Optional[Tuple[int, int]]) -> Dict[str, np.ndarray]:
        """Create visualization images for the analysis."""
        try:
            visualizations = {}
            
            # Create path binary image
            long_path_binary = np.zeros(image_result.dimensions)
            if path_result.path_coordinates is not None:
                for coord in path_result.path_coordinates:
                    y, x = int(coord[0]), int(coord[1])
                    if 0 <= y < image_result.dimensions[0] and 0 <= x < image_result.dimensions[1]:
                        long_path_binary[y, x] = 1
            
            # Add endpoint markers
            if path_result.head_point:
                y, x = path_result.head_point
                long_path_binary[max(0, y-2):min(image_result.dimensions[0], y+3),
                               max(0, x-2):min(image_result.dimensions[1], x+3)] = 1
            
            if path_result.tail_point:
                y, x = path_result.tail_point
                long_path_binary[max(0, y-2):min(image_result.dimensions[0], y+3),
                               max(0, x-2):min(image_result.dimensions[1], x+3)] = 1
            
            # Add standard length point marker
            if standard_length_point:
                y, x = standard_length_point
                long_path_binary[max(0, y-2):min(image_result.dimensions[0], y+3),
                               max(0, x-2):min(image_result.dimensions[1], x+3)] = 1
            
            visualizations['long_path_binary'] = long_path_binary
            visualizations['skeleton'] = image_result.skeleton
            visualizations['contour'] = image_result.contour
            visualizations['distance_transform'] = image_result.distance_transform
            
            # Create combined visualization
            visualizations['processed_frame'] = self.image_processor.create_visualization_overlay(
                raw_frame, image_result.skeleton, image_result.contour, long_path_binary
            )
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {e}")
            return {}
    
    def save_results(self, result: FishMeasurementResult, 
                    save_images: bool = True) -> bool:
        """
        Save measurement results to files.
        
        Args:
            result: Measurement result to save
            save_images: Whether to save visualization images
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory
            fish_output_dir = self.output_folder / result.fish_id
            fish_output_dir.mkdir(exist_ok=True)
            
            # Save log
            log_file = fish_output_dir / "processing_log.txt"
            with open(log_file, 'w') as f:
                f.write(f"Fish ID: {result.fish_id}\n")
                f.write(f"Success: {result.success}\n")
                f.write(f"Total Length: {result.total_length_pixels:.2f} pixels\n")
                if result.standard_length_pixels:
                    f.write(f"Standard Length: {result.standard_length_pixels:.2f} pixels\n")
                f.write(f"Head Point: {result.head_point}\n")
                f.write(f"Tail Point: {result.tail_point}\n")
                f.write(f"Standard Length Point: {result.standard_length_point}\n")
                f.write("\nProcessing Log:\n")
                for log_entry in result.processing_log:
                    f.write(f"{log_entry}\n")
                if result.error_message:
                    f.write(f"\nError: {result.error_message}\n")
            
            # Save images if requested
            if save_images and result.visualization_data:
                for name, image in result.visualization_data.items():
                    if image is not None and image.size > 0:
                        image_file = fish_output_dir / f"{name}.png"
                        # Convert to uint8 if needed
                        if image.dtype != np.uint8:
                            if image.max() <= 1.0:
                                image = (image * 255).astype(np.uint8)
                            else:
                                image = image.astype(np.uint8)
                        cv2.imwrite(str(image_file), image)
            
            self.logger.info(f"Saved results for {result.fish_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return False
