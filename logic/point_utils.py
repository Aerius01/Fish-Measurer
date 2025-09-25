"""
Point manipulation utilities for fish measurement processing.

This module provides utility functions for working with 2D points in image processing,
specifically for fish skeleton analysis and path optimization.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import cv2


def add_thick_binary_dots(
    array: np.ndarray, 
    *points: Tuple[int, int], 
    size: Tuple[int, int] = (7, 7)
) -> np.ndarray:
    """
    Add thick binary dots at specified points in a binary array.
    
    Args:
        array: Binary array to modify
        *points: Variable number of (y, x) coordinate tuples
        size: Size of the dot kernel (height, width)
        
    Returns:
        Modified array with dots added
        
    Raises:
        ValueError: If size contains non-positive values
    """
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("Size must contain positive values")
        
    for origin in points:
        if len(origin) != 2:
            continue
            
        try:
            kernel_indices = np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size))
            offset = ((np.asarray(size) - 1) / 2).astype(np.uint8)
            kernel_coords = kernel_indices + origin - offset
            
            # Ensure coordinates are within array bounds
            valid_coords = kernel_coords[
                (kernel_coords[:, 0] >= 0) & 
                (kernel_coords[:, 0] < array.shape[0]) &
                (kernel_coords[:, 1] >= 0) & 
                (kernel_coords[:, 1] < array.shape[1])
            ]
            
            if len(valid_coords) > 0:
                array[valid_coords[:, 0], valid_coords[:, 1]] = 1
                
        except (IndexError, ValueError):
            # Skip invalid points
            continue
    
    return array


def is_point_in_neighborhood(
    point1: Tuple[int, int], 
    point2: Tuple[int, int], 
    size: Tuple[int, int] = (5, 5)
) -> bool:
    """
    Check if point1 is within the neighborhood of point2.
    
    Args:
        point1: First point as (y, x) tuple
        point2: Second point as (y, x) tuple  
        size: Neighborhood size as (height, width) tuple
        
    Returns:
        True if point1 is in the neighborhood of point2
        
    Raises:
        ValueError: If size contains non-positive values
    """
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("Size must contain positive values")
        
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        offset = ((np.asarray(size) - 1) / 2).astype(np.int32)
        neighboring_indices = np.argwhere(kernel) + np.array(point2) - offset
        
        return np.any(np.equal(neighboring_indices, point1).all(axis=1))
        
    except (ValueError, TypeError):
        return False


def contains_mutual_points(
    array1: Union[np.ndarray, List[List[int]]], 
    array2: Union[np.ndarray, List[List[int]]]
) -> np.ndarray:
    """
    Determine which points in array2 also exist in array1.
    
    Args:
        array1: Sub-array of (y, x) points to check against
        array2: Principal array of (y, x) points to check
        
    Returns:
        Boolean array indicating which rows in array2 exist in array1
        
    Raises:
        ValueError: If arrays have incompatible shapes
    """
    # Convert to numpy arrays
    arr1 = np.asarray(array1, dtype=np.int32)
    arr2 = np.asarray(array2, dtype=np.int32)
    
    # Handle single point case
    if arr1.ndim == 1:
        if len(arr1) != 2:
            raise ValueError("Single point must have exactly 2 coordinates")
        arr1 = arr1.reshape(1, 2)
    
    if arr2.ndim == 1:
        if len(arr2) != 2:
            raise ValueError("Single point must have exactly 2 coordinates")
        arr2 = arr2.reshape(1, 2)
    
    # Validate shapes
    if arr1.shape[1] != 2 or arr2.shape[1] != 2:
        raise ValueError("Arrays must contain 2D points (shape: (n, 2))")
    
    if len(arr1) == 0:
        return np.zeros(len(arr2), dtype=bool)
    
    if len(arr2) == 0:
        return np.array([], dtype=bool)
    
    # Efficient vectorized comparison
    return (arr1[:, None] == arr2).all(axis=2).any(axis=0)


def calculate_euclidean_distance(
    point1: Tuple[Union[int, float], Union[int, float]], 
    point2: Tuple[Union[int, float], Union[int, float]]
) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        point1: First point as (x, y) or (y, x) tuple
        point2: Second point as (x, y) or (y, x) tuple
        
    Returns:
        Euclidean distance between the points
        
    Raises:
        ValueError: If points don't have exactly 2 coordinates
    """
    if len(point1) != 2 or len(point2) != 2:
        raise ValueError("Points must have exactly 2 coordinates")
    
    try:
        dx = float(point1[0] - point2[0])
        dy = float(point1[1] - point2[1])
        return (dx * dx + dy * dy) ** 0.5
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid point coordinates: {e}")


def optimize_path(
    coordinates: List[List[Union[int, float]]], 
    start_point: Optional[List[Union[int, float]]] = None
) -> List[List[Union[int, float]]]:
    """
    Optimize a path by ordering points using nearest neighbor algorithm.
    
    This function finds the optimal ordering of points to minimize total path length
    using a greedy nearest neighbor approach.
    
    Args:
        coordinates: List of [x, y] coordinate pairs
        start_point: Optional starting point. If None, uses first coordinate
        
    Returns:
        Optimized path as list of [x, y] coordinate pairs
        
    Raises:
        ValueError: If coordinates list is empty or contains invalid points
    """
    if not coordinates:
        raise ValueError("Coordinates list cannot be empty")
    
    # Validate all coordinates
    for i, coord in enumerate(coordinates):
        if len(coord) != 2:
            raise ValueError(f"Coordinate at index {i} must have exactly 2 values")
        try:
            float(coord[0])
            float(coord[1])
        except (TypeError, ValueError):
            raise ValueError(f"Coordinate at index {i} contains non-numeric values")
    
    # Create working copy
    coords_copy = [list(coord) for coord in coordinates]
    
    # Set starting point
    if start_point is None:
        current = coords_copy[0]
    else:
        if len(start_point) != 2:
            raise ValueError("Start point must have exactly 2 coordinates")
        current = list(start_point)
    
    # Initialize path
    path = [current]
    if current in coords_copy:
        coords_copy.remove(current)
    
    # Build path using nearest neighbor
    while coords_copy:
        try:
            nearest = min(coords_copy, key=lambda x: calculate_euclidean_distance(current, x))
            path.append(nearest)
            coords_copy.remove(nearest)
            current = nearest
        except ValueError:
            # If distance calculation fails, just take the first remaining point
            nearest = coords_copy[0]
            path.append(nearest)
            coords_copy.remove(nearest)
            current = nearest
    
    return path


def calculate_path_length(coordinates: List[List[Union[int, float]]]) -> float:
    """
    Calculate total length of a path defined by sequential points.
    
    Args:
        coordinates: List of [x, y] coordinate pairs defining the path
        
    Returns:
        Total path length
        
    Raises:
        ValueError: If coordinates list has fewer than 2 points
    """
    if len(coordinates) < 2:
        raise ValueError("Path must contain at least 2 points")
    
    total_length = 0.0
    
    for i in range(len(coordinates) - 1):
        try:
            current_point = coordinates[i]
            next_point = coordinates[i + 1]
            
            if len(current_point) != 2 or len(next_point) != 2:
                continue
                
            segment_length = calculate_euclidean_distance(current_point, next_point)
            total_length += segment_length
            
        except (ValueError, TypeError):
            # Skip invalid segments
            continue
    
    return total_length