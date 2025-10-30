"""
Path utility functions for fish measurement.

This module provides utility functions for path optimization, point matching,
and neighborhood calculations, mirrored from the original PointUtils.py implementation.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2


def distance(p1: Tuple, p2: Tuple) -> float:
    """
    Compute Euclidean distance between two points.

    Args:
        p1: First point (y, x) or (x, y)
        p2: Second point (y, x) or (x, y)

    Returns:
        Euclidean distance between points
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


def optimize_path(coords: List, start: Optional[Tuple] = None) -> List:
    """
    Order coordinates using greedy nearest-neighbor algorithm.

    This ensures points are ordered in a continuous path starting from
    the specified start point, which is critical for accurate path analysis.

    Args:
        coords: List of coordinate tuples [(y1, x1), (y2, x2), ...]
        start: Starting point for path optimization. If None, uses first coordinate.

    Returns:
        Optimally ordered list of coordinates

    Example:
        >>> coords = [(0, 0), (2, 2), (1, 1)]
        >>> optimize_path(coords, start=(0, 0))
        [(0, 0), (1, 1), (2, 2)]
    """
    if start is None:
        start = coords[0]

    path = [start]
    remaining = [c for c in coords if c != start]

    while remaining:
        last_point = path[-1]
        nearest = min(remaining, key=lambda x: distance(last_point, x))
        path.append(nearest)
        remaining.remove(nearest)

    return path


def point_in_neighborhood(point1: Tuple[int, int],
                         point2: Tuple[int, int],
                         size: Tuple[int, int] = (5, 5)) -> bool:
    """
    Check if point1 is in the neighborhood of point2.

    Uses an elliptical kernel to define the neighborhood region.

    Args:
        point1: Point to check (y, x)
        point2: Reference point (y, x)
        size: Size of neighborhood kernel (height, width)

    Returns:
        True if point1 is within neighborhood of point2, False otherwise
    """
    try:
        # Create elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

        # Calculate neighborhood indices centered on point2
        neighboring_indices = np.argwhere(kernel) + np.array(point2) - ((np.array(size) - 1) / 2)

        # Check if point1 is in the neighborhood
        point1_array = np.array(point1)
        matches = np.equal(neighboring_indices, point1_array).all(axis=1)

        return bool(np.any(matches))

    except Exception:
        # Fallback to simple distance check
        return distance(point1, point2) <= 2.0


def contains_mutual_points(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Determine whether any point in array1 exists within array2.

    Args:
        array1: Sub-array of (m, 2) points as (y, x) coordinates
        array2: Principal array of (n, 2) points as (y, x) coordinates

    Returns:
        Boolean array of shape (n,) indicating which rows in array2
        also exist in array1

    Example:
        >>> array1 = np.array([[0, 0], [1, 1]])
        >>> array2 = np.array([[0, 0], [2, 2], [1, 1]])
        >>> contains_mutual_points(array1, array2)
        array([ True, False,  True])
    """
    # Ensure inputs are numpy arrays
    if not isinstance(array1, np.ndarray):
        array1 = np.asarray(array1)

    if not isinstance(array2, np.ndarray):
        array2 = np.asarray(array2)

    # Handle 1D case (single point)
    if len(np.shape(array1)) == 1:
        array1 = np.reshape(array1, (1, 2))

    # Broadcast comparison: array1 (m, 1, 2) vs array2 (1, n, 2) -> (m, n, 2)
    # Then check if all coordinates match, and if any array1 point matches
    bool_overlap_array = (array1[:, None] == array2).all(axis=2).any(axis=0)

    return np.asarray(bool_overlap_array)


def create_neighborhood_kernel(point: Tuple[int, int],
                               size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Create a neighborhood kernel of points around a center point.

    Args:
        point: Center point (y, x)
        size: Kernel size (height, width)

    Returns:
        Array of points within the neighborhood kernel
    """
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        offset = ((np.asarray(size) - 1) / 2).astype('uint8')
        neighborhood = np.argwhere(kernel) + np.array(point) - offset
        return neighborhood

    except Exception:
        # Fallback to simple square neighborhood
        y, x = point
        radius = max(size) // 2
        points = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                points.append((y + dy, x + dx))
        return np.array(points)


def calculate_path_length(coordinates: np.ndarray) -> float:
    """
    Calculate total path length from ordered coordinates.

    Args:
        coordinates: Array of shape (n, 2) with ordered path coordinates

    Returns:
        Total path length in pixels
    """
    if len(coordinates) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(coordinates) - 1):
        total_length += distance(coordinates[i], coordinates[i + 1])

    return total_length
