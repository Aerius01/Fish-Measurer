"""
Modern ArUco marker generator with updated OpenCV API.

This script generates ArUco markers using the modern OpenCV API with
proper error handling and flexibility.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional


def generate_aruco_marker(marker_id: int, 
                         size: int,
                         dictionary_type: int = cv2.aruco.DICT_4X4_50,
                         output_dir: Optional[Path] = None,
                         display: bool = True) -> bool:
    """
    Generate and save an ArUco marker.
    
    Args:
        marker_id: ID of the marker to generate
        size: Size of the marker in pixels
        dictionary_type: ArUco dictionary type
        output_dir: Directory to save the marker (default: current directory)
        display: Whether to display the marker
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use modern API with fallback for older versions
        try:
            dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        except AttributeError:
            # Fallback for older OpenCV versions
            dictionary = cv2.aruco.Dictionary_get(dictionary_type)
        
        # Generate marker
        tag = np.zeros((size, size, 1), dtype="uint8")
        try:
            cv2.aruco.generateImageMarker(dictionary, marker_id, size, tag, 1)
        except AttributeError:
            # Fallback for older API
            cv2.aruco.drawMarker(dictionary, marker_id, size, tag, 1)
        
        # Save marker
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(exist_ok=True)
        
        filename = f"{marker_id}_{size}x{size}_DICT_4x4_50.png"
        filepath = output_dir / filename
        
        success = cv2.imwrite(str(filepath), tag)
        if success:
            print(f"Saved ArUco marker to: {filepath}")
        else:
            print(f"Failed to save marker to: {filepath}")
            return False
        
        # Display if requested
        if display:
            cv2.imshow("ArUco Tag", tag)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"Error generating ArUco marker: {e}")
        return False


def generate_marker_set(marker_ids: list,
                       size: int,
                       dictionary_type: int = cv2.aruco.DICT_4X4_50,
                       output_dir: Optional[Path] = None) -> bool:
    """
    Generate a set of ArUco markers.
    
    Args:
        marker_ids: List of marker IDs to generate
        size: Size of each marker in pixels
        dictionary_type: ArUco dictionary type
        output_dir: Directory to save markers
        
    Returns:
        True if all markers generated successfully, False otherwise
    """
    success_count = 0
    
    for marker_id in marker_ids:
        if generate_aruco_marker(marker_id, size, dictionary_type, output_dir, display=False):
            success_count += 1
    
    print(f"Generated {success_count}/{len(marker_ids)} markers successfully")
    return success_count == len(marker_ids)


if __name__ == "__main__":
    # Default marker generation (maintaining backward compatibility)
    marker_id = 3
    marker_size = 300
    
    success = generate_aruco_marker(marker_id, marker_size)
    
    if success:
        print("ArUco marker generated successfully!")
    else:
        print("Failed to generate ArUco marker.")
    
    # Example: Generate a set of markers for calibration
    # marker_set = [1, 3, 4, 5]
    # generate_marker_set(marker_set, 300, output_dir=Path("markers"))