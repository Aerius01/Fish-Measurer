"""
Example usage of the refactored vision module.

This script demonstrates how to use the new modular vision API
for fish measurement processing.
"""

import numpy as np
from pathlib import Path
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Import the new vision module components
from vision import (
    FishProcessor, 
    CameraManager, 
    CameraConfig,
    ArUcoDetector,
    DisplayManager,
    generate_aruco_marker,
    get_version_info
)


def example_basic_usage():
    """Demonstrate basic fish processing workflow."""
    print("=== Basic Fish Processing Example ===")
    
    # Print version info
    version_info = get_version_info()
    print(f"Vision Module Version: {version_info['version']}")
    
    # Initialize the main processor
    output_folder = Path("example_output")
    processor = FishProcessor(output_folder=output_folder)
    
    # Create some dummy data for demonstration
    # In real usage, these would come from the camera
    raw_frame = np.random.randint(0, 255, (1000, 1200, 3), dtype=np.uint8)
    binary_frame = np.random.randint(0, 2, (1000, 1200), dtype=np.uint8) * 255
    
    # Process a fish measurement
    result = processor.process_fish("example_fish_001", raw_frame, binary_frame)
    
    # Check results
    if result.success:
        print(f"✓ Processing successful!")
        print(f"  Total length: {result.total_length_pixels:.2f} pixels")
        if result.standard_length_pixels:
            print(f"  Standard length: {result.standard_length_pixels:.2f} pixels")
        print(f"  Head point: {result.head_point}")
        print(f"  Tail point: {result.tail_point}")
        
        # Save results
        if processor.save_results(result):
            print(f"  Results saved to: {output_folder / result.fish_id}")
    else:
        print(f"✗ Processing failed: {result.error_message}")
        print("Processing log:")
        for log_entry in result.processing_log:
            print(f"  {log_entry}")


def example_camera_usage():
    """Demonstrate camera management."""
    print("\n=== Camera Management Example ===")
    
    try:
        # Create camera configuration
        config = CameraConfig(framerate=30, number_of_frames=3)
        
        # Initialize camera manager
        camera = CameraManager(config)
        
        if camera.connected:
            print("✓ Camera connected successfully")
            
            # Set up a simple frame processor
            def simple_processor(frame):
                # Simple threshold for demonstration
                gray = frame if len(frame.shape) == 2 else frame[:,:,0]
                return (gray > 128).astype(np.uint8) * 255
            
            camera.set_frame_processor(simple_processor)
            
            # Capture frames
            raw_frames, processed_frames = camera.get_frames(3)
            print(f"✓ Captured {len(raw_frames)} frames")
            
            # Clean up
            camera.cleanup()
            print("✓ Camera cleaned up")
            
        else:
            print("✗ No camera connected (this is expected in most environments)")
            
    except Exception as e:
        print(f"✗ Camera error: {e}")


def example_aruco_usage():
    """Demonstrate ArUco marker detection and calibration."""
    print("\n=== ArUco Detection Example ===")
    
    # Generate a test marker
    marker_path = Path("test_marker.png")
    if generate_aruco_marker(marker_id=5, size=200, output_dir=Path("."), display=False):
        print("✓ Generated test ArUco marker")
        
        # Initialize detector
        detector = ArUcoDetector()
        
        # Create a test image with the marker
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # In a real scenario, you would load the marker image and place it in the test image
        # For demonstration, we'll just show the API usage
        
        markers = detector.detect_markers(test_image)
        print(f"✓ Detected {len(markers)} markers")
        
        if markers:
            calibration = detector.calculate_calibration(markers)
            if calibration:
                print(f"  Calibration slope: {calibration.slope:.4f}")
                print(f"  Calibration intercept: {calibration.intercept:.4f}")
                
                # Convert measurements
                pixel_length = 100
                real_length = detector.convert_pixels_to_length(pixel_length)
                if real_length:
                    print(f"  {pixel_length} pixels = {real_length:.2f} units")
        
        # Clean up
        if marker_path.exists():
            marker_path.unlink()


def example_display_usage():
    """Demonstrate display and annotation features."""
    print("\n=== Display Management Example ===")
    
    # Create display manager
    display = DisplayManager()
    
    # Create test image
    test_frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    
    # Create display image with annotations
    display_image = display.create_display_image(
        test_frame,
        fish_id="test_fish_123",
        additional_text="Length: 15.2 cm\nWeight: 45.3 g",
        show_aruco=True,
        show_timestamp=True
    )
    
    if display_image:
        print("✓ Created annotated display image")
        
        # Get calibration info
        calib_info = display.get_calibration_info()
        print(f"  Calibration samples: {calib_info.get('sample_count', 0)}")
    else:
        print("✗ Failed to create display image")


def example_multiple_frames():
    """Demonstrate processing multiple frames of the same fish."""
    print("\n=== Multiple Frame Processing Example ===")
    
    processor = FishProcessor(output_folder=Path("multi_frame_output"))
    
    # Simulate multiple frames
    raw_frames = [
        np.random.randint(0, 255, (800, 1000, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    binary_frames = [
        np.random.randint(0, 2, (800, 1000), dtype=np.uint8) * 255
        for _ in range(3)
    ]
    
    # Process all frames
    results = processor.process_multiple_frames("multi_fish_001", raw_frames, binary_frames)
    
    successful_results = [r for r in results if r.success]
    print(f"✓ Processed {len(successful_results)}/{len(results)} frames successfully")
    
    if successful_results:
        # Get average measurements
        avg_data = processor.get_average_measurement(results)
        if avg_data:
            print(f"  Average total length: {avg_data['total_length_mean']:.2f} ± {avg_data['total_length_std']:.2f} pixels")
            print(f"  Based on {avg_data['measurement_count']} measurements")


def main():
    """Run all examples."""
    print("Fish Measurer Vision Module - Example Usage")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_camera_usage()
        example_aruco_usage()
        example_display_usage()
        example_multiple_frames()
        
        print("\n" + "=" * 50)
        print("✓ All examples completed successfully!")
        print("\nFor more information, see the README.md file.")
        
    except Exception as e:
        print(f"\n✗ Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
