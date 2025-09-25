"""
Statistical analysis and data export for fish measurements.

This module handles outlier filtering, statistical calculations, and data export
for fish measurement results using modern statistical methods.
"""

from typing import List, Tuple, Optional, Dict, Any
import statistics
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from scipy import stats

from vision import ArUcoDetector
from .folder_manager import FolderManager
from .image_watermarker import ImageWatermarker


logger = logging.getLogger(__name__)


class MeasurementStatistics:
    """
    Statistical analysis and data export utilities for fish measurements.
    
    This class provides methods for filtering outliers, calculating statistics,
    and exporting measurement data to various formats.
    """
    
    @staticmethod
    def filter_outliers(
        measurements: List[Any], 
        method: str = "percentage", 
        threshold: float = 0.1
    ) -> List[Any]:
        """
        Filter outlier measurements based on length variance.
        
        Args:
            measurements: List of ProcessingInstance objects with fil_length_pixels
            method: Filtering method ("percentage", "zscore", "iqr")
            threshold: Threshold for filtering (default 10% for percentage method)
            
        Returns:
            Filtered list of measurements
            
        Raises:
            ValueError: If measurements list is empty or method is invalid
        """
        if not measurements:
            raise ValueError("Measurements list cannot be empty")
        
        # Extract lengths
        lengths = [m.fil_length_pixels for m in measurements]
        
        if len(lengths) < 2:
            return measurements
        
        try:
            if method == "percentage":
                return MeasurementStatistics._filter_by_percentage(
                    measurements, lengths, threshold
                )
            elif method == "zscore":
                return MeasurementStatistics._filter_by_zscore(
                    measurements, lengths, threshold
                )
            elif method == "iqr":
                return MeasurementStatistics._filter_by_iqr(
                    measurements, lengths, threshold
                )
            else:
                raise ValueError(f"Unknown filtering method: {method}")
                
        except Exception as e:
            logger.warning(f"Outlier filtering failed: {e}")
            return measurements
    
    @staticmethod
    def _filter_by_percentage(
        measurements: List[Any], 
        lengths: List[float], 
        threshold: float
    ) -> List[Any]:
        """Filter by percentage deviation from mean."""
        avg_length = statistics.mean(lengths)
        
        filtered = [
            m for m, length in zip(measurements, lengths)
            if abs((length - avg_length) / avg_length) <= threshold
        ]
        
        logger.info(
            f"Percentage filtering: {len(measurements)} -> {len(filtered)} "
            f"(threshold: {threshold:.1%})"
        )
        
        return filtered
    
    @staticmethod
    def _filter_by_zscore(
        measurements: List[Any], 
        lengths: List[float], 
        threshold: float = 2.0
    ) -> List[Any]:
        """Filter by Z-score (standard deviations from mean)."""
        if len(lengths) < 3:
            return measurements
        
        z_scores = np.abs(stats.zscore(lengths))
        
        filtered = [
            m for m, z_score in zip(measurements, z_scores)
            if z_score <= threshold
        ]
        
        logger.info(
            f"Z-score filtering: {len(measurements)} -> {len(filtered)} "
            f"(threshold: {threshold}σ)"
        )
        
        return filtered
    
    @staticmethod
    def _filter_by_iqr(
        measurements: List[Any], 
        lengths: List[float], 
        threshold: float = 1.5
    ) -> List[Any]:
        """Filter by Interquartile Range method."""
        if len(lengths) < 4:
            return measurements
        
        q1, q3 = np.percentile(lengths, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        filtered = [
            m for m, length in zip(measurements, lengths)
            if lower_bound <= length <= upper_bound
        ]
        
        logger.info(
            f"IQR filtering: {len(measurements)} -> {len(filtered)} "
            f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
        )
        
        return filtered
    
    @staticmethod
    def calculate_statistics(measurements: List[Any], aruco_detector: Optional[ArUcoDetector] = None) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for measurements.
        
        Args:
            measurements: List of ProcessingInstance objects
            
        Returns:
            Dictionary containing statistical measures
        """
        if not measurements:
            return {}
        
        lengths_pixels = [m.fil_length_pixels for m in measurements]
        
        # Use provided ArUco detector for calibration
        if aruco_detector:
            lengths_cm = [aruco_detector.convert_pixels_to_length(length) or 0.0 for length in lengths_pixels]
        else:
            logger.warning("No ArUco detector available for calibration")
            lengths_cm = [0.0] * len(lengths_pixels)
        
        try:
            stats_dict = {
                # Pixel measurements
                "mean_pixels": statistics.mean(lengths_pixels),
                "median_pixels": statistics.median(lengths_pixels),
                "std_pixels": statistics.stdev(lengths_pixels) if len(lengths_pixels) > 1 else 0.0,
                "min_pixels": min(lengths_pixels),
                "max_pixels": max(lengths_pixels),
                
                # Centimeter measurements  
                "mean_cm": statistics.mean(lengths_cm),
                "median_cm": statistics.median(lengths_cm),
                "std_cm": statistics.stdev(lengths_cm) if len(lengths_cm) > 1 else 0.0,
                "min_cm": min(lengths_cm),
                "max_cm": max(lengths_cm),
                
                # Sample statistics
                "count": len(measurements),
                "cv_percent": (statistics.stdev(lengths_cm) / statistics.mean(lengths_cm) * 100) if len(lengths_cm) > 1 and statistics.mean(lengths_cm) > 0 else 0.0,
            }
            
            # Add confidence interval (95%)
            if len(lengths_cm) > 1:
                sem = stats.sem(lengths_cm)
                ci = stats.t.interval(0.95, len(lengths_cm) - 1, 
                                    loc=stats_dict["mean_cm"], scale=sem)
                stats_dict["ci_95_lower"] = ci[0]
                stats_dict["ci_95_upper"] = ci[1]
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {"count": len(measurements)}
    
    @staticmethod
    def export_data(
        measurements: List[Any],
        folder_manager: FolderManager,
        image_format: str,
        watermarker: ImageWatermarker,
        aruco_detector: Optional[ArUcoDetector] = None
    ) -> None:
        """
        Export measurement data and create watermarked images.
        
        Args:
            measurements: List of successful ProcessingInstance objects
            folder_manager: Folder management instance
            image_format: Image format string (e.g., ".jpg")
            watermarker: Image watermarking instance
        """
        if not measurements:
            logger.warning("No measurements to export")
            return
        
        try:
            # Calculate statistics
            stats = MeasurementStatistics.calculate_statistics(measurements, aruco_detector)
            
            # Export CSV data
            MeasurementStatistics._export_csv(measurements, folder_manager, stats, aruco_detector)
            
            # Find representative image (closest to mean)
            representative = MeasurementStatistics._find_representative_measurement(
                measurements, stats.get("mean_cm", 0), aruco_detector
            )
            
            # Create and save watermarked images
            MeasurementStatistics._export_images(
                measurements, representative, folder_manager, image_format, watermarker, stats
            )
            
            # Log calibration information
            if aruco_detector:
                calibration = aruco_detector.get_average_calibration()
                if calibration:
                    logger.info(f"Calibration: y = {calibration.slope:.2f}x + {calibration.intercept:.2f}")
                else:
                    logger.warning("No calibration data available")
            else:
                logger.warning("No ArUco detector available for calibration info")
            logger.info(f"Final statistics: Mean={stats.get('mean_cm', 0):.2f}cm, "
                       f"Std={stats.get('std_cm', 0):.2f}cm, N={stats.get('count', 0)}")
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise
    
    @staticmethod
    def _export_csv(
        measurements: List[Any], 
        folder_manager: FolderManager,
        stats: Dict[str, float],
        aruco_detector: Optional[ArUcoDetector] = None
    ) -> None:
        """Export measurements to CSV file."""
        try:
            # Prepare data
            data = {
                "frame_number": [m.process_id for m in measurements],
                "length_pixels": [m.fil_length_pixels for m in measurements],
                "length_cm_filfinder": [aruco_detector.convert_pixels_to_length(m.fil_length_pixels) or 0.0 for m in measurements] if aruco_detector else [0.0] * len(measurements),
            }
            
            # Add optional measurements if available
            if hasattr(measurements[0], 'long_path_pixel_coords'):
                data["length_cm_pixel_count"] = [
                    aruco_detector.convert_pixels_to_length(len(m.long_path_pixel_coords)) or 0.0
                    for m in measurements
                ] if aruco_detector else [0.0] * len(measurements)
            
            if hasattr(measurements[0], 'manual_length'):
                data["length_cm_manual"] = [
                    aruco_detector.convert_pixels_to_length(m.manual_length) or 0.0
                    for m in measurements
                ] if aruco_detector else [0.0] * len(measurements)
            
            # Create DataFrame and export
            df = pd.DataFrame(data)
            
            # Add summary statistics as metadata
            metadata_rows = []
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    metadata_rows.append({
                        "frame_number": f"STAT_{key.upper()}",
                        "length_pixels": value if "pixels" in key else "",
                        "length_cm_filfinder": value if "cm" in key or key == "count" else "",
                    })
            
            if metadata_rows:
                metadata_df = pd.DataFrame(metadata_rows)
                df = pd.concat([df, metadata_df], ignore_index=True)
            
            # Save to CSV
            csv_path = folder_manager.target_folder / "measurement_data.csv"
            df.to_csv(csv_path, sep=';', index=False, float_format='%.3f')
            
            logger.info(f"CSV data exported to {csv_path}")
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise
    
    @staticmethod
    def _find_representative_measurement(
        measurements: List[Any], 
        target_cm: float,
        aruco_detector: Optional[ArUcoDetector] = None
    ) -> Any:
        """Find measurement closest to target length."""
        if not measurements:
            return None
        
        if aruco_detector:
            closest = min(
                measurements,
                key=lambda m: abs((aruco_detector.convert_pixels_to_length(m.fil_length_pixels) or 0.0) - target_cm)
            )
        else:
            # Fallback to first measurement if no calibration
            closest = measurements[0] if measurements else None
            logger.warning("No ArUco detector available, using first measurement as representative")
        
        if closest and aruco_detector:
            length_cm = aruco_detector.convert_pixels_to_length(closest.fil_length_pixels) or 0.0
            logger.info(f"Representative measurement: Frame {closest.process_id}, "
                       f"Length {length_cm:.2f}cm")
        elif closest:
            logger.info(f"Representative measurement: Frame {closest.process_id}, "
                       f"Length {closest.fil_length_pixels:.2f}px (no calibration)")
        
        return closest
    
    @staticmethod
    def _export_images(
        measurements: List[Any],
        representative: Any,
        folder_manager: FolderManager,
        image_format: str,
        watermarker: ImageWatermarker,
        stats: Dict[str, float]
    ) -> None:
        """Export watermarked images."""
        try:
            # Create representative image with full statistics
            if representative:
                main_image = watermarker.create_watermarked_image(
                    representative, stats
                )
                
                # Save main result image
                main_path = folder_manager.target_folder / f"representative_measurement{image_format}"
                cv2.imwrite(str(main_path), main_image)
                
                logger.info(f"Representative image saved to {main_path}")
            
            # Create watermarked versions of all measurements
            watermarked_folder = folder_manager.get_watermarked_folder()
            
            for measurement in measurements:
                if measurement == representative:
                    # Use the detailed version for representative
                    watermarked = main_image.copy()
                else:
                    # Simple version for others
                    watermarked = watermarker.create_watermarked_image(measurement)
                
                watermarked_path = watermarked_folder / f"watermarked-{measurement.process_id}{image_format}"
                cv2.imwrite(str(watermarked_path), watermarked)
            
            logger.info(f"Watermarked images saved to {watermarked_folder}")
            
        except Exception as e:
            logger.error(f"Image export failed: {e}")
            raise