"""
Statistics collector for measurement analysis and export.

This module handles statistical analysis and data export,
replacing the statistics logic previously embedded in MeasurerInstance.
"""

from typing import List, Optional, Any
import logging

from .measurement_statistics import MeasurementStatistics
from .measurement_config import MeasurementConfig
from .folder_manager import FolderManager
from .image_watermarker import ImageWatermarker

logger = logging.getLogger(__name__)


class StatisticsCollector:
    """
    Collects and analyzes measurement statistics.

    This class is responsible for:
    - Filtering outliers from measurements
    - Calculating statistical measures (mean, std, etc.)
    - Exporting data with watermarks
    - Validating measurement quality
    """

    def __init__(
        self,
        config: MeasurementConfig,
        folder_manager: FolderManager,
        image_format: str
    ):
        """
        Initialize the statistics collector.

        Args:
            config: Configuration manager
            folder_manager: Folder management instance
            image_format: Image file format
        """
        self.config = config
        self.folder_manager = folder_manager
        self.image_format = image_format
        self.watermarker = ImageWatermarker(config)

    def analyze_and_export(
        self,
        measurements: List[Any]
    ) -> tuple[bool, str]:
        """
        Analyze measurements and export results.

        Args:
            measurements: List of measurement instances

        Returns:
            Tuple of (success, message)
        """
        if not measurements:
            error_msg = "No measurements to analyze"
            logger.warning(error_msg)
            return False, error_msg

        try:
            # Filter outliers
            logger.info(f"Analyzing {len(measurements)} measurements")
            refined_measurements = self._filter_outliers(measurements)

            if not refined_measurements:
                error_msg = "All measurements were filtered as outliers"
                logger.warning(error_msg)
                self.config.add_error("interrupt", error_msg)
                return False, error_msg

            # Update trial count
            self.config.set_trial_count(len(refined_measurements))

            # Validate sufficient data
            if len(refined_measurements) < 2:
                error_msg = (
                    "Insufficient data for statistical analysis. "
                    f"Only {len(refined_measurements)} valid measurement(s) remained after filtering. "
                    "Please collect more measurements."
                )
                logger.warning(error_msg)
                self.config.add_error("interrupt", error_msg)
                return False, error_msg

            # Export data
            logger.info(f"Exporting {len(refined_measurements)} refined measurements")
            self._export_data(refined_measurements)

            success_msg = (
                f"Successfully analyzed and exported {len(refined_measurements)} measurements "
                f"(filtered from {len(measurements)} total)"
            )
            logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"Statistical analysis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.config.add_error("interrupt", error_msg)
            return False, error_msg

    def _filter_outliers(self, measurements: List[Any]) -> List[Any]:
        """
        Filter outliers from measurements.

        Args:
            measurements: Raw measurements

        Returns:
            Filtered measurements
        """
        try:
            refined = MeasurementStatistics.filter_outliers(measurements)

            num_filtered = len(measurements) - len(refined)
            if num_filtered > 0:
                logger.info(f"Filtered {num_filtered} outlier(s)")

            return refined

        except Exception as e:
            logger.error(f"Outlier filtering failed: {e}")
            # Return original measurements if filtering fails
            return measurements

    def _export_data(self, measurements: List[Any]) -> None:
        """
        Export measurement data with watermarks.

        Args:
            measurements: Filtered measurements to export
        """
        try:
            MeasurementStatistics.export_data(
                measurements=measurements,
                folder_manager=self.folder_manager,
                image_format=self.image_format,
                watermarker=self.watermarker
            )
            logger.info("Data export completed successfully")

        except Exception as e:
            logger.error(f"Data export failed: {e}", exc_info=True)
            raise

    def calculate_statistics(
        self,
        measurements: List[Any]
    ) -> Optional[dict]:
        """
        Calculate statistical measures for measurements.

        Args:
            measurements: List of measurements

        Returns:
            Dictionary with statistical measures or None if failed
        """
        if not measurements:
            return None

        try:
            # Extract length values
            lengths = []
            for measurement in measurements:
                if hasattr(measurement, 'total_length_pixels'):
                    lengths.append(measurement.total_length_pixels)
                elif hasattr(measurement, 'fil_length_pixels'):
                    lengths.append(measurement.fil_length_pixels)

            if not lengths:
                logger.warning("No length values found in measurements")
                return None

            import numpy as np

            stats = {
                'count': len(lengths),
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)),
                'min': float(np.min(lengths)),
                'max': float(np.max(lengths)),
                'median': float(np.median(lengths)),
                'variance': float(np.var(lengths))
            }

            logger.debug(f"Statistics calculated: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return None

    def validate_measurement_quality(
        self,
        measurements: List[Any],
        max_cv: float = 0.15
    ) -> tuple[bool, str]:
        """
        Validate measurement quality based on coefficient of variation.

        Args:
            measurements: List of measurements
            max_cv: Maximum acceptable coefficient of variation (default: 15%)

        Returns:
            Tuple of (is_valid, message)
        """
        stats = self.calculate_statistics(measurements)

        if not stats:
            return False, "Unable to calculate statistics"

        if stats['count'] < 2:
            return False, "Insufficient measurements for quality validation"

        # Calculate coefficient of variation
        cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else float('inf')

        if cv > max_cv:
            message = (
                f"Measurement quality is poor. Coefficient of variation: {cv:.2%} "
                f"(acceptable: <{max_cv:.2%}). Consider re-measuring with more stable conditions."
            )
            logger.warning(message)
            return False, message

        message = f"Measurement quality is acceptable (CV: {cv:.2%})"
        logger.info(message)
        return True, message

    def get_summary(self, measurements: List[Any]) -> str:
        """
        Get a human-readable summary of measurements.

        Args:
            measurements: List of measurements

        Returns:
            Summary string
        """
        stats = self.calculate_statistics(measurements)

        if not stats:
            return "No statistics available"

        summary_lines = [
            f"Measurement Summary:",
            f"  Count: {stats['count']}",
            f"  Mean: {stats['mean']:.2f} pixels",
            f"  Std Dev: {stats['std']:.2f} pixels",
            f"  Range: {stats['min']:.2f} - {stats['max']:.2f} pixels",
            f"  Median: {stats['median']:.2f} pixels"
        ]

        cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
        summary_lines.append(f"  CV: {cv:.2%}")

        return "\n".join(summary_lines)

    def reset(self) -> None:
        """Reset statistics collector state."""
        self.config.set_trial_count(0)
        logger.debug("Statistics collector reset")
