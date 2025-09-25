"""
Folder structure management for fish measurement data.

This module handles the creation and management of output folder structures
for organizing measurement data, images, and results.
"""

from typing import Optional, Union
from pathlib import Path
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class FolderManager:
    """
    Manages folder structure for measurement data organization.
    
    This class handles the creation and management of folder hierarchies
    for storing raw images, processed images, watermarked images, and data files.
    """
    
    def __init__(self):
        """Initialize the folder manager."""
        self.target_folder: Optional[Path] = None
        self.raw_folder: Optional[Path] = None
        self.skeleton_folder: Optional[Path] = None
        self.watermarked_folder: Optional[Path] = None
        self.frames_folder: Optional[Path] = None
    
    def setup_folders(
        self, 
        output_folder: Union[str, Path], 
        fish_id: Optional[str] = None
    ) -> bool:
        """
        Set up the complete folder structure for a measurement session.
        
        Args:
            output_folder: Base output directory path
            fish_id: Optional fish identifier for folder naming
            
        Returns:
            True if setup successful, False otherwise
            
        Raises:
            ValueError: If output_folder is invalid
            OSError: If folder creation fails
        """
        if not output_folder:
            raise ValueError("Output folder path cannot be empty")
        
        try:
            base_path = Path(output_folder)
            
            if not base_path.exists():
                logger.warning(f"Output folder does not exist: {base_path}")
                base_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output folder: {base_path}")
            
            # Create session-specific folder
            session_name = self._generate_session_name(fish_id)
            self.target_folder = base_path / session_name
            
            # Create main session folder
            self.target_folder.mkdir(exist_ok=True)
            logger.info(f"Created session folder: {self.target_folder}")
            
            # Create frames subdirectory
            self.frames_folder = self.target_folder / "frames"
            self.frames_folder.mkdir(exist_ok=True)
            
            # Create image type subdirectories
            self._create_image_folders()
            
            logger.info(f"Folder structure created successfully in {self.target_folder}")
            return True
            
        except Exception as e:
            logger.error(f"Folder setup failed: {e}")
            return False
    
    def _generate_session_name(self, fish_id: Optional[str] = None) -> str:
        """
        Generate a unique session folder name.
        
        Args:
            fish_id: Optional fish identifier
            
        Returns:
            Session folder name string
        """
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        
        if fish_id and fish_id.strip():
            # Sanitize fish_id for filesystem
            clean_fish_id = "".join(c for c in fish_id.strip() if c.isalnum() or c in "-_")
            return f"{timestamp}_ID-{clean_fish_id}"
        else:
            return timestamp
    
    def _create_image_folders(self) -> None:
        """Create subdirectories for different image types."""
        try:
            # Raw images folder
            self.raw_folder = self.frames_folder / "raw"
            self.raw_folder.mkdir(exist_ok=True)
            
            # Skeleton and longest path images
            self.skeleton_folder = self.frames_folder / "skeleton_and_longpath"
            self.skeleton_folder.mkdir(exist_ok=True)
            
            # Watermarked images
            self.watermarked_folder = self.frames_folder / "watermarked"
            self.watermarked_folder.mkdir(exist_ok=True)
            
            logger.debug("Image subfolders created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create image folders: {e}")
            raise
    
    def get_raw_path(self, frame_id: Union[int, str], format: str) -> Path:
        """
        Get the path for saving a raw image.
        
        Args:
            frame_id: Frame identifier
            format: Image format (e.g., ".jpg", ".png")
            
        Returns:
            Path object for the raw image file
        """
        if self.raw_folder is None:
            raise RuntimeError("Folders not initialized. Call setup_folders first.")
        
        filename = f"raw-{frame_id}{format}"
        return self.raw_folder / filename
    
    def get_skeleton_path(self, frame_id: Union[int, str], format: str) -> Path:
        """
        Get the path for saving a skeleton/longpath image.
        
        Args:
            frame_id: Frame identifier
            format: Image format (e.g., ".jpg", ".png")
            
        Returns:
            Path object for the skeleton image file
        """
        if self.skeleton_folder is None:
            raise RuntimeError("Folders not initialized. Call setup_folders first.")
        
        filename = f"skeleton_LP-{frame_id}{format}"
        return self.skeleton_folder / filename
    
    def get_watermarked_path(self, frame_id: Union[int, str], format: str) -> Path:
        """
        Get the path for saving a watermarked image.
        
        Args:
            frame_id: Frame identifier
            format: Image format (e.g., ".jpg", ".png")
            
        Returns:
            Path object for the watermarked image file
        """
        if self.watermarked_folder is None:
            raise RuntimeError("Folders not initialized. Call setup_folders first.")
        
        filename = f"watermarked-{frame_id}{format}"
        return self.watermarked_folder / filename
    
    def get_data_path(self, filename: str) -> Path:
        """
        Get the path for saving data files.
        
        Args:
            filename: Data file name
            
        Returns:
            Path object for the data file
        """
        if self.target_folder is None:
            raise RuntimeError("Folders not initialized. Call setup_folders first.")
        
        return self.target_folder / filename
    
    def get_target_folder(self) -> Optional[Path]:
        """Get the main target folder path."""
        return self.target_folder
    
    def get_frames_folder(self) -> Optional[Path]:
        """Get the frames folder path."""
        return self.frames_folder
    
    def get_raw_folder(self) -> Optional[Path]:
        """Get the raw images folder path."""
        return self.raw_folder
    
    def get_skeleton_folder(self) -> Optional[Path]:
        """Get the skeleton images folder path."""
        return self.skeleton_folder
    
    def get_watermarked_folder(self) -> Optional[Path]:
        """Get the watermarked images folder path."""
        return self.watermarked_folder
    
    def is_initialized(self) -> bool:
        """Check if the folder manager has been initialized."""
        return all([
            self.target_folder is not None,
            self.frames_folder is not None,
            self.raw_folder is not None,
            self.skeleton_folder is not None,
            self.watermarked_folder is not None
        ])
    
    def get_folder_summary(self) -> dict:
        """
        Get a summary of the folder structure.
        
        Returns:
            Dictionary with folder information
        """
        if not self.is_initialized():
            return {"status": "not_initialized"}
        
        try:
            return {
                "status": "initialized",
                "target_folder": str(self.target_folder),
                "frames_folder": str(self.frames_folder),
                "raw_folder": str(self.raw_folder),
                "skeleton_folder": str(self.skeleton_folder),
                "watermarked_folder": str(self.watermarked_folder),
                "target_exists": self.target_folder.exists(),
                "frames_exists": self.frames_folder.exists(),
                "raw_exists": self.raw_folder.exists(),
                "skeleton_exists": self.skeleton_folder.exists(),
                "watermarked_exists": self.watermarked_folder.exists(),
            }
        except Exception as e:
            logger.error(f"Failed to generate folder summary: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup_empty_folders(self) -> None:
        """Remove empty folders from the structure."""
        if not self.is_initialized():
            return
        
        try:
            folders_to_check = [
                self.watermarked_folder,
                self.skeleton_folder,
                self.raw_folder,
                self.frames_folder
            ]
            
            for folder in folders_to_check:
                if folder and folder.exists():
                    try:
                        # Only remove if empty
                        folder.rmdir()
                        logger.info(f"Removed empty folder: {folder}")
                    except OSError:
                        # Folder not empty, which is fine
                        pass
                        
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def reset(self) -> None:
        """Reset the folder manager to uninitialized state."""
        self.target_folder = None
        self.raw_folder = None
        self.skeleton_folder = None
        self.watermarked_folder = None
        self.frames_folder = None
        
        logger.debug("Folder manager reset")
    
    def __repr__(self) -> str:
        """String representation of the folder manager."""
        if self.is_initialized():
            return f"FolderManager(target='{self.target_folder}')"
        else:
            return "FolderManager(uninitialized)"


# Legacy function wrapper for backward compatibility
def CreateFolderStructure(output_folder, fish_id=None):
    """Legacy function wrapper for folder creation."""
    manager = FolderManager()
    success = manager.setup_folders(output_folder, fish_id)
    
    if success:
        return {
            'target_folder': str(manager.target_folder),
            'raw_folder': str(manager.raw_folder),
            'skeleton_LP_folder': str(manager.skeleton_folder),
            'watermarked_folder': str(manager.watermarked_folder)
        }
    else:
        return None
