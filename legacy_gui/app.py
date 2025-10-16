"""Modern Fish Measurer GUI Application using MVC architecture."""

import tkinter as tk
from tkinter import messagebox
import sys
import threading
import math
from typing import Optional

from .config import AppConfig
from .events import event_bus, EventType, Event, publish_error
from .components import (
    CameraSettingsPanel, 
    OutputSettingsPanel, 
    ProcessingControlPanel, 
    VideoDisplayPanel
)
from .camera_controller import CameraController

# Import the original classes for compatibility
from logic.measurer_instance import MeasurerInstance
from vision import CameraManager


class FishMeasurerApplication:
    """Modern Fish Measurer application with modular architecture."""
    
    def __init__(self, **kwargs):
        # Configuration
        self.config = AppConfig()
        
        # Error handling
        self.block_start_already_popped = False
        self.interrupt_already_popped = False
        self._camera_disconnect_handled = False
        
        # Processing components
        self.measurer_instance: Optional[MeasurerInstance] = None
        self.analysis_thread: Optional[threading.Thread] = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.geometry("1133x850")
        self.root.title('Fish Measurer')
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Create shared camera manager
        self.camera_manager = CameraManager()
        
        # Create camera controller with shared manager
        self.camera_controller = CameraController(camera_manager=self.camera_manager)
        
        # Setup UI
        self._setup_ui()
        
        # Setup event handlers
        self._setup_events()
        
        # Initialize camera controller
        if not self.camera_controller.initialize():
            messagebox.showerror("Camera Error", "No camera detected. The application will now close.")
            sys.exit()
    
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Settings panel frame
        settings_frame = tk.Frame(self.root, relief='flat', background="grey80")
        settings_frame.grid(row=0, column=0, sticky='ns')
        
        # Create components
        self.camera_settings = CameraSettingsPanel(settings_frame, self.config, self.camera_controller)
        self.output_settings = OutputSettingsPanel(settings_frame, self.config)
        self.processing_controls = ProcessingControlPanel(
            settings_frame, 
            self.config,
            on_train_background=self._on_train_background,
            on_start_analysis=self._on_start_analysis
        )
        
        # Video display
        self.video_display = VideoDisplayPanel(self.root, self.config)
    
    def _setup_events(self) -> None:
        """Setup event handlers."""
        event_bus.subscribe(EventType.ERROR_OCCURRED, self._on_error_occurred)
    
    def _start_video_feed(self) -> None:
        """Start the video feed update loop."""
        self._update_video_feed()
    
    def _update_video_feed(self) -> None:
        """Update video feed from camera."""
        try:
            # Detect camera unplug/disconnect and close app
            if not self._camera_disconnect_handled and not self.camera_controller.is_camera_connected():
                self._camera_disconnect_handled = True
                messagebox.showerror(
                    "Camera Disconnected",
                    "Camera connection lost. The application will now close."
                )
                self._on_closing()
                return

            # Update MeasurerInstance with current settings for compatibility
            if self.measurer_instance:
                self.measurer_instance.config.set_fish_id(self.config.output.fish_id)
                self.measurer_instance.config.set_additional_text(self.config.output.additional_text)
            
            # Get new frame from camera
            new_image = self.camera_controller.get_current_frame()
            
            if new_image is not None:
                # Process frame for ArUco detection if measurer instance exists
                display_image = new_image.copy()
                if self.measurer_instance:
                    # Detect ArUco markers for calibration
                    markers = self.measurer_instance.aruco_detector.detect_markers(new_image)
                    if markers:
                        # Draw markers on display image
                        display_image = self.measurer_instance.aruco_detector.draw_markers(display_image, markers)
                        # Update calibration
                        self.measurer_instance.aruco_detector.calculate_calibration(markers)
                        
                        # Update button states based on calibration (check regardless of state)
                        is_calibrated = self.check_if_calibrated()
                        self._update_start_button_state(is_calibrated)
                    else:
                        # No markers detected - still update button state
                        is_calibrated = self.check_if_calibrated()
                        self._update_start_button_state(is_calibrated)
                
                self.video_display.update_image(display_image)
            else:
                # Show a placeholder message
                if self.video_display.image_label:
                    self.video_display.image_label.configure(text="Waiting for camera feed...")
                    self.video_display.image_label.image = None
                
                # Update button state even when no camera feed
                if self.measurer_instance:
                    is_calibrated = self.check_if_calibrated()
                    self._update_start_button_state(is_calibrated)
            
            # Schedule next update
            delay = math.ceil(1000 / self.config.camera.framerate_fps)
            self.root.after(delay, self._update_video_feed)
            
        except Exception as e:
            publish_error("video_feed", f"Error updating video feed: {str(e)}")
            # Continue trying to update
            self.root.after(100, self._update_video_feed)
    
    def _on_train_background(self) -> None:
        """Handle background training button click."""
        if self.config.current_state == 0:
            # Start training
            self._start_background_training()
        elif self.config.current_state == 1:
            # Restart training
            self._restart_background_training()
        elif self.config.current_state == 2:
            # Cancel analysis
            self._cancel_analysis()
    
    def _on_start_analysis(self) -> None:
        """Handle start analysis button click."""
        if self.config.current_state == 1:
            # Check calibration before starting analysis
            if not self.check_if_calibrated():
                messagebox.showerror(
                    "Calibration Required",
                    "Camera must be calibrated with ArUco markers before starting analysis.\n\n"
                    "Please ensure ArUco markers are visible in the camera view and wait for "
                    "the 'PLS CALIBRATE' button to change to 'START'."
                )
                return
            self._start_analysis()
    
    def _start_background_training(self) -> None:
        """Start background training process."""
        self._set_app_state(2)  # Training state
        self._lock_settings(True)
        
        # Create measurer instance with shared camera manager
        self.measurer_instance = MeasurerInstance(camera_manager=self.camera_manager)
        
        # Start error checking
        self._start_error_checking()
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self.measurer_instance.train_background,
            daemon=True
        )
        training_thread.start()
        
        # Update button state
        self._update_training_button_state("GATHERING", training_thread)
    
    def _restart_background_training(self) -> None:
        """Restart background training."""
        self._set_app_state(0)
        self._lock_settings(False)
        
        # Cleanup previous instance
        if self.measurer_instance:
            # Measurer instance cleanup is handled internally
            self.measurer_instance = None
    
    def _cancel_analysis(self) -> None:
        """Cancel running analysis."""
        if self.measurer_instance:
            self.measurer_instance.config.request_stop()
        
        self.processing_controls.background_button.configure(state="disabled")
    
    def _start_analysis(self) -> None:
        """Start analysis process."""
        self._set_app_state(2)
        
        # Validate frame count
        if self.config.processing.number_of_frames < 3:
            self.config.processing.number_of_frames = 3
            self.processing_controls.frame_count_entry.set_value(3)
        
        # Lock settings
        self.processing_controls.set_processing_settings_enabled(False)
        self.output_settings.set_enabled(False)
        
        # Update measurer with current settings
        if self.measurer_instance:
            self.measurer_instance.output_folder = str(self.config.output.folder_path)
            self.measurer_instance.image_format = self.config.output.file_format
        
        # Start analysis using the measurer instance
        self.analysis_thread = threading.Thread(
            target=self._run_analysis,
            daemon=True
        )
        self.analysis_thread.start()
        
        # Monitor analysis progress
        self._monitor_analysis_progress()
    
    def _update_training_button_state(self, label: str, thread: threading.Thread) -> None:
        """Update background training button state."""
        self.processing_controls.background_button.configure(text=label)
        
        if not thread.is_alive():
            self._set_app_state(1)  # Background trained
            self._lock_settings(False)
            # Check calibration and update start button accordingly
            is_calibrated = self.check_if_calibrated()
            self._update_start_button_state(is_calibrated)
        else:
            # Animate button text
            if label.endswith("..."):
                next_label = label[:-3]
            else:
                next_label = label + "."
            
            self.root.after(1000, self._update_training_button_state, next_label, thread)
    
    def _update_start_button_state(self, is_calibrated: bool) -> None:
        """Update start button state based on calibration and training status."""
        current_state = self.config.current_state
        
        # Don't update button during training/analysis (state 2)
        if current_state == 2:
            return
            
        background_trained = (current_state == 1)
        
        if is_calibrated:
            # Calibration exists - show START text
            if background_trained:
                # Both background trained AND calibrated - enable button
                self.processing_controls.start_button.configure(
                    bg="#74B224", 
                    state="normal", 
                    text='START'
                )
            else:
                # Calibrated but no background - disabled START
                self.processing_controls.start_button.configure(
                    bg="grey50", 
                    state="disabled", 
                    text='START'
                )
        else:
            # No calibration - always show PLS CALIBRATE (disabled)
            self.processing_controls.start_button.configure(
                bg="grey50", 
                state="disabled", 
                text='PLS CALIBRATE'
            )
    
    def _monitor_analysis_progress(self) -> None:
        """Monitor analysis progress."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            # Update progress display
            if self.measurer_instance:
                processing_frame = self.measurer_instance.config.get_processing_frame()
                if processing_frame is None:
                    text = "COLLECTING..."
                else:
                    progress = self.measurer_instance.config.get_completed_threads() + 1
                    total = self.config.processing.number_of_frames
                    text = f"PROCESSED: {progress}/{total}"
                
                self.processing_controls.start_button.configure(text=text)
            
            # Schedule next check
            self.root.after(500, self._monitor_analysis_progress)
        else:
            # Analysis completed
            self._analysis_completed()
    
    def _analysis_completed(self) -> None:
        """Handle analysis completion."""
        self._set_app_state(1)  # Back to background trained state
        
        if self.measurer_instance:
            self.measurer_instance.config.reset_stop()
        
        # Unlock settings
        self.processing_controls.set_processing_settings_enabled(True)
        self.output_settings.set_enabled(True)
    
    def _start_error_checking(self) -> None:
        """Start error checking loop."""
        if self.root.winfo_exists():
            self.root.after(100, self._check_for_errors)
    
    def _check_for_errors(self) -> None:
        """Check for processing errors."""
        if not self.measurer_instance:
            return
        
        try:
            # Check for interruption errors
            interrupt_errors = self.measurer_instance.config.get_errors("interrupt")
            if interrupt_errors:
                if not self.interrupt_already_popped:
                    self.interrupt_already_popped = True
                    messagebox.showerror("Analysis Error", interrupt_errors[0])
                    self.measurer_instance.config.clear_errors("interrupt")
            else:
                self.interrupt_already_popped = False
            
            # Check for start block errors
            if hasattr(self.measurer_instance, 'block_tkinter_start_button'):
                if self.measurer_instance.block_tkinter_start_button:
                    if not self.block_start_already_popped:
                        self.block_start_already_popped = True
                        messagebox.showerror(
                            "Shape is Missing!",
                            "Failing to register any objects in the arena. Please ensure an object is present and contrasted against the trained background"
                        )
                else:
                    self.block_start_already_popped = False
            
            # Continue checking if window still exists
            if self.root.winfo_exists():
                self.root.after(100, self._check_for_errors)
            
        except Exception as e:
            publish_error("error_checking", f"Error in error checking loop: {str(e)}")
    
    def _set_app_state(self, new_state: int) -> None:
        """Set application state and update UI."""
        old_state = self.config.current_state
        self.config.current_state = new_state
        
        # Update button states
        self.processing_controls.update_button_states(new_state)
    
    def _lock_settings(self, lock: bool) -> None:
        """Lock or unlock settings panels."""
        self.camera_settings.set_enabled(not lock)
    
    def _on_error_occurred(self, event: Event) -> None:
        """Handle error events."""
        if event.data:
            error_type = event.data.get("error_type", "Unknown")
            message = event.data.get("message", "An error occurred")
            messagebox.showerror(f"Error: {error_type}", message)
    
    def _on_closing(self) -> None:
        """Handle application closing."""
        # Cleanup camera system
        if hasattr(self, 'camera_controller') and self.camera_controller:
            self.camera_controller.shutdown()
        
        if hasattr(self, 'camera_manager') and self.camera_manager:
            self.camera_manager.cleanup()
        
        # Close stdout if redirected (for debugging)
        if hasattr(sys.stdout, 'close') and sys.stdout != sys.__stdout__:
            sys.stdout.close()
        
        self.root.destroy()
    
    def run(self) -> None:
        """Run the application."""
        # Start video feed after GUI is ready
        self.root.after(100, self._start_video_feed)
        self.root.mainloop()
    
    def check_if_calibrated(self) -> bool:
        """Check if camera is calibrated with ArUco markers."""
        if not self.measurer_instance:
            return False
        
        # Check if we have valid calibration data
        calibration = self.measurer_instance.aruco_detector.get_average_calibration()
        if calibration is None:
            return False
        
        # Require at least 3 calibration samples for stability
        if calibration.marker_count < 3:
            return False
        
        return True
    
    def _run_analysis(self) -> None:
        """Run analysis in separate thread."""
        try:
            if self.measurer_instance:
                # Capture frames for analysis
                frames = self.camera_controller.capture_frames(
                    self.config.processing.number_of_frames
                )
                
                if frames:
                    # Use the first frame for analysis (simplified)
                    # The actual analysis would need to be integrated with the new vision module
                    pass
        except Exception as e:
            publish_error("analysis", f"Analysis failed: {str(e)}")


