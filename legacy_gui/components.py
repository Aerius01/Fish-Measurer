"""Modular GUI components for the Fish Measurer application."""

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path

from .config import AppConfig, CameraSettings, OutputSettings, ProcessingSettings
from .events import event_bus, EventType, Event, publish_settings_changed
from .widgets import ValidatedEntry, create_exposure_entry, create_framerate_entry, create_frame_count_entry
from .camera_controller import CameraController


# Constants
CAMERA_MODES = ["Once", "Continuous", "Off"]
OUTPUT_FORMATS = [".jpeg", ".png", ".tiff"]


class BaseComponent(tk.Frame):
    """Base class for GUI components with event handling."""
    
    def __init__(self, parent: tk.Widget, config: AppConfig, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config
        self._event_handlers: Dict[EventType, Callable[[Event], None]] = {}
        self._setup_ui()
        self._setup_events()
    
    def _setup_ui(self) -> None:
        """Setup the UI components. Override in subclasses."""
        pass
    
    def _setup_events(self) -> None:
        """Setup event handlers. Override in subclasses."""
        pass
    
    def subscribe_to_event(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Subscribe to an event type."""
        self._event_handlers[event_type] = handler
        event_bus.subscribe(event_type, handler)
    
    def destroy(self) -> None:
        """Clean up event subscriptions when component is destroyed."""
        for event_type, handler in self._event_handlers.items():
            event_bus.unsubscribe(event_type, handler)
        super().destroy()


class CameraSettingsPanel(BaseComponent):
    """Panel for camera settings and selection."""
    
    def __init__(self, parent: tk.Widget, config: AppConfig, camera_controller: CameraController):
        self.camera_controller = camera_controller
        super().__init__(parent, config, relief='flat', borderwidth=2, padx=2, pady=10, bg="grey80")
    
    def _setup_ui(self) -> None:
        """Setup camera settings UI."""
        self.pack(fill=tk.X)
        
        # Header
        header_frame = tk.Frame(self, relief='groove', borderwidth=2, padx=5, pady=5, bg="grey25")
        header_frame.pack(fill=tk.X)
        
        header_label = tk.Label(
            header_frame, 
            text="CAMERA SETTINGS", 
            background="grey25", 
            fg="white",
            font=("Courier", 24)
        )
        header_label.pack(fill=tk.X)
        
        # Camera info (read-only)
        self._create_camera_info()
        
        # Camera settings
        self._create_camera_settings()
    
    def _create_camera_info(self) -> None:
        """Create camera info display (no selection)."""
        info_frame = tk.Frame(self, relief='flat', borderwidth=2, pady=5, bg="grey80")
        info_frame.pack(fill=tk.X)

        prompt_label = tk.Label(
            info_frame,
            text="Camera in use:",
            bg="grey80",
            anchor="w",
            font=("Courier", 10)
        )
        prompt_label.pack(fill=tk.X)

        self.camera_info_var = tk.StringVar(value=self._get_camera_info_text())
        info_label = tk.Label(
            info_frame,
            textvariable=self.camera_info_var,
            bg="grey80",
            anchor="w"
        )
        info_label.pack(fill=tk.X)

    def _get_camera_info_text(self) -> str:
        """Return a short description of the active camera."""
        info = self.camera_controller.get_current_camera_info()
        if info is None:
            return "No camera detected"
        return f"{info.model_name}; S/N: {info.serial_number}"
    
    def _create_camera_settings(self) -> None:
        """Create camera settings inputs."""
        settings_frame = tk.Frame(self, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey60")
        settings_frame.pack(fill=tk.X)
        
        inputs_frame = tk.Frame(settings_frame, relief='flat', borderwidth=2)
        inputs_frame.pack(fill=tk.X)
        inputs_frame.columnconfigure(0, weight=0)
        inputs_frame.columnconfigure(1, weight=1)
        
        # Exposure setting
        self.exposure_entry = create_exposure_entry(
            inputs_frame,
            on_change=self._on_exposure_changed,
            row=0
        )
        
        # Gain setting
        self._create_dropdown_setting(
            inputs_frame,
            "Gain Setting:",
            CAMERA_MODES,
            self.config.camera.gain_mode,
            self._on_gain_changed,
            row=1
        )
        
        # White balance setting
        self._create_dropdown_setting(
            inputs_frame,
            "White Balance:",
            CAMERA_MODES,
            self.config.camera.white_balance_mode,
            self._on_white_balance_changed,
            row=2
        )
        
        # Framerate setting
        self.framerate_entry = create_framerate_entry(
            inputs_frame,
            on_change=self._on_framerate_changed,
            row=3
        )
    
    def _create_dropdown_setting(self, parent: tk.Widget, label_text: str, options: List[str], 
                                default: str, callback: Callable[[str], None], row: int) -> None:
        """Create a dropdown setting."""
        label = tk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky='w', padx=5)
        
        var = tk.StringVar(value=default)
        dropdown = ttk.Combobox(parent, textvariable=var, values=options, state="readonly")
        dropdown.grid(row=row, column=1, sticky='ew', padx=5)
        dropdown.bind('<<ComboboxSelected>>', lambda e: callback(var.get()))
        
        # Store reference for state management
        setattr(self, f"_{label_text.lower().replace(' ', '_').replace(':', '')}_dropdown", dropdown)
    
    # Removed camera selection UI and related handlers
    
    def _on_exposure_changed(self, value: str) -> None:
        """Handle exposure change."""
        try:
            exposure = float(value) if value else 2000.0
            self.config.camera.exposure_ms = exposure
            self.camera_controller.update_exposure(exposure)
            publish_settings_changed("camera", self.config.camera.__dict__)
        except ValueError:
            pass
    
    def _on_gain_changed(self, value: str) -> None:
        """Handle gain mode change."""
        self.config.camera.gain_mode = value
        self.camera_controller.update_gain_mode(value)
        publish_settings_changed("camera", self.config.camera.__dict__)
    
    def _on_white_balance_changed(self, value: str) -> None:
        """Handle white balance change."""
        self.config.camera.white_balance_mode = value
        self.camera_controller.update_white_balance_mode(value)
        publish_settings_changed("camera", self.config.camera.__dict__)
    
    def _on_framerate_changed(self, value: str) -> None:
        """Handle framerate change."""
        try:
            framerate = float(value) if value else 30.0
            self.config.camera.framerate_fps = framerate
            self.camera_controller.update_framerate(framerate)
            publish_settings_changed("camera", self.config.camera.__dict__)
        except ValueError:
            pass
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable all camera settings."""
        state = "readonly" if enabled else "disabled"

        # Enable/disable entry widgets
        self.exposure_entry.set_enabled(enabled)
        self.framerate_entry.set_enabled(enabled)
        
        # Enable/disable dropdowns
        if hasattr(self, '_gain_setting_dropdown'):
            self._gain_setting_dropdown.configure(state=state)
        if hasattr(self, '_white_balance_dropdown'):
            self._white_balance_dropdown.configure(state=state)


class OutputSettingsPanel(BaseComponent):
    """Panel for output settings."""
    
    def _setup_ui(self) -> None:
        """Setup output settings UI."""
        self.pack(fill=tk.BOTH)
        
        # Header
        header_frame = tk.Frame(self, relief='groove', borderwidth=2, padx=5, pady=5, bg="grey25")
        header_frame.pack(fill=tk.X)
        
        header_label = tk.Label(
            header_frame,
            text="OUTPUT SETTINGS",
            background="grey25",
            fg="white",
            font=("Courier", 24)
        )
        header_label.pack(fill=tk.X)
        
        # Settings frame
        settings_frame = tk.Frame(self, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey80")
        settings_frame.pack(fill=tk.X)
        
        # Folder selection
        self._create_folder_selection(settings_frame)
        
        # Format selection
        self._create_format_selection(settings_frame)
        
        # Watermark settings
        self._create_watermark_settings(settings_frame)
    
    def _create_folder_selection(self, parent: tk.Widget) -> None:
        """Create folder selection UI."""
        prompt_label = tk.Label(
            parent,
            text="Choose output folder:",
            bg="grey80",
            anchor="w",
            font=("Courier", 10)
        )
        prompt_label.pack(fill=tk.X)
        
        browse_frame = tk.Frame(parent, relief='flat', borderwidth=2, bg="grey80")
        browse_frame.columnconfigure(0, weight=0)
        browse_frame.columnconfigure(1, weight=1)
        browse_frame.pack(fill=tk.X)
        
        self.browse_button = tk.Button(
            browse_frame,
            text='Browse...',
            command=self._browse_folder
        )
        self.browse_button.grid(row=0, column=0, sticky="nsew")
        
        self.folder_var = tk.StringVar(value=str(self.config.output.folder_path))
        folder_label = tk.Label(
            browse_frame,
            textvariable=self.folder_var,
            width=31,
            anchor='e'
        )
        folder_label.grid(row=0, column=1, sticky="nsew")
    
    def _create_format_selection(self, parent: tk.Widget) -> None:
        """Create format selection UI."""
        format_frame = tk.Frame(parent, relief='flat', borderwidth=2, bg="grey80")
        format_frame.columnconfigure(0, weight=0)
        format_frame.columnconfigure(1, weight=1)
        format_frame.pack(fill=tk.X)
        
        format_label = tk.Label(
            format_frame,
            text="Select output format:",
            bg="grey80",
            anchor="w",
            font=("Courier", 10)
        )
        format_label.grid(row=0, column=0, sticky='w')
        
        self.format_var = tk.StringVar(value=self.config.output.file_format)
        format_dropdown = ttk.Combobox(
            format_frame,
            textvariable=self.format_var,
            values=OUTPUT_FORMATS,
            state="readonly"
        )
        format_dropdown.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        format_dropdown.bind('<<ComboboxSelected>>', self._on_format_changed)
        
        self.format_dropdown = format_dropdown
    
    def _create_watermark_settings(self, parent: tk.Widget) -> None:
        """Create watermark settings UI."""
        watermark_frame = tk.Frame(parent, relief='flat', borderwidth=2, bg="grey60", padx=5, pady=2)
        watermark_frame.pack(fill=tk.X)
        
        # Fish ID
        fish_id_frame = tk.Frame(watermark_frame, relief='flat', borderwidth=2)
        fish_id_frame.columnconfigure(0, weight=0)
        fish_id_frame.columnconfigure(1, weight=1)
        fish_id_frame.pack(fill=tk.X, pady=3)
        
        fish_id_label = tk.Label(fish_id_frame, text="Fish ID:", pady=3)
        fish_id_label.grid(row=0, column=0, sticky='w', padx=5)
        
        self.fish_id_var = tk.StringVar(value=self.config.output.fish_id)
        self.fish_id_entry = tk.Entry(fish_id_frame, textvariable=self.fish_id_var, justify='left')
        self.fish_id_entry.grid(row=0, column=1, sticky='ew', padx=5)
        self.fish_id_var.trace_add('write', self._on_fish_id_changed)
        
        # Additional text
        freetext_frame = tk.Frame(watermark_frame, relief='flat', borderwidth=2, padx=5)
        freetext_frame.pack(fill=tk.BOTH, pady=4)
        
        freetext_label = tk.Label(
            freetext_frame,
            text="Additional text for watermark:",
            anchor="w",
            font=("Courier", 10)
        )
        freetext_label.pack(fill=tk.X)
        
        self.additional_text = tk.Text(freetext_frame, height=5, width=5)
        self.additional_text.pack(fill=tk.X, pady=5)
        self.additional_text.insert('1.0', self.config.output.additional_text)
        self.additional_text.bind('<KeyRelease>', self._on_additional_text_changed)
    
    def _browse_folder(self) -> None:
        """Open folder browser dialog."""
        folder_name = filedialog.askdirectory()
        if folder_name:
            self.folder_var.set(folder_name)
            self.config.output.folder_path = Path(folder_name)
            publish_settings_changed("output", self.config.output.__dict__)
    
    def _on_format_changed(self, event: tk.Event) -> None:
        """Handle format selection change."""
        self.config.output.file_format = self.format_var.get()
        publish_settings_changed("output", self.config.output.__dict__)
    
    def _on_fish_id_changed(self, *args) -> None:
        """Handle fish ID change."""
        self.config.output.fish_id = self.fish_id_var.get()
        publish_settings_changed("output", self.config.output.__dict__)
    
    def _on_additional_text_changed(self, event: tk.Event) -> None:
        """Handle additional text change."""
        text = self.additional_text.get("1.0", 'end-1c')
        self.config.output.additional_text = text
        publish_settings_changed("output", self.config.output.__dict__)
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable output settings."""
        state = "normal" if enabled else "disabled"
        
        self.browse_button.configure(state=state)
        self.fish_id_entry.configure(state=state)
        self.additional_text.configure(state=state)
        
        dropdown_state = "readonly" if enabled else "disabled"
        self.format_dropdown.configure(state=dropdown_state)


class ProcessingControlPanel(BaseComponent):
    """Panel for processing controls and settings."""
    
    def __init__(self, parent: tk.Widget, config: AppConfig, 
                 on_train_background: Callable[[], None],
                 on_start_analysis: Callable[[], None]):
        self.on_train_background = on_train_background
        self.on_start_analysis = on_start_analysis
        super().__init__(parent, config, relief='flat', borderwidth=2, padx=2, pady=10, bg="grey80")
    
    def _setup_ui(self) -> None:
        """Setup processing control UI."""
        self.pack(fill=tk.BOTH)
        
        # Processing settings
        self._create_processing_settings()
        
        # Control buttons
        self._create_control_buttons()
    
    def _create_processing_settings(self) -> None:
        """Create processing settings."""
        settings_frame = tk.Frame(self, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey60")
        settings_frame.pack(fill=tk.X, padx=5)
        
        inner_frame = tk.Frame(settings_frame, relief='flat', borderwidth=2, padx=5)
        inner_frame.pack(fill=tk.X)
        inner_frame.columnconfigure(0, weight=0)
        inner_frame.columnconfigure(1, weight=1)
        
        # Shadow setting
        self._create_shadow_setting(inner_frame)
        
        # Number of frames
        self.frame_count_entry = create_frame_count_entry(
            inner_frame,
            on_change=self._on_frame_count_changed,
            row=1
        )
        self.frame_count_entry.set_value(self.config.processing.number_of_frames)
    
    def _create_shadow_setting(self, parent: tk.Widget) -> None:
        """Create shadow inclusion setting."""
        label = tk.Label(parent, text="Include shadows?", pady=3)
        label.grid(row=0, column=0, sticky='w', padx=5)
        
        radio_frame = tk.Frame(parent, relief='flat', borderwidth=2, padx=5)
        radio_frame.grid(row=0, column=1, sticky='ew', padx=5)
        
        self.shadow_var = tk.IntVar(value=0 if self.config.processing.include_shadows else 1)
        
        yes_radio = tk.Radiobutton(
            radio_frame,
            text="Yes",
            padx=5,
            variable=self.shadow_var,
            command=self._on_shadow_changed,
            value=0
        )
        yes_radio.grid(row=0, column=0, sticky='ew', padx=5)
        
        no_radio = tk.Radiobutton(
            radio_frame,
            text="No",
            padx=5,
            variable=self.shadow_var,
            command=self._on_shadow_changed,
            value=1
        )
        no_radio.grid(row=0, column=1, sticky='ew', padx=5)
    
    def _create_control_buttons(self) -> None:
        """Create control buttons."""
        # Background training button
        bg_frame = tk.Frame(self, relief='flat', borderwidth=2, padx=10, pady=10, bg="grey80")
        bg_frame.pack(fill=tk.BOTH)
        
        self.background_button = tk.Button(
            bg_frame,
            text='TRAIN',
            command=self.on_train_background,
            bg="#74B224",
            font=("Courier", 24),
            fg="white",
            disabledforeground="white"
        )
        self.background_button.pack(fill=tk.BOTH)
        
        # Start analysis button
        start_frame = tk.Frame(self, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey60")
        start_frame.pack(fill=tk.BOTH, padx=5)
        
        self.start_button = tk.Button(
            start_frame,
            text='PLS CALIBRATE',
            command=self.on_start_analysis,
            bg="grey50",
            font=("Courier", 24),
            fg="white",
            state="disabled",
            disabledforeground="white"
        )
        self.start_button.pack(fill=tk.BOTH, pady=5)
    
    def _on_shadow_changed(self) -> None:
        """Handle shadow setting change."""
        self.config.processing.include_shadows = (self.shadow_var.get() == 0)
        publish_settings_changed("processing", self.config.processing.__dict__)
    
    def _on_frame_count_changed(self, value: str) -> None:
        """Handle frame count change."""
        try:
            count = int(float(value)) if value else 10
            self.config.processing.number_of_frames = count
            publish_settings_changed("processing", self.config.processing.__dict__)
        except ValueError:
            pass
    
    def update_button_states(self, app_state: int) -> None:
        """Update button states based on application state."""
        if app_state == 0:  # Base state
            self.background_button.configure(text='TRAIN', bg="#74B224", state="normal")
            self.start_button.configure(bg="grey50", state="disabled", text='PLS CALIBRATE')
        elif app_state == 1:  # Background trained
            self.background_button.configure(text='RESTART', bg="#185CA8", state="normal")
            # Start button state will be updated separately based on calibration status
            # Don't override it here - let the calibration system manage it
            pass
        elif app_state == 2:  # Running analysis/training
            self.background_button.configure(text='CANCEL', bg="#185CA8", state="normal")
            # Don't change start button during training/analysis - let calibration system manage it
            pass
    
    def set_processing_settings_enabled(self, enabled: bool) -> None:
        """Enable or disable processing settings."""
        self.frame_count_entry.set_enabled(enabled)


class VideoDisplayPanel(BaseComponent):
    """Panel for displaying video feed."""
    
    def __init__(self, parent: tk.Widget, config: AppConfig):
        self.image_label: Optional[tk.Label] = None  # Initialize before super() call
        super().__init__(parent, config, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey5")
    
    def _setup_ui(self) -> None:
        """Setup video display UI."""
        self.grid(row=0, column=1, sticky="nsew")
        
        self.image_label = tk.Label(self, text="Starting camera feed...", bg="black", fg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)
    
    def update_image(self, image: Any) -> None:
        """Update displayed image."""
        if not self.image_label:
            return
            
        if image is not None:
            try:
                # Convert numpy array to PhotoImage for tkinter display
                if not hasattr(self, '_pil_cache'):
                    # Lazy import and cache on the instance to avoid per-frame import cost
                    from PIL import Image, ImageTk
                    import numpy as np
                    self._pil_cache = (Image, ImageTk, np)
                else:
                    Image, ImageTk, np = self._pil_cache
                
                if isinstance(image, np.ndarray):
                    # Convert BGR to RGB if needed
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image_rgb = image[:, :, ::-1]  # BGR to RGB
                    else:
                        image_rgb = image
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(image_rgb.astype('uint8'))
                    
                    # Resize to fit display area (maintain aspect ratio)
                    display_width = 800
                    display_height = 600
                    pil_image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage
                    photo_image = ImageTk.PhotoImage(pil_image)
                    
                    self.image_label.configure(image=photo_image, text="")
                    self.image_label.image = photo_image  # Keep a reference
                    self.image_label.update()
                else:
                    # If it's already a PhotoImage, use it directly
                    self.image_label.configure(image=image, text="")
                    self.image_label.image = image  # Keep a reference
                    self.image_label.update()
                    
            except Exception as e:
                # If conversion fails, display an error message
                self.image_label.configure(text=f"Display Error: {e}", image="")
                self.image_label.image = None
