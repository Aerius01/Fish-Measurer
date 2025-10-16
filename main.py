from __future__ import annotations

import sys
import signal
import logging
from pathlib import Path
from typing import Optional, List

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional in this mockup environment
    cv2 = None  # type: ignore
    np = None  # type: ignore

try:
    # pypylon is optional and only used if available
    from pypylon import pylon  # type: ignore
except Exception:  # pragma: no cover - not always installed
    pylon = None  # type: ignore

from PySide6 import QtCore, QtGui, QtWidgets
from logic import (
    MeasurementOrchestrator,
    MeasurementStateMachine,
    MeasurementState,
    StateTransition
)
from vision import CameraFactory

logger = logging.getLogger(__name__)


# ============================ Constants ============================

# Resource limits
MAX_FRAME_COUNT = 1000  # Maximum frames that can be captured in one session
ESTIMATED_FRAME_SIZE_MB = 5  # Estimated memory per frame (1920x1080 RGB + processing)

# ============================ UI Components ============================


class Spinner(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        # Modern, smooth rotating arc (material-style)
        self._angle_deg = 0.0
        self._thickness = 4
        self._color = QtGui.QColor(233, 234, 238)
        self._span_deg = 270  # length of arc
        self._speed_deg_per_tick = 6.0  # 60fps -> full rotation ~1s
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)  # ~60 FPS
        self._timer.timeout.connect(self._advance)
        self._timer.start()
        self.setFixedSize(48, 48)

    def _advance(self) -> None:
        self._angle_deg = (self._angle_deg + self._speed_deg_per_tick) % 360.0
        self.update()

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        return QtCore.QSize(48, 48)

    def minimumSizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        return QtCore.QSize(32, 32)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(self._thickness, self._thickness, -self._thickness, -self._thickness)
        start_angle = int(self._angle_deg * 16)
        span_angle = int(-self._span_deg * 16)
        pen = QtGui.QPen(self._color)
        pen.setWidth(self._thickness)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, start_angle, span_angle)


class VideoView(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setObjectName("VideoView")

        self._stack = QtWidgets.QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)

        # Image page
        self._image_label = QtWidgets.QLabel()
        self._image_label.setAlignment(QtCore.Qt.AlignCenter)
        self._stack.addWidget(self._image_label)

        # Searching page
        search_page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(search_page)
        # Add generous padding so spinner/text never look clipped
        vbox.setContentsMargins(24, 24, 24, 24)
        vbox.setSpacing(12)
        vbox.addStretch(1)
        self._search_label = QtWidgets.QLabel("Searching for camera")
        self._search_label.setAlignment(QtCore.Qt.AlignCenter)
        spinner = Spinner()
        spinner.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(spinner)
        row.addStretch(1)
        vbox.addWidget(self._search_label)
        vbox.addLayout(row)
        vbox.addStretch(1)
        self._stack.addWidget(search_page)

        self.show_searching()

    # Backward-compat for previous setText usage
    def setText(self, text: str) -> None:  # type: ignore[override]
        self._search_label.setText(text)
        self.show_searching()

    def show_searching(self) -> None:
        self._stack.setCurrentIndex(1)

    def update_with_frame(self, frame: "np.ndarray") -> None:
        if frame is None or np is None or cv2 is None:
            self.show_searching()
            return
        # Convert (copy) once here for display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        # Scale to label size using fast transform
        self._image_label.setPixmap(
            pix.scaled(self._image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        )
        self._stack.setCurrentIndex(0)


class Heading(QtWidgets.QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setObjectName("Heading")


def create_round_button(text: str, variant: str) -> QtWidgets.QPushButton:
    btn = QtWidgets.QPushButton(text)
    btn.setCursor(QtCore.Qt.PointingHandCursor)
    btn.setMinimumHeight(48)
    btn.setProperty("variant", variant)
    btn.setObjectName("ActionButton")
    return btn


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fish Measurer")
        self.resize(1200, 780)

        # Camera via CameraFactory (HAL abstraction)
        from vision import CameraHAL
        self.camera: Optional[CameraHAL] = None
        self.camera = CameraFactory.create_camera()
        if self.camera:
            self.camera.start_grabbing()

        # App settings
        self.output_extension = ".png"  # hardcoded as requested

        # Calibration UI state (orchestrator manages actual calibration)
        self._calibration_mode = False
        self._calibration_completed = False
        self._required_calibration_samples = 15

        # Orchestrator for all workflows (replaces MeasurerInstance and thread management)
        self.orchestrator = MeasurementOrchestrator(
            camera_manager=self.camera,
            output_folder=None,
            image_format=".png"
        )
        self.orchestrator.set_progress_callback(self._on_progress_update)

        # Use orchestrator's state machine (single source of truth)
        self.state_machine = self.orchestrator.state_machine
        self.state_machine.add_observer(self._on_state_changed)

        # State machine starts in NOCAM by default
        # Only transition to UNCALIBRATED if we have a camera
        if self.camera is not None:
            # Transition from NOCAM to UNCALIBRATED when camera is available
            self.state_machine.transition_to(MeasurementState.UNCALIBRATED, reason="camera_connected")

        # Camera detection timer (only runs when no camera)
        self._camera_detection_timer = QtCore.QTimer(self)
        self._camera_detection_timer.setInterval(2000)  # Check every 2 seconds
        self._camera_detection_timer.timeout.connect(self._attempt_camera_connection)
        if self.camera is None:
            self._camera_detection_timer.start()

        # Root layout
        wrapper = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(wrapper)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # Left column (controls)
        self.left_panel = self._build_left_panel()
        root.addWidget(self.left_panel, 0)

        # Right column (video)
        self.video_view = VideoView()
        video_card = QtWidgets.QFrame()
        video_card.setObjectName("Card")
        video_layout = QtWidgets.QVBoxLayout(video_card)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(self.video_view)
        root.addWidget(video_card, 1)

        self.setCentralWidget(wrapper)

        # Update UI for camera state (camera already started if available)
        if self.camera is None:
            self.video_view.setText("No camera detected")

        # Ensure buttons reflect current camera state on launch
        self._sync_buttons_for_camera_state()

        # UI update timer (limits UI work; still polls latest frame)
        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.setInterval(17)  # ~58-60 FPS target
        self._ui_timer.timeout.connect(self._update_from_camera)
        self._ui_timer.start()

        # Update camera name label now that camera has started (Basler name may be enriched in start())
        self._update_camera_name_label()

        self._apply_style()

    # ---- state management (now via observer pattern) ----
    def _on_state_changed(self, transition: StateTransition) -> None:
        """Observer callback for state machine transitions."""
        # Queue UI update on main thread using QTimer instead of invokeMethod
        # (invokeMethod requires @Slot decorator which we want to avoid for private methods)
        QtCore.QTimer.singleShot(0, self._sync_buttons_for_camera_state)

    # ---- path validation ----
    def _validate_output_path(self, path: str) -> bool:
        """
        Validate that the output path is within the user's home directory.

        Args:
            path: Path to validate

        Returns:
            True if path is valid and within home directory, False otherwise
        """
        if not path:
            return False

        try:
            resolved = Path(path).resolve()
            allowed_base = Path.home()

            # Check if path is within home directory
            return resolved.is_relative_to(allowed_base)
        except (ValueError, OSError, RuntimeError):
            return False

    # ---- UI builders ----
    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setObjectName("SidePanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(12)

        # Camera Settings
        layout.addWidget(Heading("CAMERA SETTINGS"))
        layout.addWidget(self._build_camera_settings_card())

        # Output Settings
        layout.addWidget(Heading("OUTPUT SETTINGS"))
        layout.addWidget(self._build_output_settings_card())

        # Bottom controls
        layout.addWidget(self._build_bottom_controls_card())
        layout.addStretch(1)
        return panel

    def _build_camera_settings_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        form = QtWidgets.QFormLayout(card)
        form.setLabelAlignment(QtCore.Qt.AlignLeft)

        # Camera name/in use
        self.camera_name_label = QtWidgets.QLabel()
        if self.camera:
            name_text = self.camera.capabilities.name
        else:
            name_text = "No camera detected"
        self.camera_name_label.setText(f"Camera in use: {name_text}")
        self.camera_name_label.setObjectName("DimLabel")
        form.addRow(self.camera_name_label)

        # Exposure (default 100 ms per your environment)
        self.exposure_edit = QtWidgets.QLineEdit()
        self.exposure_edit.setText("100")
        self.exposure_edit.setValidator(QtGui.QIntValidator(1, 100000, self))
        self.exposure_edit.textChanged.connect(lambda t: self.camera and t.isdigit() and self.camera.set_exposure_ms(float(t)))
        exposure_row = self._make_field_with_info(
            self.exposure_edit,
            title="Exposure",
            info_text=(
                "Exposure controls how long the sensor collects light.\n\n"
                "Higher values brighten the image but increase motion blur."
            ),
        )
        form.addRow("Exposure (ms):", exposure_row)

        # Gain auto
        self.gain_combo = QtWidgets.QComboBox()
        if self.camera and hasattr(self.camera, 'capabilities'):
            gain_items = self.camera.capabilities.gain_auto_options
        else:
            gain_items = ["Off", "Once", "Continuous"]
        self.gain_combo.addItems(gain_items)
        self.gain_combo.currentTextChanged.connect(lambda t: self.camera and self.camera.set_gain_auto(t))
        gain_row = self._make_field_with_info(
            self.gain_combo,
            title="Gain",
            info_text=(
                "Controls automatic gain.\n\n"
                "Off: fixed gain. Once: auto-adjusts then locks. Continuous: keeps adjusting."
            ),
        )
        form.addRow("Gain Setting:", gain_row)

        # White balance auto
        self.wb_combo = QtWidgets.QComboBox()
        if self.camera and hasattr(self.camera, 'capabilities'):
            wb_items = self.camera.capabilities.white_balance_auto_options
        else:
            wb_items = ["Off", "Once", "Continuous"]
        self.wb_combo.addItems(wb_items)
        self.wb_combo.currentTextChanged.connect(lambda t: self.camera and self.camera.set_white_balance_auto(t))
        wb_row = self._make_field_with_info(
            self.wb_combo,
            title="White Balance",
            info_text=(
                "Balances color temperature.\n\n"
                "Off: fixed. Once: auto-calibrates then locks. Continuous: keeps adjusting colors."
            ),
        )
        form.addRow("White Balance:", wb_row)

        # Frame rate
        self.fps_edit = QtWidgets.QLineEdit()
        self.fps_edit.setText("30")
        self.fps_edit.setValidator(QtGui.QIntValidator(1, 240, self))
        self.fps_edit.textChanged.connect(lambda t: self.camera and t.isdigit() and self.camera.set_framerate(float(t)))
        fps_row = self._make_field_with_info(
            self.fps_edit,
            title="Framerate",
            info_text=(
                "Limits capture rate.\n\n"
                "Higher FPS reduces exposure time and may reduce image brightness."
            ),
        )
        form.addRow("Framerate (fps):", fps_row)

        return card

    def _make_field_with_info(self, field: QtWidgets.QWidget, title: str, info_text: str) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Ensure field expands while the button stays compact on the right
        layout.addWidget(field, 1)

        btn = QtWidgets.QToolButton()
        btn.setObjectName("InfoButton")
        btn.setToolTip(f"More about {title}")
        btn.setText("?")
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setAutoRaise(False)
        btn.setFixedSize(22, 22)
        btn.clicked.connect(lambda: self._show_info_dialog(title, info_text))
        layout.addWidget(btn, 0)

        return container

    def _show_info_dialog(self, title: str, text: str) -> None:
        try:
            QtWidgets.QMessageBox.information(self, title, text)
        except Exception:
            # Fallback: update status area if message box fails
            if hasattr(self, "camera_name_label"):
                self.camera_name_label.setText(f"{title}: {text}")

    def _build_output_settings_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        grid = QtWidgets.QGridLayout(card)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # Row 0: folder chooser
        folder_label = QtWidgets.QLabel("Choose output folder:")
        self.folder_edit = QtWidgets.QLineEdit()
        self.browse_btn = QtWidgets.QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._choose_folder)
        grid.addWidget(folder_label, 0, 0, 1, 2)
        grid.addWidget(self.browse_btn, 1, 0)
        grid.addWidget(self.folder_edit, 1, 1)

        # Row 2: fish id
        fish_label = QtWidgets.QLabel("Fish ID:")
        self.fish_edit = QtWidgets.QLineEdit()
        grid.addWidget(fish_label, 2, 0)
        grid.addWidget(self.fish_edit, 2, 1)

        # Row 3-4: watermark
        wm_label = QtWidgets.QLabel("Additional text for watermark:")
        self.wm_edit = QtWidgets.QPlainTextEdit()
        self.wm_edit.setFixedHeight(80)
        grid.addWidget(wm_label, 3, 0, 1, 2)
        grid.addWidget(self.wm_edit, 4, 0, 1, 2)

        return card

    def _build_bottom_controls_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        vbox = QtWidgets.QVBoxLayout(card)

        # Include shadows
        shadows_row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Include shadows?")
        self.shadows_yes_radio = QtWidgets.QRadioButton("Yes")
        self.shadows_no_radio = QtWidgets.QRadioButton("No")
        self.shadows_yes_radio.setChecked(True)
        self.shadows_button_group = QtWidgets.QButtonGroup(card)
        self.shadows_button_group.addButton(self.shadows_yes_radio)
        self.shadows_button_group.addButton(self.shadows_no_radio)
        shadows_row.addWidget(label)
        shadows_row.addStretch(1)
        shadows_row.addWidget(self.shadows_yes_radio)
        shadows_row.addWidget(self.shadows_no_radio)
        vbox.addLayout(shadows_row)

        # Number of frames
        frames_row = QtWidgets.QHBoxLayout()
        frames_label = QtWidgets.QLabel("Number of Frames:")
        self.frames_edit = QtWidgets.QLineEdit()
        self.frames_edit.setText("10")
        self.frames_edit.setValidator(QtGui.QIntValidator(1, 10000, self))
        frames_row.addWidget(frames_label)
        frames_row.addStretch(1)
        frames_row.addWidget(self.frames_edit)
        vbox.addLayout(frames_row)

        # Buttons
        vbox.addSpacing(8)
        self.calibrate_btn = create_round_button("CALIBRATE", variant="train")
        self.start_btn = create_round_button("START", variant="calib")
        self.start_btn.setEnabled(False)
        self.calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        self.start_btn.clicked.connect(self._on_start_clicked)
        vbox.addWidget(self.calibrate_btn)
        vbox.addWidget(self.start_btn)

        return card

    # ---- interactions ----
    def _choose_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            if self._validate_output_path(path):
                self.folder_edit.setText(path)
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Path",
                    f"The selected path must be within your home directory:\n{Path.home()}\n\n"
                    f"Selected path: {path}"
                )
                self.folder_edit.clear()

    def _update_from_camera(self) -> None:
        if self.camera is None:
            self.video_view.setText("Searching for camera...")
            return

        # Check if camera disconnected
        if not self.camera.is_connected:
            self.camera = None
            self.orchestrator.camera_manager = None
            self.state_machine.transition_to(MeasurementState.NOCAM, reason="camera_disconnected")
            self._calibration_mode = False
            self.video_view.setText("Camera disconnected - searching...")
            self._sync_buttons_for_camera_state()
            # Restart camera detection
            if not self._camera_detection_timer.isActive():
                self._camera_detection_timer.start()
            return

        frame = self.camera.get_latest_frame()
        if frame is not None:
            # If in calibration mode, update calibration with orchestrator
            if self._calibration_mode:
                try:
                    # Update calibration through orchestrator
                    markers_detected, marker_count = self.orchestrator.update_calibration(frame)

                    # Draw ArUco markers on display frame (thicker borders during calibration)
                    markers = self.orchestrator.aruco_detector.detect_markers(frame)
                    display_frame = self.orchestrator.aruco_detector.draw_markers(frame, markers, thickness=12) if markers else frame

                    # Check if calibration auto-completed
                    if self.orchestrator.state_machine.current_state == MeasurementState.CALIBRATED:
                        self._calibration_completed = True
                        self._stop_calibration(manual=False)

                    self.video_view.update_with_frame(display_frame)
                    return
                except Exception:
                    # On detection failure, fall back to normal display
                    pass

            # If in trained or processing state, show binarized feed (largest foreground)
            current_state = self.state_machine.current_state
            if current_state in [MeasurementState.TRAINED, MeasurementState.PROCESSING]:
                try:
                    mask = self.orchestrator.create_binarized_mask(frame)
                    if mask is not None:
                        # Convert single-channel mask to 3-channel for display
                        display_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        self.video_view.update_with_frame(display_frame)
                        return
                except Exception:
                    # Fall back to raw frame on error
                    pass

            # Default: show raw frame
            self.video_view.update_with_frame(frame)
        else:
            self.video_view.setText("Searching for camera")

    def _update_camera_name_label(self) -> None:
        if hasattr(self, "camera_name_label"):
            if self.camera is not None:
                self.camera_name_label.setText(f"Camera in use: {self.camera.capabilities.name}")
            else:
                self.camera_name_label.setText("Camera in use: No camera detected")

    def _on_camera_connected(self) -> None:
        # Refresh UI after camera connects
        self._update_camera_name_label()

        # Update orchestrator with connected camera
        if self.camera is not None:
            self.orchestrator.camera_manager = self.camera

        # Transition to UNCALIBRATED state
        if self.state_machine.current_state == MeasurementState.NOCAM:
            self.state_machine.transition_to(MeasurementState.UNCALIBRATED, reason="camera_detected")

        # Unlock settings now that camera is available
        self._lock_settings(False)

        # Enable CALIBRATE button and restore label once camera is connected
        self._sync_buttons_for_camera_state()

    def _attempt_camera_connection(self) -> None:
        """Periodically attempt to connect to a camera."""
        if self.camera is not None:
            # Already have a camera, stop detection timer
            self._camera_detection_timer.stop()
            return

        # Try to create and connect to camera
        try:
            camera = CameraFactory.create_camera()
            if camera and camera.is_connected:
                camera.start_grabbing()
                self.camera = camera
                self._on_camera_connected()
                self._camera_detection_timer.stop()
                logger.info("Camera detected and connected")
        except Exception as e:
            # Silently continue detection loop
            logger.debug(f"Camera detection attempt failed: {e}")

    def _on_progress_update(self, current: int, total: int, message: str) -> None:
        """Progress callback from orchestrator (called from worker thread)."""
        QtCore.QMetaObject.invokeMethod(
            self,
            "_update_progress_display",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, current),
            QtCore.Q_ARG(int, total),
            QtCore.Q_ARG(str, message)
        )

    @QtCore.Slot(int, int, str)
    def _update_progress_display(self, current: int, total: int, message: str) -> None:
        """Update progress display on main thread."""
        if hasattr(self, 'start_btn'):
            try:
                # Only show progress while actively TRAINING or PROCESSING
                current_state = self.state_machine.current_state
                if current_state in (MeasurementState.TRAINING, MeasurementState.PROCESSING):
                    if isinstance(message, str) and message.strip():
                        self.start_btn.setText(message.upper())
                    else:
                        self.start_btn.setText(f"PROCESSED: {current}/{total}")
                else:
                    # Reset label when not busy
                    self.start_btn.setText("START")
            except Exception:
                self.start_btn.setText("START")

        # Update control options from capabilities
        try:
            if hasattr(self, "gain_combo") and self.camera is not None:
                self.gain_combo.blockSignals(True)
                self.gain_combo.clear()
                items = self.camera.capabilities.gain_auto_options or ["Off", "Once", "Continuous"]
                self.gain_combo.addItems(items)
                self.gain_combo.blockSignals(False)
            if hasattr(self, "wb_combo") and self.camera is not None:
                self.wb_combo.blockSignals(True)
                self.wb_combo.clear()
                items = self.camera.capabilities.white_balance_auto_options or ["Off", "Once", "Continuous"]
                self.wb_combo.addItems(items)
                self.wb_combo.blockSignals(False)
        except Exception:
            pass

    def _on_calibrate_clicked(self) -> None:
        """Handle CALIBRATE/TRAIN/RETRAIN/STOP button clicks based on current state."""
        if self.camera is None or not hasattr(self, "calibrate_btn"):
            return

        current_state = self.state_machine.current_state

        # UNCALIBRATED: Start calibration
        if current_state == MeasurementState.UNCALIBRATED:
            self._start_calibration()

        # CALIBRATING: Stop calibration
        elif current_state == MeasurementState.CALIBRATING or self._calibration_mode:
            self._stop_calibration(manual=True)

        # CALIBRATED: Start training
        elif current_state == MeasurementState.CALIBRATED:
            self._start_training()

        # TRAINING: Stop training
        elif current_state == MeasurementState.TRAINING:
            self._stop_training(manual=True)

        # TRAINED: Retrain - reset to calibrated state
        elif current_state == MeasurementState.TRAINED:
            self._reset_training()

    def _reset_training(self) -> None:
        """Reset training and return to Calibrated state."""
        # Orchestrator handles background reset internally
        self.state_machine.transition_to(MeasurementState.CALIBRATED)
        self.calibrate_btn.setText("TRAIN")
        self.start_btn.setEnabled(False)

    def _on_start_clicked(self) -> None:
        """Handle START/STOP button clicks based on current state."""
        current_state = self.state_machine.current_state

        # TRAINED: Start processing
        if current_state == MeasurementState.TRAINED:
            self._start_processing()

        # PROCESSING: Stop processing
        elif current_state == MeasurementState.PROCESSING:
            self._stop_processing(manual=True)

    def _start_processing(self) -> None:
        """Start image capture and processing."""
        # Validate frame count
        try:
            frame_count = int(self.frames_edit.text()) if self.frames_edit.text().isdigit() else 10
            if frame_count < 1:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Frame Count",
                    "Frame count must be at least 1."
                )
                return
            if frame_count > MAX_FRAME_COUNT:
                estimated_memory = frame_count * ESTIMATED_FRAME_SIZE_MB
                QtWidgets.QMessageBox.warning(
                    self,
                    "Excessive Frame Count",
                    f"Frame count exceeds maximum of {MAX_FRAME_COUNT}.\n\n"
                    f"Requested: {frame_count} frames\n"
                    f"Estimated memory: ~{estimated_memory}MB\n\n"
                    "Please reduce the frame count."
                )
                return
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Frame Count",
                "Please enter a valid number for frame count."
            )
            return

        # Validate output path before starting
        output_path = self.folder_edit.text()
        if not output_path or not output_path.strip():
            QtWidgets.QMessageBox.warning(
                self,
                "Select Output Folder",
                "Please choose an output folder before starting processing.\n\n"
                "Use the Browse button under Output Settings."
            )
            return
        if output_path and not self._validate_output_path(output_path):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Output Path",
                f"The output path must be within your home directory:\n{Path.home()}\n\n"
                f"Current path: {output_path}\n\n"
                "Please select a valid output folder."
            )
            return

        # Configure orchestrator with GUI settings
        self.orchestrator.set_output_folder(output_path if output_path else None)
        self.orchestrator.set_image_format(self.output_extension)
        self.orchestrator.set_fish_id(self.fish_edit.text())
        self.orchestrator.set_additional_text(self.wm_edit.toPlainText())

        # Start processing through orchestrator
        if self.orchestrator.start_processing(num_frames=frame_count, async_mode=True):
            # Switch button to STOP (red)
            self.start_btn.setText("STOP")
            self._set_button_variant(self.start_btn, "stop")
            self.calibrate_btn.setEnabled(False)  # Disable RETRAIN during processing

            # Lock settings during processing
            self._lock_settings(True)

            # Monitor processing progress
            self._monitor_processing()
        else:
            # Show user-facing message if orchestrator refused to start
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot Start Processing",
                "Processing could not be started.\n\n"
                "Ensure an output folder is selected and try again."
            )

    def _stop_processing(self, manual: bool) -> None:
        """Stop processing."""
        if manual:
            # Request cancellation
            self.orchestrator.cancel_processing()

        # Unlock settings
        self._lock_settings(False)

        # Button state will be updated by state machine observer

    def _monitor_processing(self) -> None:
        """Monitor processing progress and update UI."""
        # Check if processing is complete
        if self.orchestrator.is_processing_complete():
            # Processing finished, update UI
            self._lock_settings(False)
            self._sync_buttons_for_camera_state()
            return

        # Check again soon
        QtCore.QTimer.singleShot(100, self._monitor_processing)

    def _start_calibration(self) -> None:
        # Start calibration through orchestrator
        if self.orchestrator.start_calibration():
            self._calibration_mode = True
            self._calibration_completed = False

            # Switch button to STOP (red)
            self.calibrate_btn.setText("STOP")
            self._set_button_variant(self.calibrate_btn, "stop")
            self.calibrate_btn.setEnabled(True)

            # Lock settings during calibration
            self._lock_settings(True)

    def _stop_calibration(self, manual: bool) -> None:
        # End calibration processing
        self._calibration_mode = False

        # Unlock settings
        self._lock_settings(False)

        # Restore button to state before starting, or to TRAIN if completed
        if self._calibration_completed and not manual:
            # Orchestrator already transitioned to CALIBRATED; just update UI
            self.calibrate_btn.setText("TRAIN")
        else:
            # Manual stop returns to UNCALIBRATED state
            self.state_machine.transition_to(MeasurementState.UNCALIBRATED)
            self.calibrate_btn.setText("CALIBRATE")
        self._set_button_variant(self.calibrate_btn, "train")

    def _start_training(self) -> None:
        """Start background training process."""
        # Start training through orchestrator
        if self.orchestrator.start_background_training(duration_seconds=2, async_mode=True):
            # Switch button to STOP (red)
            self.calibrate_btn.setText("STOP")
            self._set_button_variant(self.calibrate_btn, "stop")
            self.calibrate_btn.setEnabled(True)

            # Lock settings during training
            self._lock_settings(True)

            # Monitor training progress
            self._monitor_training()

    def _stop_training(self, manual: bool) -> None:
        """Stop background training process."""
        if manual:
            # Request cancellation
            self.orchestrator.cancel_training()

        # Unlock settings
        self._lock_settings(False)

        # Proactively reset start button label; observer will finalize state shortly
        if hasattr(self, 'start_btn'):
            self.start_btn.setText("START")
            self._set_button_variant(self.start_btn, "calib")
        # Button state will be finalized by state machine observer

    def _monitor_training(self) -> None:
        """Monitor training progress and update UI."""
        # Check if training is complete
        if self.orchestrator.is_training_complete():
            # Training finished, update UI
            self._lock_settings(False)
            # Ensure buttons reflect final state (TRAINED or ERROR)
            self._sync_buttons_for_camera_state()
            return

        # Check again soon
        QtCore.QTimer.singleShot(100, self._monitor_training)

    def _lock_settings(self, locked: bool) -> None:
        """Lock or unlock all settings during productive operations."""
        enabled = not locked

        # Camera settings
        if hasattr(self, "exposure_edit"):
            self.exposure_edit.setEnabled(enabled)
        if hasattr(self, "gain_combo"):
            self.gain_combo.setEnabled(enabled)
        if hasattr(self, "wb_combo"):
            self.wb_combo.setEnabled(enabled)
        if hasattr(self, "fps_edit"):
            self.fps_edit.setEnabled(enabled)

        # Output settings
        if hasattr(self, "browse_btn"):
            self.browse_btn.setEnabled(enabled)
        if hasattr(self, "folder_edit"):
            self.folder_edit.setEnabled(enabled)
        if hasattr(self, "fish_edit"):
            self.fish_edit.setEnabled(enabled)
        if hasattr(self, "wm_edit"):
            self.wm_edit.setEnabled(enabled)

        # Bottom controls
        if hasattr(self, "shadows_yes_radio"):
            self.shadows_yes_radio.setEnabled(enabled)
        if hasattr(self, "shadows_no_radio"):
            self.shadows_no_radio.setEnabled(enabled)
        if hasattr(self, "frames_edit"):
            self.frames_edit.setEnabled(enabled)

    def _set_button_variant(self, btn: QtWidgets.QPushButton, variant: str) -> None:
        try:
            btn.setProperty("variant", variant)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()
        except Exception:
            pass

    def _sync_buttons_for_camera_state(self) -> None:
        """Update button states based on current app state."""
        if not hasattr(self, "calibrate_btn") or not hasattr(self, "start_btn"):
            return

        current_state = self.state_machine.current_state

        # NOCAM: disable everything, show searching message
        if current_state == MeasurementState.NOCAM or self.camera is None:
            self.calibrate_btn.setEnabled(False)
            self.calibrate_btn.setText("SEARCHING FOR CAMERA...")
            self._set_button_variant(self.calibrate_btn, "train")
            self.start_btn.setEnabled(False)
            self.start_btn.setText("START")
            self._set_button_variant(self.start_btn, "calib")
            # Disable all settings when no camera
            self._lock_settings(True)
            return

        # UNCALIBRATED
        if current_state == MeasurementState.UNCALIBRATED:
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setText("CALIBRATE")
            self._set_button_variant(self.calibrate_btn, "train")
            self.start_btn.setEnabled(False)
            self.start_btn.setText("START")
            self._set_button_variant(self.start_btn, "calib")

        # CALIBRATING
        elif current_state == MeasurementState.CALIBRATING:
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setText("STOP")
            self._set_button_variant(self.calibrate_btn, "stop")
            self.start_btn.setEnabled(False)
            self.start_btn.setText("START")

        # CALIBRATED
        elif current_state == MeasurementState.CALIBRATED:
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setText("TRAIN")
            self._set_button_variant(self.calibrate_btn, "train")
            self.start_btn.setEnabled(False)
            self.start_btn.setText("START")

        # TRAINING
        elif current_state == MeasurementState.TRAINING:
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setText("STOP")
            self._set_button_variant(self.calibrate_btn, "stop")
            self.start_btn.setEnabled(False)
            self.start_btn.setText("START")

        # TRAINED
        elif current_state == MeasurementState.TRAINED:
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setText("RETRAIN")
            self._set_button_variant(self.calibrate_btn, "retrain")
            self.start_btn.setEnabled(True)
            self.start_btn.setText("START")
            self._set_button_variant(self.start_btn, "calib")

        # PROCESSING
        elif current_state == MeasurementState.PROCESSING:
            self.calibrate_btn.setEnabled(False)
            self.calibrate_btn.setText("RETRAIN")
            self._set_button_variant(self.calibrate_btn, "retrain")
            self.start_btn.setEnabled(True)
            # Text will be updated by _monitor_processing()
            self._set_button_variant(self.start_btn, "stop")

        # ERROR
        elif current_state == MeasurementState.ERROR:
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setText("CALIBRATE")
            self._set_button_variant(self.calibrate_btn, "train")
            self.start_btn.setEnabled(False)
            self.start_btn.setText("START")
            self._set_button_variant(self.start_btn, "calib")

    # ---- styling ----
    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            * { font-family: Inter, Segoe UI, Ubuntu, Arial; font-size: 13px; }
            QMainWindow { background: #1e1f24; }
            #SidePanel { min-width: 360px; }
            #Card { background: #2a2c33; border-radius: 10px; padding: 12px; }
            #VideoView { background: #0f1116; border-radius: 10px; }
            #Heading { color: #d6d8de; font-size: 12px; letter-spacing: 2px; font-weight: 700; }
            #DimLabel { color: #9aa0a6; }
            QLabel { color: #e9eaee; }
            QLineEdit, QPlainTextEdit, QSpinBox, QComboBox { 
                background: #1a1b20; 
                color: #e9eaee; 
                border: 1px solid #3a3c45; 
                border-radius: 8px; 
                padding: 6px 8px; 
            }
            /* Dark popup and comfortable item spacing */
            QComboBox QAbstractItemView { background: #2a2c33; border: 1px solid #3a3c45; }
            QComboBox QAbstractItemView::item { padding: 8px 10px; }
            QScrollBar:vertical { width: 12px; }
            QScrollBar::handle:vertical { background: #3a3f4b; min-height: 20px; border-radius: 6px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QPlainTextEdit { padding: 8px; }
            QPushButton { 
                background: #343844; 
                color: #e9eaee; 
                border: 1px solid #434857; 
                border-radius: 10px; 
                padding: 8px 12px; 
            }
            /* Removed unsupported CSS 'filter' property to avoid runtime warnings */
            /* Action buttons with clear borders, distinct colors, and pressed states */
            QPushButton#ActionButton { font-weight: 600; }
            QPushButton#ActionButton[variant="train"] {
                background: #2e7d32; /* green */
                border: 2px solid #3fa64a;
                color: #ffffff;
            }
            QPushButton#ActionButton[variant="train"]:hover { background: #338a3e; }
            QPushButton#ActionButton[variant="train"]:pressed { background: #27672c; }
            QPushButton#ActionButton[variant="train"]:disabled {
                background: #3a3f4b; /* slate */
                border: 2px solid #8a8f9c;
                color: #ffffff;
            }

            /* STOP variant (red) */
            QPushButton#ActionButton[variant="stop"] {
                background: #c62828; /* red */
                border: 2px solid #ef5350;
                color: #ffffff;
            }
            QPushButton#ActionButton[variant="stop"]:hover { background: #d32f2f; }
            QPushButton#ActionButton[variant="stop"]:pressed { background: #ab2424; }

            /* RETRAIN variant (blue) */
            QPushButton#ActionButton[variant="retrain"] {
                background: #1976d2; /* blue */
                border: 2px solid #42a5f5;
                color: #ffffff;
            }
            QPushButton#ActionButton[variant="retrain"]:hover { background: #1e88e5; }
            QPushButton#ActionButton[variant="retrain"]:pressed { background: #1565c0; }
            QPushButton#ActionButton[variant="retrain"]:disabled {
                background: #3a3f4b; /* slate */
                border: 2px solid #8a8f9c;
                color: #ffffff;
            }

            QPushButton#ActionButton[variant="calib"]:enabled {
                background: #2e7d32; /* green */
                border: 2px solid #3fa64a;
                color: #ffffff;
            }
            QPushButton#ActionButton[variant="calib"]:enabled:hover { background: #338a3e; }
            QPushButton#ActionButton[variant="calib"]:enabled:pressed { background: #27672c; }

            QPushButton#ActionButton[variant="calib"]:disabled {
                background: #3a3f4b; /* slate */
                border: 2px solid #8a8f9c;
                color: #ffffff;
            }
            /* Compact circular info buttons with strong contrast */
            QToolButton#InfoButton {
                background: #1a1b20;
                color: #ffffff;
                border: 1px solid #8a8f9c;
                border-radius: 11px; /* circle for 22x22 */
                min-width: 22px;
                min-height: 22px;
                font-weight: 700;
                padding: 0; /* center the letter */
            }
            QToolButton#InfoButton:hover {
                background: #2a2c33;
                border-color: #b4bac7;
            }
            QToolButton#InfoButton:pressed {
                background: #0f1116;
            }
            QToolButton#InfoButton:disabled {
                background: #2a2c33;
                color: #9aa0a6;
                border-color: #3a3c45;
            }
            /* High-contrast message boxes */
            QMessageBox {
                background: #2a2c33;
                color: #e9eaee;
                border: 1px solid #3a3c45;
            }
            QMessageBox QLabel { color: #e9eaee; }
            QMessageBox QPushButton {
                background: #343844;
                color: #e9eaee;
                border: 1px solid #434857;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QMessageBox QPushButton:hover { background: #3a4050; }
            QMessageBox QPushButton:pressed { background: #2b2f3a; }
            QGroupBox::title { subcontrol-origin: margin; }
            QRadioButton, QLabel { padding: 2px; }
        """
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            # Cleanup orchestrator (stops any ongoing operations and threads)
            self.orchestrator.cleanup()

            # Stop camera
            if self.camera is not None:
                self.camera.stop_grabbing()
        finally:
            super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    # Ensure Ctrl+C quits the app gracefully without leaving threads running
    try:
        signal.signal(signal.SIGINT, lambda sig, frame: app.quit())
        # Heartbeat timer keeps Python signal handling responsive
        sig_timer = QtCore.QTimer()
        sig_timer.setInterval(150)
        sig_timer.timeout.connect(lambda: None)
        sig_timer.start()
        # Keep a reference on the app object
        setattr(app, "_sigint_timer", sig_timer)
    except Exception:
        pass

    window = MainWindow()
    window.show()
    try:
        return app.exec()
    except KeyboardInterrupt:
        # Fallback: if SIGINT propagated, exit cleanly
        return 0
    finally:
        try:
            window.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())


