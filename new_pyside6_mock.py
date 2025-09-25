from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
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


# ============================ Camera Backends ============================


@dataclass
class CameraCapabilities:
    name: str
    supports_auto_gain: bool = False
    supports_auto_white_balance: bool = False
    supports_framerate: bool = False
    supports_exposure_ms: bool = True
    gain_auto_options: Optional[List[str]] = None
    white_balance_auto_options: Optional[List[str]] = None


class CameraBackend(QtCore.QObject):
    frame_ready = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._last_frame: Optional["np.ndarray"] = None
        self._running = False
        self.capabilities = CameraCapabilities(name="Unknown Camera")

    # ---- lifecycle ----
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    # ---- data ----
    def get_latest_frame(self) -> Optional["np.ndarray"]:
        # Return the latest frame reference (no copy) for speed. Publisher replaces
        # the reference atomically under the lock, so readers see stable arrays.
        with self._lock:
            return self._last_frame

    # ---- controls (no-ops by default) ----
    def set_exposure_ms(self, value_ms: float) -> None:  # noqa: ARG002
        pass

    def set_gain_auto(self, mode: str) -> None:  # noqa: ARG002
        pass

    def set_white_balance_auto(self, mode: str) -> None:  # noqa: ARG002
        pass

    def set_framerate(self, fps: float) -> None:  # noqa: ARG002
        pass

    # ---- helpers for subclasses ----
    def _publish(self, frame: "np.ndarray") -> None:
        with self._lock:
            self._last_frame = frame
        self.frame_ready.emit()


class OpenCVCameraBackend(CameraBackend):
    def __init__(self, index: int = 0) -> None:
        super().__init__()
        self._index = index
        self._capture: Optional["cv2.VideoCapture"] = None
        self._thread: Optional[threading.Thread] = None
        self._target_delay_s = 1.0 / 120.0  # cap capture loop to 120 fps max
        self.capabilities = CameraCapabilities(
            name=f"OpenCV Camera #{index}",
            supports_auto_gain=False,
            supports_auto_white_balance=False,
            supports_framerate=True,
        )

    def start(self) -> None:
        if cv2 is None:
            return
        self._capture = cv2.VideoCapture(self._index)
        # Try setting a reasonable default resolution and fps if supported
        try:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._capture.set(cv2.CAP_PROP_FPS, 60)
        except Exception:
            pass
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def set_exposure_ms(self, value_ms: float) -> None:
        if cv2 is None or self._capture is None:
            return
        # Some drivers expect negative values to enable manual exposure; best-effort only
        self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        self._capture.set(cv2.CAP_PROP_EXPOSURE, float(value_ms) / 1000.0)

    def set_framerate(self, fps: float) -> None:
        if cv2 is None or self._capture is None:
            return
        self._capture.set(cv2.CAP_PROP_FPS, float(fps))

    def _loop(self) -> None:
        assert cv2 is not None
        assert np is not None
        while self._running and self._capture is not None:
            start = time.time()
            ok, frame = self._capture.read()
            if ok and frame is not None:
                self._publish(frame)
            else:
                time.sleep(0.01)
            # Avoid busy loop pegging a CPU core
            elapsed = time.time() - start
            if elapsed < self._target_delay_s:
                time.sleep(self._target_delay_s - elapsed)


class BaslerCameraBackend(CameraBackend):
    def __init__(self, device: "pylon.PylonDevice" | None = None) -> None:  # type: ignore[name-defined]
        super().__init__()
        self._camera: Optional["pylon.InstantCamera"] = None  # type: ignore[name-defined]
        self._converter: Optional["pylon.ImageFormatConverter"] = None  # type: ignore[name-defined]
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.capabilities = CameraCapabilities(
            name="Basler Camera",
            supports_auto_gain=True,
            supports_auto_white_balance=True,
            supports_framerate=True,
        )
        self._device = device

    def start(self) -> None:
        if pylon is None:
            return
        try:
            if self._device is None:
                tl_factory = pylon.TlFactory.GetInstance()
                devices = tl_factory.EnumerateDevices()
                if not devices:
                    return
                self._device = devices[0]
            self._camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self._device))
            self._camera.Open()
            # Enrich camera name with model and serial if available after open
            try:
                dinfo = self._camera.GetDeviceInfo()
                model = dinfo.GetModelName() if hasattr(dinfo, "GetModelName") else "Basler"
                serial = dinfo.GetSerialNumber() if hasattr(dinfo, "GetSerialNumber") else "?"
                self.capabilities.name = f"Basler {model} (S/N {serial})"
            except Exception:
                pass
            # Configure fast pixel/output conversion
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            # Minimal camera configuration for smooth preview
            try:
                # Free-run
                if hasattr(self._camera, "TriggerMode"):
                    self._camera.TriggerMode.SetValue("Off")

                # Pixel format preferred for fast conversion
                if hasattr(self._camera, "PixelFormat"):
                    symbols = str(self._camera.PixelFormat.Symbols)
                    if "BGR8" in symbols:
                        self._camera.PixelFormat.SetValue("BGR8")
                    elif "RGB8" in symbols:
                        self._camera.PixelFormat.SetValue("RGB8")

            except Exception:
                pass

            # Start grabbing after configuration
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            # Default reasonable configuration
            self._populate_auto_options()
            self.set_gain_auto("Once")
            self.set_white_balance_auto("Off")
            # Set sensible default exposure for smooth preview
            try:
                self.set_exposure_ms(100.0)
            except Exception:
                pass
        except Exception:
            self._running = False
            try:
                if self._camera and self._camera.IsOpen():
                    self._camera.Close()
            finally:
                self._camera = None

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        try:
            if self._camera is not None:
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
                if self._camera.IsOpen():
                    self._camera.Close()
        finally:
            self._camera = None

    def set_exposure_ms(self, value_ms: float) -> None:
        if pylon is None or self._camera is None or not self._camera.IsOpen():
            return
        try:
            # Basler uses microseconds
            us_value = float(value_ms) * 1000.0
            if hasattr(self._camera, "ExposureAuto"):
                self._camera.ExposureAuto.SetValue("Off")
            if hasattr(self._camera, "ExposureTime"):
                self._camera.ExposureTime.SetValue(us_value)
        except Exception:
            pass

    def set_gain_auto(self, mode: str) -> None:
        if pylon is None or self._camera is None or not self._camera.IsOpen():
            return
        try:
            if hasattr(self._camera, "GainAuto"):
                allowed = {"Off": "Off", "Once": "Once", "Continuous": "Continuous"}
                self._camera.GainAuto.SetValue(allowed.get(mode, "Off"))
        except Exception:
            pass

    def set_white_balance_auto(self, mode: str) -> None:
        if pylon is None or self._camera is None or not self._camera.IsOpen():
            return
        try:
            if hasattr(self._camera, "BalanceWhiteAuto"):
                allowed = {"Off": "Off", "Once": "Once", "Continuous": "Continuous"}
                self._camera.BalanceWhiteAuto.SetValue(allowed.get(mode, "Off"))
        except Exception:
            pass

    def set_framerate(self, fps: float) -> None:
        if pylon is None or self._camera is None or not self._camera.IsOpen():
            return
        try:
            if hasattr(self._camera, "AcquisitionFrameRateEnable"):
                self._camera.AcquisitionFrameRateEnable.SetValue(True)
            if hasattr(self._camera, "AcquisitionFrameRate"):
                self._camera.AcquisitionFrameRate.SetValue(float(fps))
        except Exception:
            pass

    def _populate_auto_options(self) -> None:
        # Query supported auto modes and store them in capabilities
        try:
            gain_opts = None
            wb_opts = None
            if self._camera is not None and self._camera.IsOpen():
                if hasattr(self._camera, "GainAuto"):
                    try:
                        sym = [str(s) for s in list(self._camera.GainAuto.Symbols)]
                        # Normalize common names
                        mapping = {"Off": "Off", "Once": "Once", "Continuous": "Continuous"}
                        gain_opts = [mapping.get(s, s) for s in sym]
                    except Exception:
                        pass
                if hasattr(self._camera, "BalanceWhiteAuto"):
                    try:
                        sym = [str(s) for s in list(self._camera.BalanceWhiteAuto.Symbols)]
                        mapping = {"Off": "Off", "Once": "Once", "Continuous": "Continuous"}
                        wb_opts = [mapping.get(s, s) for s in sym]
                    except Exception:
                        pass
            self.capabilities.gain_auto_options = gain_opts or ["Off", "Once", "Continuous"]
            self.capabilities.white_balance_auto_options = wb_opts or ["Off", "Once", "Continuous"]
        except Exception:
            self.capabilities.gain_auto_options = ["Off", "Once", "Continuous"]
            self.capabilities.white_balance_auto_options = ["Off", "Once", "Continuous"]

    def _loop(self) -> None:
        assert np is not None
        if pylon is None or self._camera is None or self._converter is None:
            return
        while self._running and self._camera.IsGrabbing():
            try:
                grab = self._camera.RetrieveResult(10, pylon.TimeoutHandling_ThrowException)
                if grab.GrabSucceeded():
                    # Convert directly to OpenCV-compatible BGR8
                    image = self._converter.Convert(grab)
                    frame = image.GetArray()
                    self._publish(frame)
                grab.Release()
            except Exception:
                time.sleep(0.01)


# ============================ UI Components ============================


class VideoView(QtWidgets.QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText("Starting camera feed…")
        self.setMinimumSize(640, 360)
        self.setObjectName("VideoView")

    def update_with_frame(self, frame: "np.ndarray") -> None:
        if frame is None or np is None or cv2 is None:
            return
        # Convert (copy) once here for display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        # Scale the latest frame every time (fast transform keeps cost low)
        self.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))


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

        # Camera backend selection
        self.camera: Optional[CameraBackend] = None
        self._select_camera_backend()

        # App settings
        self.output_extension = ".png"  # hardcoded as requested

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

        # Wire camera feed to view
        if self.camera is not None:
            self.camera.start()

        # UI update timer (limits UI work; still polls latest frame)
        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.setInterval(17)  # ~58-60 FPS target
        self._ui_timer.timeout.connect(self._update_from_camera)
        self._ui_timer.start()

        # Update camera name label now that camera has started (Basler name may be enriched in start())
        self._update_camera_name_label()

        self._apply_style()

    # ---- camera selection ----
    def _select_camera_backend(self) -> None:
        # Prefer Basler if available, else fallback to OpenCV
        if pylon is not None:
            try:
                tl_factory = pylon.TlFactory.GetInstance()
                devices = tl_factory.EnumerateDevices()
                if devices:
                    # Prefer a device with user-friendly name if present
                    chosen = devices[0]
                    for d in devices:
                        try:
                            if hasattr(d, "GetSerialNumber") and d.GetSerialNumber():
                                chosen = d
                                break
                        except Exception:
                            pass
                    self.camera = BaslerCameraBackend(chosen)
            except Exception:
                self.camera = None
        if self.camera is None:
            self.camera = OpenCVCameraBackend(0)

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
        name_text = self.camera.capabilities.name if self.camera else "No camera detected"
        self.camera_name_label.setText(f"Camera in use: {name_text}")
        self.camera_name_label.setObjectName("DimLabel")
        form.addRow(self.camera_name_label)

        # Exposure (default 100 ms per your environment)
        self.exposure_edit = QtWidgets.QLineEdit()
        self.exposure_edit.setText("100")
        self.exposure_edit.setValidator(QtGui.QIntValidator(1, 100000, self))
        self.exposure_edit.textChanged.connect(lambda t: self.camera and t.isdigit() and self.camera.set_exposure_ms(float(t)))
        form.addRow("Exposure (ms):", self.exposure_edit)

        # Gain auto
        self.gain_combo = QtWidgets.QComboBox()
        gain_items = (self.camera.capabilities.gain_auto_options if self.camera and self.camera.capabilities.gain_auto_options
                      else ["Off", "Once", "Continuous"])
        self.gain_combo.addItems(gain_items)
        self.gain_combo.currentTextChanged.connect(lambda t: self.camera and self.camera.set_gain_auto(t))
        form.addRow("Gain Setting:", self.gain_combo)

        # White balance auto
        self.wb_combo = QtWidgets.QComboBox()
        wb_items = (self.camera.capabilities.white_balance_auto_options if self.camera and self.camera.capabilities.white_balance_auto_options
                    else ["Off", "Once", "Continuous"])
        self.wb_combo.addItems(wb_items)
        self.wb_combo.currentTextChanged.connect(lambda t: self.camera and self.camera.set_white_balance_auto(t))
        form.addRow("White Balance:", self.wb_combo)

        # Frame rate
        self.fps_edit = QtWidgets.QLineEdit()
        self.fps_edit.setText("30")
        self.fps_edit.setValidator(QtGui.QIntValidator(1, 240, self))
        self.fps_edit.textChanged.connect(lambda t: self.camera and t.isdigit() and self.camera.set_framerate(float(t)))
        form.addRow("Framerate (fps):", self.fps_edit)

        return card

    def _build_output_settings_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        grid = QtWidgets.QGridLayout(card)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # Row 0: folder chooser
        folder_label = QtWidgets.QLabel("Choose output folder:")
        self.folder_edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._choose_folder)
        grid.addWidget(folder_label, 0, 0, 1, 2)
        grid.addWidget(browse, 1, 0)
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
        yes = QtWidgets.QRadioButton("Yes")
        no = QtWidgets.QRadioButton("No")
        yes.setChecked(True)
        group = QtWidgets.QButtonGroup(card)
        group.addButton(yes)
        group.addButton(no)
        shadows_row.addWidget(label)
        shadows_row.addStretch(1)
        shadows_row.addWidget(yes)
        shadows_row.addWidget(no)
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
        train_btn = create_round_button("TRAIN", variant="train")
        calib_btn = create_round_button("PLS CALIBRATE", variant="calib")
        vbox.addWidget(train_btn)
        vbox.addWidget(calib_btn)

        return card

    # ---- interactions ----
    def _choose_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.folder_edit.setText(path)

    def _update_from_camera(self) -> None:
        if self.camera is None:
            return
        frame = self.camera.get_latest_frame()
        if frame is not None:
            self.video_view.update_with_frame(frame)

    def _update_camera_name_label(self) -> None:
        if hasattr(self, "camera_name_label") and self.camera is not None:
            self.camera_name_label.setText(f"Camera in use: {self.camera.capabilities.name}")

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

            QPushButton#ActionButton[variant="calib"] {
                background: #3a3f4b; /* slate */
                border: 2px solid #8a8f9c;
                color: #ffffff;
            }
            QPushButton#ActionButton[variant="calib"]:hover { background: #444a57; }
            QPushButton#ActionButton[variant="calib"]:pressed { background: #2e323c; }
            QGroupBox::title { subcontrol-origin: margin; }
            QRadioButton, QLabel { padding: 2px; }
        """
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            if self.camera is not None:
                self.camera.stop()
        finally:
            super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


