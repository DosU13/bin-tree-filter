"""Add / change a camera's source: a webcam, an Astra depth camera, or a video / image / GIF file for testing."""

from __future__ import annotations

import os
import threading
import time

import cv2
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QTabWidget,
                               QVBoxLayout, QWidget)

from align_perform.panels.widgets import HintLabel, combo, dspin
from align_perform.runtime.session import bgr_to_qimage
from core import astra
from core.scene import SourceSpec
from core.sources import SourceError, kind_for_path, list_cameras, make_reader

RESOLUTIONS = [("Driver default", (0, 0)), ("640 × 480", (640, 480)), ("1280 × 720", (1280, 720)),
               ("1920 × 1080", (1920, 1080)), ("3840 × 2160", (3840, 2160))]
ASTRA_SIZES = {"depth": [(f"{w} × {h}" + (" (7 fps)" if w > 640 else ""), (w, h)) for w, h in astra.DEPTH_SIZES],
               "colour": [(f"{w} × {h}", (w, h)) for w, h in astra.COLOUR_SIZES]}
MEDIA_FILTER = "Media (*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv *.gif *.jpg *.jpeg *.png *.bmp *.webp);;All files (*)"


class _Probe(QObject):
    """Opens a source on a worker thread and reports (frame, fps) or an error."""

    done = Signal(object, float, str)

    def start(self, spec: SourceSpec, resolve) -> None:
        def run():
            reader = make_reader(spec, resolve)
            try:
                reader.open()
                frame = None
                deadline = time.perf_counter() + 10
                while frame is None and time.perf_counter() < deadline:
                    frame = reader.read()
                    if frame is None:
                        time.sleep(0.05)
                fps = 1 / reader.frame_interval if reader.frame_interval > 0 else 0.0
                if frame is None:
                    self.done.emit(None, 0.0, "opened, but no frames arrived")
                else:
                    self.done.emit(frame, fps, "")
            except Exception as exc:
                self.done.emit(None, 0.0, str(exc))
            finally:
                try:
                    reader.close()
                except Exception:
                    pass
        threading.Thread(target=run, daemon=True).start()


class AddCameraDialog(QDialog):
    def __init__(self, parent=None, resolve=lambda p: p, spec: SourceSpec | None = None, name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera source" if spec else "Add camera")
        self.resolve = resolve
        self.result_spec: SourceSpec | None = None
        self._serial = ""                 # serial of the Astra being edited (kept if not reconnected)
        self._depth_clean = SourceSpec.depth_clean     # its clean-up setting (edited in the inspector)
        self._frame_size = (0, 0)
        self._fps = 0.0

        self.tabs = QTabWidget()
        # Webcam tab
        cam_tab = QWidget()
        cl = QVBoxLayout(cam_tab)
        self.cam_list = QListWidget()
        refresh = QPushButton("Look for webcams again")
        refresh.clicked.connect(self._fill_cameras)
        cl.addWidget(self.cam_list, 1)
        cl.addWidget(refresh)
        form = QFormLayout()
        self.resolution = combo([r[0] for r in RESOLUTIONS], [r[1] for r in RESOLUTIONS])
        self.fps = dspin(0, 240, 1, 0, " fps")
        self.fps.setSpecialValueText("driver default")
        self.mjpg = QCheckBox("Request MJPG (helps several webcams share one USB bus)")
        form.addRow("Capture size", self.resolution)
        form.addRow("Frame rate", self.fps)
        form.addRow("", self.mjpg)
        cl.addLayout(form)
        self.cam_tab = cam_tab
        self.tabs.addTab(cam_tab, "Webcam")
        # Depth camera tab
        self.astra_tab = QWidget()
        al = QVBoxLayout(self.astra_tab)
        self.astra_list = QListWidget()
        refresh = QPushButton("Look for Astras again")
        refresh.clicked.connect(self._fill_astras)
        self.astra_msg = HintLabel()
        al.addWidget(self.astra_list, 1)
        al.addWidget(self.astra_msg)
        al.addWidget(refresh)
        form = QFormLayout()
        self.astra_stream = combo(["Depth", "Colour"], ["depth", "colour"])
        self.astra_size = combo([])
        self.near = dspin(0.0, 9.5, 0.1, 2, " m")
        self.far = dspin(0.5, 10.0, 0.1, 2, " m")
        self.near.setValue(SourceSpec.depth_near)
        self.far.setValue(SourceSpec.depth_far)
        form.addRow("Stream", self.astra_stream)
        form.addRow("Size", self.astra_size)
        form.addRow("Near (white)", self.near)
        form.addRow("Far (dark)", self.far)
        al.addLayout(form)
        al.addWidget(HintLabel("Orbbec Astra / Astra S. Depth is a grey picture: near is white, far is dark, "
                               "no reading is black; depth mode's band picks a slice of it. Plug the Astra straight "
                               "into the PC (not a hub) and keep half a metre clear in front of it: anything very "
                               "close switches its laser off."))
        self.astra_stream.currentIndexChanged.connect(self._fill_astra_sizes)
        self.tabs.addTab(self.astra_tab, "Depth camera")
        # File tab
        file_tab = QWidget()
        fl = QVBoxLayout(file_tab)
        row = QHBoxLayout()
        self.path = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        fl.addWidget(QLabel("Video, image or GIF (plays in a loop, for testing without cameras):"))
        fl.addLayout(row)
        self.noise = dspin(0, 30, 1, 1)
        nf = QFormLayout()
        nf.addRow("Image noise", self.noise)
        fl.addLayout(nf)
        fl.addStretch(1)
        self.file_tab = file_tab
        self.tabs.addTab(file_tab, "File")

        self.name = QLineEdit(name)
        self.name.setPlaceholderText("e.g. Left")
        self.test_btn = QPushButton("Test source")
        self.test_btn.clicked.connect(self._test)
        self.thumb = QLabel()
        self.thumb.setFixedSize(240, 135)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("background:#111; color:#888;")
        self.thumb.setText("no preview")
        self.info = QLabel()
        self.info.setWordWrap(True)

        bottom = QHBoxLayout()
        left = QFormLayout()
        left.addRow("Name", self.name)
        left.addRow(self.test_btn)
        left.addRow(self.info)
        bottom.addLayout(left, 1)
        bottom.addWidget(self.thumb)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self.ok = buttons.button(QDialogButtonBox.StandardButton.Ok)

        lay = QVBoxLayout(self)
        lay.addWidget(self.tabs, 1)
        lay.addLayout(bottom)
        lay.addWidget(buttons)
        self.resize(560, 520)

        self._probe = _Probe(self)
        self._probe.done.connect(self._probed)
        self._fill_cameras()
        self._fill_astra_sizes()
        self._fill_astras()
        if spec is not None:
            self._load(spec)

    # -- inputs --------------------------------------------------------------

    def _fill_cameras(self) -> None:
        self.cam_list.clear()
        cams = list_cameras()
        for c in cams:
            item = QListWidgetItem(f"{c.index}: {c.name}")
            item.setData(Qt.ItemDataRole.UserRole, (c.index, c.name))
            self.cam_list.addItem(item)
        if not cams:
            for i in range(3):
                item = QListWidgetItem(f"{i}: (index {i})")
                item.setData(Qt.ItemDataRole.UserRole, (i, ""))
                self.cam_list.addItem(item)
        self.cam_list.setCurrentRow(0)

    def _fill_astras(self) -> None:
        row = self.astra_list.currentRow()
        self.astra_list.clear()
        try:
            found = astra.list_astras()
        except SourceError as exc:
            self.astra_msg.setText(str(exc))
            return
        for a in found:
            item = QListWidgetItem(a.label())
            item.setData(Qt.ItemDataRole.UserRole, (a.index, a.uri))
            self.astra_list.addItem(item)
        self.astra_msg.setText("" if found else "No Astra connected.")
        self.astra_list.setCurrentRow(min(max(0, row), len(found) - 1))

    def _fill_astra_sizes(self) -> None:
        current = self.astra_size.currentData()
        self.astra_size.clear()
        for text, size in ASTRA_SIZES[self.astra_stream.currentData()]:
            self.astra_size.addItem(text, size)
        self.astra_size.setCurrentIndex(max(0, self.astra_size.findData(current)))

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video, image or GIF", self.path.text(), MEDIA_FILTER)
        if path:
            self.path.setText(path)
            if not self.name.text():
                self.name.setText(os.path.splitext(os.path.basename(path))[0])
            self._test()

    def _load(self, spec: SourceSpec) -> None:
        if spec.kind == "astra":
            self.tabs.setCurrentWidget(self.astra_tab)
            for i in range(self.astra_list.count()):
                idx, uri = self.astra_list.item(i).data(Qt.ItemDataRole.UserRole)
                if (spec.device_name and astra.known_serial(uri) == spec.device_name) or idx == spec.index:
                    self.astra_list.setCurrentRow(i)
                    break
            self.astra_stream.setCurrentIndex(max(0, self.astra_stream.findData(spec.stream)))
            self.astra_size.setCurrentIndex(max(0, self.astra_size.findData((spec.width, spec.height))))
            self.near.setValue(spec.depth_near)
            self.far.setValue(spec.depth_far)
            self._serial = spec.device_name
            self._depth_clean = spec.depth_clean
        elif spec.kind == "webcam":
            self.tabs.setCurrentWidget(self.cam_tab)
            for i in range(self.cam_list.count()):
                idx, name = self.cam_list.item(i).data(Qt.ItemDataRole.UserRole)
                if (spec.device_name and name == spec.device_name) or idx == spec.index:
                    self.cam_list.setCurrentRow(i)
                    break
            j = self.resolution.findData((spec.width, spec.height))
            self.resolution.setCurrentIndex(max(0, j))
            self.fps.setValue(spec.fps)
            self.mjpg.setChecked(spec.fourcc.upper() == "MJPG")
        else:
            self.tabs.setCurrentWidget(self.file_tab)
            self.path.setText(self.resolve(spec.path))
            self.noise.setValue(spec.noise)

    def current_spec(self) -> SourceSpec | None:
        if self.tabs.currentWidget() is self.astra_tab:
            item = self.astra_list.currentItem()
            if item is None:
                return None
            idx, uri = item.data(Qt.ItemDataRole.UserRole)
            w, h = self.astra_size.currentData()
            near, far = self.near.value(), self.far.value()
            return SourceSpec(kind="astra", index=idx, device_name=astra.known_serial(uri) or self._serial,
                              stream=self.astra_stream.currentData(), width=w, height=h,
                              depth_near=near, depth_far=max(far, near + 0.1), depth_clean=self._depth_clean)
        if self.tabs.currentWidget() is self.cam_tab:
            item = self.cam_list.currentItem()
            if item is None:
                return None
            idx, name = item.data(Qt.ItemDataRole.UserRole)
            w, h = self.resolution.currentData()
            return SourceSpec(kind="webcam", index=idx, device_name=name, width=w, height=h,
                              fps=self.fps.value(), fourcc="MJPG" if self.mjpg.isChecked() else "")
        path = self.path.text().strip()
        if not path:
            return None
        return SourceSpec(kind=kind_for_path(path), path=path, noise=self.noise.value())

    # -- probing -------------------------------------------------------------

    def _test(self) -> None:
        spec = self.current_spec()
        if spec is None:
            self.info.setText("Choose a camera or a file first.")
            return
        self.info.setText("Opening… (cameras can take a few seconds)")
        self.test_btn.setEnabled(False)
        self._probe.start(spec, self.resolve)

    def _probed(self, frame, fps: float, error: str) -> None:
        self.test_btn.setEnabled(True)
        if self.tabs.currentWidget() is self.astra_tab:
            self._fill_astras()            # opening it taught us its serial number
        if frame is None:
            self.info.setText(f"<span style='color:#e66'>Failed: {error}</span>")
            self._frame_size = (0, 0)
            return
        h, w = frame.shape[:2]
        self._frame_size, self._fps = (w, h), fps
        self.info.setText(f"OK: {w} × {h}" + (f" @ {fps:.0f} fps" if fps else ""))
        thumb = cv2.resize(frame, (240, max(1, int(240 * h / w))), interpolation=cv2.INTER_AREA)
        self.thumb.setPixmap(QPixmap.fromImage(bgr_to_qimage(thumb)))

    def _accept(self) -> None:
        spec = self.current_spec()
        if spec is None:
            self.info.setText("Choose a camera or a file first.")
            return
        if spec.kind == "astra":          # the size is the chosen mode: no test needed
            self.result_spec = spec
            self.accept()
            return
        if self._frame_size != (0, 0):
            spec.width, spec.height = self._frame_size
            if self._fps and spec.kind != "webcam":
                spec.fps = round(self._fps, 2)
        elif spec.kind != "webcam":
            self.info.setText("Testing the file first…")
            self._test()
            return
        self.result_spec = spec
        self.accept()
