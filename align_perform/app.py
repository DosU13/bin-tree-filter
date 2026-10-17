"""Start Align Perform."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import core  # noqa: F401  (OpenCV env before cv2 loads)
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from align_perform.runtime.qt_gl import configure_opengl


def dark_palette() -> QPalette:
    p = QPalette()
    base, alt, text = QColor(37, 37, 40), QColor(48, 48, 52), QColor(225, 225, 225)
    for role, color in (
        (QPalette.ColorRole.Window, QColor(45, 45, 48)), (QPalette.ColorRole.WindowText, text),
        (QPalette.ColorRole.Base, base), (QPalette.ColorRole.AlternateBase, alt),
        (QPalette.ColorRole.ToolTipBase, QColor(60, 60, 64)), (QPalette.ColorRole.ToolTipText, text),
        (QPalette.ColorRole.Text, text), (QPalette.ColorRole.Button, QColor(55, 55, 60)),
        (QPalette.ColorRole.ButtonText, text), (QPalette.ColorRole.Highlight, QColor(42, 110, 200)),
        (QPalette.ColorRole.HighlightedText, QColor(255, 255, 255)), (QPalette.ColorRole.Link, QColor(90, 160, 255)),
        (QPalette.ColorRole.PlaceholderText, QColor(130, 130, 130)),
    ):
        p.setColor(role, color)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(120, 120, 120))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(120, 120, 120))
    return p


def make_app(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance()
    if app is not None:
        return app
    configure_opengl(vsync=True)
    QCoreApplication.setOrganizationName("bin-tree-filter")
    QCoreApplication.setApplicationName("Align Perform")
    app = QApplication(argv if argv is not None else sys.argv[:1])
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m align_perform",
                                 description="Set up cameras and projectors for the ornament filter.")
    ap.add_argument("scene", nargs="?", help="scene file to open")
    ap.add_argument("--reset-layout", action="store_true", help="forget the saved window layout")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = make_app()
    from align_perform.main_window import MainWindow
    if args.reset_layout:
        from PySide6.QtCore import QSettings
        QSettings().clear()
    win = MainWindow(os.path.abspath(args.scene) if args.scene else None)
    win.show()
    return app.exec()
