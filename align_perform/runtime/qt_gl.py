"""Qt <-> moderngl glue: OpenGL format setup and a QOpenGLWindow that hosts a GLRenderer."""

from __future__ import annotations

import moderngl
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow

from align_perform.runtime.gl_renderer import GLRenderer


def configure_opengl(vsync: bool = True) -> None:
    """Request OpenGL 3.3 core for every Qt GL surface. Call before creating QApplication."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapInterval(1 if vsync else 0)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    QSurfaceFormat.setDefaultFormat(fmt)


def qimage_to_bgr(img: QImage) -> np.ndarray:
    """QImage (any format) -> (h, w, 3) BGR uint8."""
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    data = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.bytesPerLine() * h)
    rgb = data.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
    return np.ascontiguousarray(rgb[:, :, ::-1])


class RendererWindow(QOpenGLWindow):
    """QOpenGLWindow with a moderngl context and a GLRenderer.

    Subclasses implement draw(fbo, width, height) using self.renderer; sizes are
    in physical pixels (device-pixel-ratio applied).
    """

    def __init__(self) -> None:
        super().__init__(QOpenGLWindow.UpdateBehavior.NoPartialUpdate)
        self.ctx: moderngl.Context | None = None
        self.renderer: GLRenderer | None = None
        self.frames = 0

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context(require=330)
        self.renderer = GLRenderer(self.ctx)
        self.on_gl_ready()

    def on_gl_ready(self) -> None:
        """Called once the GL context exists (upload tiles etc. here)."""

    def pixel_size(self) -> tuple[int, int]:
        dpr = self.devicePixelRatio()
        return max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr))

    def paintGL(self) -> None:
        w, h = self.pixel_size()
        fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        fbo.use()
        fbo.viewport = (0, 0, w, h)
        self.draw(fbo, w, h)
        self.frames += 1

    def draw(self, fbo: moderngl.Framebuffer, width: int, height: int) -> None:
        raise NotImplementedError

    def release_gl(self) -> None:
        if self.renderer is not None and self.context() is not None:
            self.makeCurrent()
            self.renderer.release()
            self.doneCurrent()
            self.renderer = None

    def hide_cursor(self, hidden: bool = True) -> None:
        self.setCursor(Qt.CursorShape.BlankCursor if hidden else Qt.CursorShape.ArrowCursor)
