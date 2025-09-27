'''
Professional, Commercial Image Editor.
Dream beyond.
imagine3.5light.py
'''


import importlib.util
import shutil
import threading
import time
import types
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import tkinter as tk
from tkinter import colorchooser, filedialog, simpledialog, messagebox

from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
from PIL.ImageQt import ImageQt
import numpy as np
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False

try:
    from PySide6 import QtWidgets, QtGui, QtCore
except Exception as exc:  # pragma: no cover - PySide6 is required for the enhanced canvas
    raise RuntimeError(
        "PySide6 is required for the Imagine advanced canvas but could not be imported"
    ) from exc


def to_device(arr: np.ndarray):
    """Move a NumPy array to GPU (CuPy) if available, else return the original NumPy array.

    Returns a tuple (array_on_device, is_gpu) where is_gpu is True when using CuPy.
    """
    if _CUPY_AVAILABLE:
        try:
            return cp.asarray(arr), True
        except Exception:
            return arr, False
    return arr, False


def to_host(arr):
    """Move array back to host (NumPy) if it's a CuPy array; otherwise return as-is."""
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


DEFAULT_CANVAS_SIZE = (800, 600)


def pil_to_tk(img: Image.Image):
    return ImageTk.PhotoImage(img.convert('RGBA'))


def pil_to_pixmap(img: Image.Image) -> QtGui.QPixmap:
    """Convert a PIL image to a QPixmap with alpha support."""

    image = img.convert('RGBA')
    qimage = ImageQt(image)
    return QtGui.QPixmap.fromImage(qimage)


def blend_arrays(dst: np.ndarray, src: np.ndarray, mode='normal'):
    # dst, src are HxWx4 uint8
    if mode == 'normal':
        a = src[..., 3:4] / 255.0
        dst[..., :3] = (src[..., :3] * a + dst[..., :3] * (1 - a)).astype(np.uint8)
        dst[..., 3] = np.clip(dst[..., 3] + src[..., 3], 0, 255).astype(np.uint8)
    elif mode == 'add':
        dst[..., :3] = np.clip(dst[..., :3].astype(int) + src[..., :3].astype(int), 0, 255).astype(np.uint8)
    elif mode == 'multiply':
        dst[..., :3] = (dst[..., :3].astype(int) * src[..., :3].astype(int) / 255).astype(np.uint8)
    else:
        # fallback
        return blend_arrays(dst, src, 'normal')


@dataclass
class Layer:
    name: str
    image: Image.Image
    visible: bool = True
    opacity: float = 1.0
    is_guide: bool = False  # guide layers won't be exported


class UndoRedo:
    def __init__(self, maxlen=200):
        self.undo = deque(maxlen=maxlen)
        self.redo = deque(maxlen=maxlen)

    def push(self, snapshot):
        self.undo.append(snapshot)
        self.redo.clear()

    def can_undo(self):
        return len(self.undo) > 0

    def can_redo(self):
        return len(self.redo) > 0

    def undo_step(self, current_snapshot):
        if not self.can_undo():
            return None
        last = self.undo.pop()
        self.redo.append(current_snapshot)
        return last

    def redo_step(self, current_snapshot):
        if not self.can_redo():
            return None
        nxt = self.redo.pop()
        self.undo.append(current_snapshot)
        return nxt


class BrushCache:
    def __init__(self):
        self.masks = {}
        self.gradients = {}
        # device-side cached gradients (CuPy arrays) to avoid repeated host->device copies
        self.gradients_dev = {}

    def get_circular_mask(self, radius, hardness=0.8):
        key = ('circle', radius, hardness)
        if key in self.masks:
            return self.masks[key]
        size = radius * 2
        arr = np.zeros((size, size), dtype=np.uint8)
        cy, cx = radius, radius
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = np.clip((radius - dist) / radius, 0, 1)
        mask = (mask * 255).astype(np.uint8)
        self.masks[key] = mask
        return mask

    def get_linear_gradient(self, size, color1, color2, direction=0):
        # direction in degrees: 0 = left->right
        key = ('linear', size, color1, color2, direction)
        if key in self.gradients:
            return self.gradients[key]
        w, h = size
        # create gradient array HxWx4
        grad = np.zeros((h, w, 4), dtype=np.uint8)
        # compute position along direction
        theta = np.deg2rad(direction)
        vx, vy = np.cos(theta), np.sin(theta)
        # coords normalized from 0..1 along direction
        xs = np.linspace(0, 1, w)
        ys = np.linspace(0, 1, h)
        X, Y = np.meshgrid(xs, ys)
        pos = X * vx + Y * vy
        pos = (pos - pos.min()) / (pos.max() - pos.min() + 1e-9)
        c1 = np.array(color1, dtype=float)
        c2 = np.array(color2, dtype=float)
        col = (c1 * (1 - pos[..., None]) + c2 * pos[..., None]).astype(np.uint8)
        grad[..., :3] = col
        grad[..., 3] = 255
        self.gradients[key] = grad
        # also store device copy if CuPy is available
        if _CUPY_AVAILABLE:
            try:
                self.gradients_dev[key] = cp.asarray(grad)
            except Exception:
                pass
        return grad

    def get_radial_gradient(self, size, color1, color2, center=None):
        key = ('radial', size, color1, color2, center)
        if key in self.gradients:
            return self.gradients[key]
        w, h = size
        cx, cy = (w // 2, h // 2) if center is None else center
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        pos = dist / dist.max()
        c1 = np.array(color1, dtype=float)
        c2 = np.array(color2, dtype=float)
        col = (c1 * (1 - pos[..., None]) + c2 * pos[..., None]).astype(np.uint8)
        grad = np.zeros((h, w, 4), dtype=np.uint8)
        grad[..., :3] = col
        grad[..., 3] = 255
        self.gradients[key] = grad
        if _CUPY_AVAILABLE:
            try:
                self.gradients_dev[key] = cp.asarray(grad)
            except Exception:
                pass
        return grad


class ImagineCanvasView(QtWidgets.QGraphicsView):
    """High-fidelity canvas powered by Qt's graphics view framework."""

    mouse_pressed = QtCore.Signal(int, int)
    mouse_moved = QtCore.Signal(int, int)
    mouse_released = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap_item = self.scene().addPixmap(QtGui.QPixmap())
        self.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
            | QtGui.QPainter.HighQualityAntialiasing
        )
        gradient = QtGui.QLinearGradient(0, 0, 0, 1)
        gradient.setCoordinateMode(QtGui.QGradient.ObjectBoundingMode)
        gradient.setColorAt(0.0, QtGui.QColor('#1c1f2b'))
        gradient.setColorAt(1.0, QtGui.QColor('#2e3250'))
        self.setBackgroundBrush(QtGui.QBrush(gradient))
        self.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def update_canvas(self, image: Image.Image):
        pixmap = pil_to_pixmap(image)
        self._pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(pixmap.rect())
        # keep a reference to prevent premature garbage collection
        self._current_pixmap = pixmap

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if event.modifiers() & QtCore.Qt.ControlModifier:
            angle = event.angleDelta().y()
            factor = 1.15 if angle > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            self.mouse_pressed.emit(int(pos.x()), int(pos.y()))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            self.mouse_moved.emit(int(pos.x()), int(pos.y()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_released.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ImagineApp(QtWidgets.QMainWindow):
    def __init__(self, root=None):
        super().__init__()
        self.setWindowTitle('Imagine 3.2 Zinnia - Enhanced')
        self.setMinimumSize(960, 720)
        self.canvas_w, self.canvas_h = DEFAULT_CANVAS_SIZE

        self.layers: List[Layer] = []
        base = Image.new('RGBA', (self.canvas_w, self.canvas_h), (255, 255, 255, 255))
        self.layers.append(Layer('Background', base.copy(), visible=True, opacity=1.0, is_guide=False))

        # guide layer example
        guide = Image.new('RGBA', (self.canvas_w, self.canvas_h), (0, 0, 0, 0))
        self.layers.append(Layer('Guides', guide, visible=True, opacity=0.6, is_guide=True))

        self.current_layer_idx = 1

        self.undo = UndoRedo()
        self.cache = BrushCache()

        # GPU toggle
        self.gpu_enabled = bool(_CUPY_AVAILABLE)

        # painting state
        self.brush = 'gradient'  # default new feature
        self.brush_size = 40
        self.gradient_color1 = (255, 0, 0)
        self.gradient_color2 = (0, 0, 255)
        self.gradient_mode = 'linear'
        self.gradient_direction = 0
        self.blend_mode = 'normal'
        self.brush_opacity = 0.9

        self._build_ui()
        self._bind_events()

        self.dragging = False
        self.last_pos = None

        # incremental composed image cache for partial refresh
        self.composed_image = Image.new('RGBA', (self.canvas_w, self.canvas_h), (0, 0, 0, 0))
        self.partial_bbox = None

        self._compose_all()
        self._update_canvas_image()

    def _build_ui(self):
        self.setStyleSheet(
            """
            QMainWindow { background-color: #11131b; }
            QWidget#controlBar { background-color: rgba(28, 32, 52, 200); border-radius: 14px; }
            QPushButton { color: #f2f4ff; background-color: #3f4b8c; border: none; padding: 8px 14px; border-radius: 8px; }
            QPushButton:hover { background-color: #5564b8; }
            QLabel { color: #d7dbff; }
            QComboBox, QSlider, QCheckBox { color: #e7e9ff; }
            QStatusBar { background: #181b2a; color: #9aa3ff; }
            """
        )

        central = QtWidgets.QWidget(self)
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(18, 18, 18, 18)
        central_layout.setSpacing(16)
        self.setCentralWidget(central)

        control_bar = QtWidgets.QWidget(objectName='controlBar')
        control_layout = QtWidgets.QHBoxLayout(control_bar)
        control_layout.setContentsMargins(16, 12, 16, 12)
        control_layout.setSpacing(12)

        def add_button(text, slot):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(slot)
            control_layout.addWidget(btn)
            return btn

        add_button('Open', self.open_image)
        add_button('Save', self.save_image)
        add_button('Undo', self.undo_action)
        add_button('Redo', self.redo_action)

        add_button('Color A', self.pick_color1)
        add_button('Color B', self.pick_color2)

        control_layout.addWidget(QtWidgets.QLabel('Mode'))
        self.gradient_mode_combo = QtWidgets.QComboBox()
        self.gradient_mode_combo.addItems(['linear', 'radial'])
        self.gradient_mode_combo.setCurrentText(self.gradient_mode)
        control_layout.addWidget(self.gradient_mode_combo)

        control_layout.addWidget(QtWidgets.QLabel('Direction'))
        self.gradient_dir_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gradient_dir_slider.setRange(0, 360)
        self.gradient_dir_slider.setValue(self.gradient_direction)
        self.gradient_dir_slider.setFixedWidth(140)
        control_layout.addWidget(self.gradient_dir_slider)

        control_layout.addWidget(QtWidgets.QLabel('Layer'))
        self.layer_combo = QtWidgets.QComboBox()
        self._refresh_layer_combo()
        control_layout.addWidget(self.layer_combo)

        control_layout.addWidget(QtWidgets.QLabel('Brush'))
        self.brush_combo = QtWidgets.QComboBox()
        self.brush_combo.addItems(['gradient', 'normal'])
        self.brush_combo.setCurrentText(self.brush)
        control_layout.addWidget(self.brush_combo)

        control_layout.addWidget(QtWidgets.QLabel('Size'))
        self.size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.size_slider.setRange(1, 200)
        self.size_slider.setValue(self.brush_size)
        self.size_slider.setFixedWidth(160)
        control_layout.addWidget(self.size_slider)

        add_button('Text', self.add_text)
        add_button('Guide Layer', self.add_guide_layer)
        add_button('Guide Note', self.add_guide_note)
        add_button('Help', self.show_help)

        self.gpu_checkbox = QtWidgets.QCheckBox('GPU')
        self.gpu_checkbox.setChecked(self.gpu_enabled)
        control_layout.addWidget(self.gpu_checkbox)

        control_layout.addStretch(1)
        central_layout.addWidget(control_bar)

        canvas_frame = QtWidgets.QFrame()
        canvas_frame.setStyleSheet('QFrame { background-color: rgba(18, 21, 34, 230); border-radius: 20px; }')
        frame_layout = QtWidgets.QVBoxLayout(canvas_frame)
        frame_layout.setContentsMargins(24, 24, 24, 24)
        frame_layout.setSpacing(0)

        self.canvas_view = ImagineCanvasView(self)
        frame_layout.addWidget(self.canvas_view)
        central_layout.addWidget(canvas_frame, 1)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(32)
        shadow.setColor(QtGui.QColor(20, 20, 40, 160))
        shadow.setOffset(0, 12)
        canvas_frame.setGraphicsEffect(shadow)

        self.status = self.statusBar()
        self.status.showMessage('Ready to imagine ✨')

    def _bind_events(self):
        self.gradient_mode_combo.currentTextChanged.connect(self._set_gradient_mode)
        self.gradient_dir_slider.valueChanged.connect(self.set_gradient_dir)
        self.layer_combo.currentTextChanged.connect(self.change_current_layer)
        self.brush_combo.currentTextChanged.connect(self._set_brush)
        self.size_slider.valueChanged.connect(self.change_size)
        self.gpu_checkbox.toggled.connect(self._toggle_gpu)
        self.canvas_view.mouse_pressed.connect(self.on_mouse_down)
        self.canvas_view.mouse_moved.connect(self.on_mouse_move)
        self.canvas_view.mouse_released.connect(self.on_mouse_up)

    def _refresh_layer_combo(self):
        if not hasattr(self, 'layer_combo'):
            return
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer in self.layers:
            self.layer_combo.addItem(layer.name)
        if self.layers:
            current_name = self.layers[self.current_layer_idx].name
            index = self.layer_combo.findText(current_name)
            self.layer_combo.setCurrentIndex(max(0, index))
        self.layer_combo.blockSignals(False)

    def _set_gradient_mode(self, mode: str):
        self.gradient_mode = mode
        self.cache.gradients.clear()

    def _set_brush(self, name: str):
        self.brush = name

    def _toggle_gpu(self, checked: bool):
        self.gpu_enabled = bool(checked)

    # Convenience wrappers around Qt dialogs ---------------------------------
    def _ask_text(self, title: str, prompt: str, default: str = "") -> Optional[str]:
        text, ok = QtWidgets.QInputDialog.getText(self, title, prompt, text=default)
        if not ok or not text:
            return None
        return text

    def _ask_int(self, title: str, prompt: str, default: int = 0, minimum: int = 0, maximum: int = 10_000) -> Optional[int]:
        value, ok = QtWidgets.QInputDialog.getInt(self, title, prompt, value=default, min=minimum, max=maximum)
        return value if ok else None

    def _ask_yes_no(self, title: str, prompt: str) -> bool:
        res = QtWidgets.QMessageBox.question(
            self,
            title,
            prompt,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return res == QtWidgets.QMessageBox.Yes

    def change_size(self, value):
        self.brush_size = int(value)

    def pick_color1(self):
        initial = QtGui.QColor(*self.gradient_color1)
        color = QtWidgets.QColorDialog.getColor(initial, self, 'Select primary gradient color')
        if color.isValid():
            self.gradient_color1 = (color.red(), color.green(), color.blue())
            self.cache.gradients.clear()

    def pick_color2(self):
        initial = QtGui.QColor(*self.gradient_color2)
        color = QtWidgets.QColorDialog.getColor(initial, self, 'Select secondary gradient color')
        if color.isValid():
            self.gradient_color2 = (color.red(), color.green(), color.blue())
            self.cache.gradients.clear()

    def set_gradient_dir(self, value):
        self.gradient_direction = int(value)
        self.cache.gradients.clear()

    def change_current_layer(self, name):
        for i, L in enumerate(self.layers):
            if L.name == name:
                self.current_layer_idx = i
                break
        self._refresh_layer_combo()

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Open image',
            '',
            'Images (*.png *.jpg *.jpeg *.bmp)'
        )
        if not path:
            return
        img = Image.open(path).convert('RGBA')
        w, h = img.size
        self.canvas_w, self.canvas_h = w, h
        self.layers[0].image = img.copy()
        for L in self.layers[1:]:
            L.image = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        self.composed_image = Image.new('RGBA', (w, h), (255, 255, 255, 255))
        self.canvas_view.resetTransform()
        self._refresh_layer_combo()
        self._compose_all()
        self._update_canvas_image()
        self.status.showMessage(f'Loaded {Path(path).name}', 5000)

    def save_image(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save image',
            '',
            'PNG Image (*.png)'
        )
        if not path:
            return
        # export: compose only non-guide layers
        out = Image.new('RGBA', (self.canvas_w, self.canvas_h), (255, 255, 255, 255))
        for L in self.layers:
            if not L.visible or L.is_guide:
                continue
            if L.opacity < 1.0:
                tmp = L.image.copy()
                alpha = tmp.split()[3].point(lambda p: int(p * L.opacity))
                tmp.putalpha(alpha)
                out = Image.alpha_composite(out, tmp)
            else:
                out = Image.alpha_composite(out, L.image)
        change = self._ask_yes_no('DPI', 'Change output DPI metadata?')

        try:
            if change:
                # get current dpi if available
                cur_dpi = get_image_ppi_from_dpi(out)
                dx = self._ask_int('DPI - Horizontal', 'Horizontal DPI (pixels per inch):', default=cur_dpi[0], minimum=1)
                if dx is None:
                    # user cancelled; abort save
                    return
                dy = self._ask_int('DPI - Vertical', 'Vertical DPI (pixels per inch):', default=cur_dpi[1], minimum=1)
                if dy is None:
                    return
                # set metadata and save with dpi parameter
                try:
                    out.save(path, dpi=(int(dx), int(dy)))
                except Exception:
                    # fallback: set info and save normally
                    out = set_image_dpi(out, (int(dx), int(dy)))
                    out.save(path)
            else:
                out.save(path)
        except Exception as e:
            # fallback: try to save normally and report error if any
            try:
                out.save(path)
            except Exception as e2:
                QtWidgets.QMessageBox.critical(self, 'Save Error', f'Could not save image: {e2}')
                return

        QtWidgets.QMessageBox.information(self, 'Saved', f'Saved to {path}')
        self.status.showMessage(f'Saved {Path(path).name}', 5000)

    def add_text(self):
        txt = self._ask_text('Add Text', 'Text:')
        if not txt:
            return
        font = ImageFont.load_default()
        L = self.layers[self.current_layer_idx]
        self._snapshot()
        draw = ImageDraw.Draw(L.image)
        x, y = 50, 50
        effect = self._ask_text(
            'Effect',
            'Preset (none, outline, shadow, comic, calligraphy):',
            default='outline'
        ) or 'none'
        if effect == 'outline':
            draw.text((x - 1, y), txt, font=font, fill=(0, 0, 0))
            draw.text((x + 1, y), txt, font=font, fill=(0, 0, 0))
            draw.text((x, y - 1), txt, font=font, fill=(0, 0, 0))
            draw.text((x, y + 1), txt, font=font, fill=(0, 0, 0))
            draw.text((x, y), txt, font=font, fill=(255, 255, 255))
        elif effect == 'shadow':
            draw.text((x + 2, y + 2), txt, font=font, fill=(0, 0, 0, 160))
            draw.text((x, y), txt, font=font, fill=(255, 255, 255))
        elif effect == 'comic':
            # bold with black outline and bright fill
            draw.text((x - 2, y), txt, font=font, fill=(0, 0, 0))
            draw.text((x, y), txt, font=font, fill=(255, 200, 0))
        elif effect == 'calligraphy':
            # simulate with repeated slightly offset strokes
            for dx, dy in [(-1, 0), (0, 0), (1, 0)]:
                draw.text((x + dx, y + dy), txt, font=font, fill=(10, 10, 10))
        else:
            draw.text((x, y), txt, font=font, fill=(0, 0, 0))
        self._compose_partial((x, y, x + 200, y + 50))
        self._update_canvas_image()

    def add_guide_layer(self):
        name = self._ask_text('Guide Layer', 'Name for guide layer:', default=f'Guide {len(self.layers)}')
        if not name:
            return
        img = Image.new('RGBA', (self.canvas_w, self.canvas_h), (0,0,0,0))
        L = Layer(name, img, visible=True, opacity=0.6, is_guide=True)
        self._snapshot()
        self.layers.append(L)
        self.current_layer_idx = len(self.layers)-1
        self._refresh_layer_combo()
        self._compose_all()
        self._update_canvas_image()

    def add_guide_note(self):
        # add text annotation to current guide layer or to a guide layer selected
        guide_idxs = [i for i,L in enumerate(self.layers) if L.is_guide]
        if not guide_idxs:
            QtWidgets.QMessageBox.information(self, 'No Guide', 'No guide layers exist. Create one first.')
            return
        # pick first guide layer for simplicity
        idx = guide_idxs[0]
        txt = self._ask_text('Guide Note', 'Note text:')
        if not txt:
            return
        x = self._ask_int('X', 'X position', default=50)
        y = self._ask_int('Y', 'Y position', default=50)
        color = (50, 100, 220, 180)  # default semi-blue
        self._snapshot()
        draw = ImageDraw.Draw(self.layers[idx].image)
        draw.text((x,y), txt, fill=color)
        self._compose_partial((x, y, x + 200, y + 50))
        self._update_canvas_image()

    def show_help(self):
        help_text = (
            'Imagine 3.2 Zinnia - Help\n\n'
            'Gradient Brush:\n'
            '- Pick two colors with Pick Color 1/2.\n'
            "- Brush blends between them along chosen direction (default left->right).\n"
            '- Gradient mode: linear or radial.\n\n'
            'Guide Layers:\n'
            "- Guide layers are semi-transparent annotation layers that do NOT export.\n"
            '- Toggle visibility in future UI.\n\n'
            'Performance:\n'
            '- Uses NumPy arrays for raster ops, cached brush masks/gradients, and partial refresh.\n\n'
            'Text Effects:\n'
            '- Presets: outline, shadow, comic, calligraphy.\n\n'
            'Undo/Redo:\n'
            '- Undo and Redo buttons cover painting, text, and guide edits.\n'
        )
        QtWidgets.QMessageBox.information(self, 'Help', help_text)

    def _snapshot(self):
        # push copy of current layers images
        snap = [(L.name, L.image.copy(), L.visible, L.opacity, L.is_guide) for L in self.layers]
        self.undo.push(snap)

    def undo_action(self):
        snap = self.undo.undo_step([(L.name, L.image.copy(), L.visible, L.opacity, L.is_guide) for L in self.layers])
        if snap is None:
            return
        # apply snap
        self.layers = [Layer(name, img.copy(), vis, op, isg) for (name, img, vis, op, isg) in snap]
        self.current_layer_idx = min(self.current_layer_idx, len(self.layers)-1)
        self._refresh_layer_combo()
        self._compose_all()
        self._update_canvas_image()

    def redo_action(self):
        snap = self.undo.redo_step([(L.name, L.image.copy(), L.visible, L.opacity, L.is_guide) for L in self.layers])
        if snap is None:
            return
        self.layers = [Layer(name, img.copy(), vis, op, isg) for (name, img, vis, op, isg) in snap]
        self.current_layer_idx = min(self.current_layer_idx, len(self.layers)-1)
        self._refresh_layer_combo()
        self._compose_all()
        self._update_canvas_image()

    def _compose_all(self):
        # compose all visible non-guide layers into self.composed_image
        base = Image.new('RGBA', (self.canvas_w, self.canvas_h), (255,255,255,255))
        for L in self.layers:
            if not L.visible or L.is_guide:
                continue
            if L.opacity < 1.0:
                tmp = L.image.copy()
                alpha = tmp.split()[3].point(lambda p: int(p * L.opacity))
                tmp.putalpha(alpha)
                base = Image.alpha_composite(base, tmp)
            else:
                base = Image.alpha_composite(base, L.image)
        self.composed_image = base

    def _compose_partial(self, bbox: Tuple[int,int,int,int]):
        # bbox in image coordinates; compose only that region
        x0,y0,x1,y1 = bbox
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(self.canvas_w, x1); y1 = min(self.canvas_h, y1)
        if x0>=x1 or y0>=y1:
            return
        region = Image.new('RGBA', (x1-x0, y1-y0), (255,255,255,0))
        for L in self.layers:
            if not L.visible or L.is_guide:
                continue
            crop = L.image.crop((x0,y0,x1,y1))
            if L.opacity < 1.0:
                alpha = crop.split()[3].point(lambda p: int(p * L.opacity))
                crop.putalpha(alpha)
            region = Image.alpha_composite(region, crop)
        self.composed_image.paste(region, (x0,y0), region)

    def _update_canvas_image(self):
        image = self._composite_with_guides()
        self.canvas_view.update_canvas(image)
        self.status.showMessage(f'Canvas {self.canvas_w}×{self.canvas_h}', 3000)

    def _composite_with_guides(self):
        # overlay guide layers on top of composed_image
        out = self.composed_image.copy()
        for L in self.layers:
            if not L.visible:
                continue
            if L.is_guide:
                tmp = L.image.copy().convert('RGBA')
                alpha = tmp.split()[3].point(lambda p: int(p * L.opacity))
                tmp.putalpha(alpha)
                out = Image.alpha_composite(out, tmp)
        return out

    def on_mouse_down(self, x: int, y: int):
        self.dragging = True
        self.last_pos = (x, y)
        if hasattr(self, 'brush_combo'):
            self.brush_combo.blockSignals(True)
            self.brush_combo.setCurrentText(self.brush)
            self.brush_combo.blockSignals(False)
        self._snapshot()

    def on_mouse_move(self, x: int, y: int):
        if not self.dragging:
            return
        x = max(0, min(self.canvas_w - 1, int(x)))
        y = max(0, min(self.canvas_h - 1, int(y)))
        self._paint_line(self.last_pos, (x, y))
        self.last_pos = (x, y)

    def on_mouse_up(self):
        if not self.dragging:
            return
        self.dragging = False
        self.last_pos = None

    def _paint_line(self, p0, p1):
        # Bresenham-ish line stepping with cached stamps
        x0,y0 = p0; x1,y1 = p1
        dx = x1 - x0; dy = y1 - y0
        dist = int(max(abs(dx), abs(dy)))
        if dist == 0:
            points = [(x0,y0)]
        else:
            points = [(int(x0 + dx * t / dist), int(y0 + dy * t / dist)) for t in range(dist+1)]
        # fetch current layer and its array
        L = self.layers[self.current_layer_idx]
        arr = np.array(L.image)
        h, w = arr.shape[:2]

        stamp = self._create_brush_stamp(self.brush_size)
        sh, sw = stamp.shape[:2]
        r = sh // 2

        # Use vectorized premultiplied alpha blending for each stamp region
        # Try to accelerate with CuPy when available by moving the whole layer to device
        use_gpu = _CUPY_AVAILABLE and getattr(self, 'gpu_enabled', True)

        d_dev = None
        if use_gpu:
            try:
                # move full layer to device as float32 normalized 0..1
                d_dev = cp.asarray(arr.astype(cp.float32) / 255.0)
            except Exception:
                use_gpu = False
                d_dev = None

        for (cx, cy) in points:
            x0 = cx - r; y0 = cy - r; x1 = x0 + sw; y1 = y0 + sh
            if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
                continue
            sx0 = max(0, -x0); sy0 = max(0, -y0)
            dx0 = max(0, x0); dy0 = max(0, y0)
            dx1 = min(w, x1); dy1 = min(h, y1)
            region = arr[dy0:dy1, dx0:dx1]
            stamp_region = stamp[sy0:sy0+(dy1-dy0), sx0:sx0+(dx1-dx0)]

            if use_gpu and d_dev is not None:
                try:
                    key = (self.brush_size, tuple(self.gradient_color1), tuple(self.gradient_color2), self.gradient_mode, self.gradient_direction, self.brush_opacity)
                    if key in self.cache.gradients_dev:
                        s_dev_full = self.cache.gradients_dev[key].astype(cp.float32) / 255.0
                    else:
                        s_dev_full = cp.asarray(stamp.astype(cp.float32) / 255.0)

                    s = s_dev_full[sy0:sy0+(dy1-dy0), sx0:sx0+(dx1-dx0)] if s_dev_full.shape[0] >= (dy1-dy0) and s_dev_full.shape[1] >= (dx1-dx0) else (cp.asarray(stamp_region.astype(cp.float32) / 255.0))

                    # apply brush opacity
                    s[..., 3] = s[..., 3] * float(self.brush_opacity)

                    # destination slice on device
                    d = d_dev[dy0:dy1, dx0:dx1]

                    s_rgb_pm = s[..., :3] * s[..., 3:4]
                    d_rgb_pm = d[..., :3] * d[..., 3:4]
                    out_a = s[..., 3:4] + d[..., 3:4] * (1.0 - s[..., 3:4])

                    if self.blend_mode == 'normal':
                        out_rgb = s_rgb_pm + d_rgb_pm * (1.0 - s[..., 3:4])
                    elif self.blend_mode == 'add':
                        out_rgb = cp.clip((s[..., :3] + d[..., :3]) * 0.5 * (s[..., 3:4] + d[..., 3:4]), 0, 1)
                    elif self.blend_mode == 'multiply':
                        out_rgb = s[..., :3] * d[..., :3]
                    else:
                        out_rgb = s_rgb_pm + d_rgb_pm * (1.0 - s[..., 3:4])

                    # avoid division by zero and convert from premultiplied
                    mask = (out_a[..., 0] > 1e-6)
                    out = cp.zeros_like(d)
                    if mask.any():
                        if self.blend_mode == 'normal':
                            out_rgb = s_rgb_pm + d_rgb_pm * (1.0 - s[..., 3:4])
                        elif self.blend_mode == 'multiply':
                            out_rgb = s[..., :3] * d[..., :3] * out_a
                        else:
                            out_rgb = s_rgb_pm + d_rgb_pm * (1.0 - s[..., 3:4])
                        out_rgb_un = cp.zeros_like(out_rgb)
                        out_rgb_un[mask] = (out_rgb[mask] / out_a[mask])
                        out[..., :3] = out_rgb_un
                    out[..., 3] = out_a[..., 0]

                    # write back float 0..1 into device buffer
                    d_dev[dy0:dy1, dx0:dx1] = out
                except Exception:
                    # if any error, fall back to CPU path for remaining points
                    use_gpu = False
                    d_dev = None

            if not use_gpu:
                src = stamp_region.astype(np.float32) / 255.0
                dst = region.astype(np.float32) / 255.0
                # apply brush opacity
                src[..., 3] = src[..., 3] * self.brush_opacity
                # premultiply
                src_rgb_pm = src[..., :3] * src[..., 3:4]
                dst_rgb_pm = dst[..., :3] * dst[..., 3:4]
                out_a = src[..., 3:4] + dst[..., 3:4] * (1.0 - src[..., 3:4])
                if self.blend_mode == 'normal':
                    out_rgb = src_rgb_pm + dst_rgb_pm * (1.0 - src[..., 3:4])
                elif self.blend_mode == 'add':
                    out_rgb = np.clip((src[..., :3] + dst[..., :3]) * 0.5 * (src[..., 3:4] + dst[..., 3:4]), 0, 1)
                elif self.blend_mode == 'multiply':
                    out_rgb = src[..., :3] * dst[..., :3]
                else:
                    out_rgb = src_rgb_pm + dst_rgb_pm * (1.0 - src[..., 3:4])
                # avoid division by zero
                mask = out_a[..., 0] > 1e-6
                out = np.zeros_like(region, dtype=np.uint8)
                # convert back from premultiplied
                if mask.any():
                    if self.blend_mode == 'normal':
                        out_rgb = src_rgb_pm + dst_rgb_pm * (1.0 - src[..., 3:4])
                    elif self.blend_mode == 'multiply':
                        out_rgb = src[..., :3] * dst[..., :3] * out_a
                    else:
                        out_rgb = src_rgb_pm + dst_rgb_pm * (1.0 - src[..., 3:4])
                    out_rgb_un = np.zeros_like(out_rgb)
                    out_rgb_un[mask] = (out_rgb[mask] / out_a[mask])
                    out[..., :3] = np.clip((out_rgb_un * 255.0), 0, 255).astype(np.uint8)
                out[..., 3] = np.clip(out_a[..., 0] * 255.0, 0, 255).astype(np.uint8)
                arr[dy0:dy1, dx0:dx1] = out

        # If GPU path used, copy device buffer back to host once
        if use_gpu and d_dev is not None:
            try:
                # convert float 0..1 device buffer back to uint8 host array
                arr = cp.asnumpy(cp.clip(d_dev * 255.0, 0, 255).astype(cp.uint8))
            except Exception:
                # fallback: keep arr as-is (CPU-updates already applied)
                pass

        L.image = Image.fromarray(arr, mode='RGBA')
        # partial compose and refresh
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        bbox = (min(xs)-self.brush_size, min(ys)-self.brush_size, max(xs)+self.brush_size, max(ys)+self.brush_size)
        self._compose_partial(bbox)
        self._update_canvas_image()

    def _create_brush_stamp(self, size):
        # return HxWx4 uint8 array stamp with gradient applied
        key = (size, tuple(self.gradient_color1), tuple(self.gradient_color2), self.gradient_mode, self.gradient_direction, self.brush_opacity)
        if key in self.cache.gradients:
            stamp = self.cache.gradients[key]
            return stamp.copy()

        diameter = size
        # base mask
        mask = self.cache.get_circular_mask(diameter, hardness=0.9)
        sh = sw = mask.shape[0]
        # create color layer
        if self.gradient_mode == 'linear':
            grad = self.cache.get_linear_gradient((sw, sh), self.gradient_color1, self.gradient_color2, self.gradient_direction)
        else:
            grad = self.cache.get_radial_gradient((sw, sh), self.gradient_color1, self.gradient_color2)

        stamp = np.zeros((sh, sw, 4), dtype=np.uint8)
        stamp[..., :3] = grad[..., :3]
        stamp[..., 3] = mask
        # store host copy
        self.cache.gradients[key] = stamp.copy()
        # also create device copy in separate cache if possible
        if _CUPY_AVAILABLE:
            try:
                self.cache.gradients_dev[key] = cp.asarray(stamp)
            except Exception:
                pass
        return stamp


# NOTE: previous lightweight ImagineApp startup removed to avoid launching
# two Tkinter main windows. The main application is started by the
# original ImageEditor main() at the end of this file.
"""
Image Editor Program
====================

This script implements a simple yet powerful image editing application
using Python's built‑in ``tkinter`` GUI toolkit and the `Pillow`
library (also known as PIL).  It provides many of the fundamental
features you would expect from a desktop image editor, including

* **Layers** — load images or create blank canvases as layers that
  can be stacked, reordered, shown/hidden, merged or deleted.
* **Brush** — freehand drawing on any layer with adjustable colour and
  size.  Drawing strokes are committed directly to the selected
  layer.
* **Filters** — apply common filters such as grayscale, invert,
  blur and sharpen to the selected layer.
* **Adjustments** — modify layer properties like transparency (alpha)
  and brightness using sliders.
* **Text** — add text labels to layers by clicking on the image and
  specifying the content, font size and colour.
* **File operations** — open existing images as layers and save the
  composite result of all visible layers to disk.

The application is designed to be straightforward to use while still
showing how to orchestrate multiple image operations in an object
oriented fashion.  Each layer holds both its original image and a
modifiable working copy so that adjustments and filters can be applied
without losing the ability to revert or re‑apply operations.  The
main canvas always displays a composited preview of all visible layers.

Prerequisites
-------------

Before running this program, you must install the Pillow library, which
adds advanced image processing capabilities to Python.  You can
install it with pip:

```
pip install pillow
```

The built in `tkinter` module is used for the GUI, so there are no
additional dependencies.  On most systems Tkinter comes bundled with
Python; if not, consult your platform's documentation on how to
enable Tk support.

Usage
-----

Run this file with Python (`python3 image_editor.py`) and a window
will open.  Use the menu bar at the top to load images, create new
layers, select tools (brush, text), apply filters and adjustments,
toggle layer visibility and save your work.  The current layer can
be changed via the layer list at the left of the window.

This implementation is intentionally kept self contained for easy
review and extension.  Feel free to explore the code, learn how it
works and add your own features!

This example demonstrates that building an image editor in Python is
feasible using Tkinter and Pillow.  In fact, tutorials like the one
on GeeksforGeeks show how to combine these libraries to perform
operations such as opening images, resizing, blurring, flipping and
rotating【182082927575283†L82-L104】.  The program here builds on those
concepts to provide a multi‑layer editing environment with drawing and
filter capabilities.
"""

import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, colorchooser, scrolledtext
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageChops, ImageOps
import numpy as np  # Ensure numpy is available globally for magic wand and filters
import math
import json
import time
import threading
import queue
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """Collects timing statistics for painting and compositing operations."""

    stroke_times: deque = field(default_factory=lambda: deque(maxlen=32))
    composite_times: deque = field(default_factory=lambda: deque(maxlen=32))
    last_gpu: bool = False

    def register_stroke(self, duration_ms: float, gpu_used: bool) -> None:
        if duration_ms >= 0:
            self.stroke_times.append(duration_ms)
        self.last_gpu = gpu_used

    def register_compose(self, duration_ms: float) -> None:
        if duration_ms >= 0:
            self.composite_times.append(duration_ms)

    def summary(self) -> str:
        parts: List[str] = []
        if self.stroke_times:
            parts.append(f"Stroke {self.stroke_times[-1]:.1f} ms")
            if len(self.stroke_times) > 1:
                avg = sum(self.stroke_times) / len(self.stroke_times)
                parts.append(f"Avg {avg:.1f} ms")
        if self.composite_times:
            parts.append(f"Compose {self.composite_times[-1]:.1f} ms")
        if self.last_gpu:
            parts.append("GPU active")
        return " | ".join(parts)


class ScopedTimer:
    """Context manager used to time expensive sections of code."""

    def __init__(self, callback):
        self._callback = callback
        self._start: float | None = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._callback is not None and self._start is not None:
            duration = (time.perf_counter() - self._start) * 1000.0
            try:
                self._callback(duration)
            except Exception:
                pass


def ensure_directory(path: Path) -> Path:
    """Create *path* (and parents) if it does not exist and return it."""

    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path


def fast_normal_composite(layers_info: List[Tuple[Image.Image, Tuple[int, int], float]],
                          base_size: Tuple[int, int]) -> Image.Image:
    """Composite layers that all use normal blending using NumPy acceleration."""

    width, height = base_size
    if width <= 0 or height <= 0:
        return Image.new("RGBA", base_size, (0, 0, 0, 0))

    canvas = np.zeros((height, width, 4), dtype=np.float32)
    for img, offset, opacity in layers_info:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if opacity < 1.0:
            arr[..., 3] *= float(opacity)
        ox, oy = int(offset[0]), int(offset[1])
        src_h, src_w = arr.shape[:2]
        x0 = max(0, ox)
        y0 = max(0, oy)
        x1 = min(width, ox + src_w)
        y1 = min(height, oy + src_h)
        if x0 >= x1 or y0 >= y1:
            continue
        src = arr[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        dst = canvas[y0:y1, x0:x1]
        src_a = src[..., 3:4]
        dst_a = dst[..., 3:4]
        out_a = src_a + dst_a * (1.0 - src_a)
        out_rgb = src[..., :3] * src_a + dst[..., :3] * dst_a * (1.0 - src_a)
        np.divide(out_rgb, np.maximum(out_a, 1e-6), out=dst[..., :3], where=out_a > 1e-6)
        dst[..., 3:4] = out_a
    return Image.fromarray(np.clip(canvas * 255.0, 0, 255).astype(np.uint8), "RGBA")


def generate_focus_peaking_overlay(image: Image.Image,
                                   highlight_color: Tuple[int, int, int] = (0, 255, 180),
                                   threshold: int = 32) -> Image.Image:
    """Create a coloured overlay highlighting areas of strong local contrast."""

    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=np.float32)
    if arr.size == 0:
        return Image.new("RGBA", image.size, (0, 0, 0, 0))
    arr -= arr.min()
    peak = arr.max() or 1.0
    norm = (arr / peak) ** 1.2
    norm = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    norm[norm < threshold] = 0
    mask = Image.fromarray(norm, mode="L")
    overlay = Image.new("RGBA", image.size, highlight_color + (0,))
    overlay.putalpha(mask)
    return overlay


def apply_high_pass_detail(image: Image.Image, radius: float, amount: float) -> Image.Image:
    """Return a sharpened copy of *image* using a high-pass enhancement."""

    if radius <= 0:
        return image.copy()
    base = image.convert("RGBA")
    blurred = base.filter(ImageFilter.GaussianBlur(radius=radius))
    base_arr = np.asarray(base, dtype=np.float32)
    blur_arr = np.asarray(blurred, dtype=np.float32)
    high = base_arr - blur_arr
    enhanced = base_arr + high * float(max(0.0, amount))
    enhanced[..., 3] = base_arr[..., 3]
    return Image.fromarray(np.clip(enhanced, 0, 255).astype(np.uint8), "RGBA")


def frequency_separation_layers(image: Image.Image, radius: float) -> Tuple[Image.Image, Image.Image]:
    """Split *image* into low/high frequency layers for retouching workflows."""

    base = image.convert("RGBA")
    low = base.filter(ImageFilter.GaussianBlur(radius=radius))
    base_arr = np.asarray(base, dtype=np.float32)
    low_arr = np.asarray(low, dtype=np.float32)
    high_arr = np.clip((base_arr - low_arr) + 128.0, 0, 255).astype(np.uint8)
    high = Image.fromarray(high_arr, "RGBA")
    return low, high


def liquify_deform(image: Image.Image,
                   mode: str = "push",
                   center: Optional[Tuple[float, float]] = None,
                   radius: float = 80.0,
                   strength: float = 0.5,
                   angle_degrees: float = 0.0) -> Image.Image:
    """Apply a lightweight liquify deformation to *image*.

    The implementation uses a Gaussian-weighted displacement field so that
    effects remain local to the selected radius while keeping the operation
    responsive enough for live previews inside Tkinter.
    """

    if radius <= 1e-3 or strength == 0:
        return image.copy()

    base = image.convert("RGBA")
    arr = np.asarray(base, dtype=np.float32)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return base

    if center is None:
        cx = w / 2.0
        cy = h / 2.0
    else:
        cx, cy = center

    yy, xx = np.indices((h, w), dtype=np.float32)
    dx = xx - float(cx)
    dy = yy - float(cy)
    dist = np.sqrt(dx ** 2 + dy ** 2)

    radius = max(radius, 1.0)
    falloff = np.exp(-(dist ** 2) / (2.0 * (radius ** 2)))
    falloff *= (dist <= radius).astype(np.float32)

    disp_x = np.zeros_like(dx)
    disp_y = np.zeros_like(dy)

    if mode in {"push", "pull"}:
        angle = math.radians(angle_degrees)
        vx = math.cos(angle)
        vy = math.sin(angle)
        if mode == "pull":
            vx = -vx
            vy = -vy
        disp_x = vx * strength * radius * falloff
        disp_y = vy * strength * radius * falloff
    elif mode in {"bloat", "pucker"}:
        norm = np.maximum(dist, 1e-3)
        direction_x = dx / norm
        direction_y = dy / norm
        scale = strength * radius * falloff
        if mode == "pucker":
            scale = -scale
        disp_x = direction_x * scale
        disp_y = direction_y * scale
    else:
        return base

    src_x = np.clip(xx - disp_x, 0, w - 1)
    src_y = np.clip(yy - disp_y, 0, h - 1)
    sample_x = np.rint(src_x).astype(int)
    sample_y = np.rint(src_y).astype(int)
    warped = arr[sample_y, sample_x]
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), "RGBA")


def content_aware_fill(image: Image.Image, blur_radius: int = 12) -> Image.Image:
    """Fill transparent pixels in *image* by blending neighbouring colours."""

    if blur_radius <= 0:
        blur_radius = 1
    base = image.convert("RGBA")
    arr = np.asarray(base).copy()
    if arr.size == 0:
        return base
    alpha = arr[..., 3]
    mask = alpha == 0
    if not mask.any():
        return base
    for channel in range(3):
        chan_img = Image.fromarray(arr[..., channel])
        blurred = chan_img.filter(ImageFilter.BoxBlur(radius=int(blur_radius)))
        arr[..., channel][mask] = np.asarray(blurred)[mask]
    arr[..., 3][mask] = 255
    return Image.fromarray(arr, "RGBA")


def render_histogram_image(image: Image.Image, size: Tuple[int, int] = (256, 150)) -> Image.Image:
    """Render an RGB histogram for *image* as a small preview graphic."""

    hist = image.convert("RGB").histogram()
    width, height = size
    hist_img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(hist_img)
    max_val = max(hist) or 1
    for i in range(256):
        r = hist[i] / max_val
        g = hist[i + 256] / max_val
        b = hist[i + 512] / max_val
        draw.line([(i, height), (i, height - int(r * height))], fill="#ff4d4d")
        draw.line([(i, height), (i, height - int(g * height))], fill="#4dff4d")
        draw.line([(i, height), (i, height - int(b * height))], fill="#4d4dff")
    return hist_img


def build_contact_sheet(images: List[Image.Image],
                        thumb_size: Tuple[int, int] = (256, 256),
                        columns: int = 3,
                        padding: int = 12,
                        background: Tuple[int, int, int, int] = (30, 30, 30, 255)) -> Image.Image:
    """Create a contact sheet from a list of images."""

    valid_images: List[Image.Image] = []
    for img in images:
        if img is None:
            continue
        thumb = img.convert("RGBA")
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except Exception:
            resample = Image.ANTIALIAS
        thumb.thumbnail(thumb_size, resample=resample)
        valid_images.append(thumb)

    if not valid_images:
        raise ValueError("No images supplied for contact sheet")

    cols = max(1, columns)
    rows = int(math.ceil(len(valid_images) / cols))
    cell_w, cell_h = thumb_size
    sheet_w = cols * cell_w + padding * (cols + 1)
    sheet_h = rows * cell_h + padding * (rows + 1)
    sheet = Image.new("RGBA", (sheet_w, sheet_h), background)

    for idx, thumb in enumerate(valid_images):
        row = idx // cols
        col = idx % cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        sheet.paste(thumb, (x, y), thumb)

    return sheet

# ----------------------------------------------------------------------
# Image utility functions
# ----------------------------------------------------------------------
class Tooltip:
    """Lightweight tooltip for Tk widgets."""
    def __init__(self, widget, text: str, delay_ms: int = 600):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._on_enter, add=True)
        widget.bind("<Leave>", self._on_leave, add=True)
        widget.bind("<ButtonPress>", self._on_leave, add=True)

    def _on_enter(self, _event=None):
        self._schedule()

    def _on_leave(self, _event=None):
        self._cancel()
        self._hide()

    def _schedule(self):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        try:
            if self._after_id:
                self.widget.after_cancel(self._after_id)
        except Exception:
            pass
        self._after_id = None

    def _show(self):
        if self._tip or not self.widget.winfo_viewable():
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self._tip = tk.Toplevel(self.widget)
            self._tip.wm_overrideredirect(True)
            self._tip.wm_geometry(f"+{x}+{y}")
            bg = "#333333"
            fg = "#ffffff"
            frame = tk.Frame(self._tip, bg=bg, bd=1)
            frame.pack()
            label = tk.Label(frame, text=self.text, bg=bg, fg=fg, padx=8, pady=4, justify=tk.LEFT)
            label.pack()
        except Exception:
            self._tip = None

    def _hide(self):
        try:
            if self._tip is not None:
                self._tip.destroy()
        except Exception:
            pass
        self._tip = None

def add_tooltip(widget, text: str):
    try:
        Tooltip(widget, text)
    except Exception:
        pass
def swirl_image(img: Image.Image, center: tuple[int, int] | None = None, strength: float = 4.0, radius: float | None = None) -> Image.Image:
    """Apply a swirl (liquify) effect to the image.

    This effect warps pixels around the centre point.  Pixels closer to
    the centre are rotated more than those further away, creating a
    whirlpool effect reminiscent of liquify tools in professional
    editors.  The algorithm iterates over the output image and
    computes source coordinates in the input image using polar
    coordinates.

    :param img: input PIL image (RGBA).
    :param center: optional (x, y) centre of swirl; defaults to image
        centre.
    :param strength: controls the amount of rotation (higher values
        produce a stronger swirl).
    :param radius: maximum distance from centre affected; defaults
        to half the minimum dimension.
    :returns: new PIL image with swirl effect applied.
    """
    width, height = img.size
    if center is None:
        cx, cy = width / 2.0, height / 2.0
    else:
        cx, cy = center
    if radius is None:
        radius = min(width, height) / 2.0
    # Prepare pixel access
    src_pixels = img.load()
    result = Image.new("RGBA", img.size)
    dst_pixels = result.load()
    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            r = math.hypot(dx, dy)
            if r < 1:
                # very close to centre; no change
                nx, ny = x, y
            else:
                if r > radius:
                    # outside swirl radius: unchanged
                    nx, ny = x, y
                else:
                    # compute angle and apply swirl
                    theta = math.atan2(dy, dx)
                    factor = 1 - (r / radius)
                    angle = strength * factor
                    new_theta = theta + angle
                    nx = cx + r * math.cos(new_theta)
                    ny = cy + r * math.sin(new_theta)
            ix = int(nx)
            iy = int(ny)
            if 0 <= ix < width and 0 <= iy < height:
                dst_pixels[x, y] = src_pixels[ix, iy]
            else:
                dst_pixels[x, y] = (0, 0, 0, 0)
        # Progress callback support (if registered) - report by row
        try:
            if _progress_callback is not None:
                _progress_callback(int((y + 1) / height * 100))
        except Exception:
            pass
    return result


# ----------------------- New utility functions -----------------------
def set_image_dpi(img: Image.Image, dpi: Tuple[int, int]) -> Image.Image:
    """Return image with DPI metadata set (Pillow stores as info).

    Note: This does not resample pixels; it only updates metadata used when
    saving formats that support DPI (e.g., JPEG, PNG). dpi is (x_dpi, y_dpi).
    """
    new = img.copy()
    try:
        new.info['dpi'] = dpi
    except Exception:
        pass
    return new


def get_image_ppi_from_dpi(img: Image.Image) -> Tuple[int, int]:
    """Return PPI (pixels per inch) from image DPI metadata if present.

    Falls back to (72, 72) if not present.
    """
    dpi = img.info.get('dpi')
    if isinstance(dpi, tuple) and len(dpi) == 2:
        return (int(dpi[0]), int(dpi[1]))
    if isinstance(dpi, (int, float)):
        return (int(dpi), int(dpi))
    return (72, 72)


# Global progress callback used by long-running image ops (set by UI)
_progress_callback = None

def register_progress_callback(cb):
    """Register a callable cb(percent:int) used by heavy image ops to report progress."""
    global _progress_callback
    _progress_callback = cb


class ProgressDialog:
    """Simple modal progress dialog with a percentage label and progress bar."""
    def __init__(self, parent, title="Processing", initial_text="Working..."):
        self.parent = parent
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        self.top.grab_set()
        self.top.resizable(False, False)
        self.label = ttk.Label(self.top, text=initial_text)
        self.label.pack(padx=12, pady=(12, 6))
        self.pb = ttk.Progressbar(self.top, orient='horizontal', length=300, mode='determinate')
        self.pb.pack(padx=12, pady=(0, 12))
        self.pb['value'] = 0
        self.top.update_idletasks()

    def set(self, percent: int, text: Optional[str] = None):
        try:
            self.pb['value'] = max(0, min(100, int(percent)))
            if text:
                self.label.config(text=text)
            self.top.update_idletasks()
        except Exception:
            pass

        brush_settings_frame = tk.LabelFrame(left_frame, text="Tool Settings", bg=bg_toolbar, fg=label_fg)
        brush_settings_frame.pack(padx=5, pady=(6, 6), fill=tk.X)

        tk.Button(brush_settings_frame, text="Configure Stamp", command=self._configure_stamp_tool, bg=btn_bg, fg=btn_fg, bd=1).pack(padx=4, pady=2, fill=tk.X)
        tk.Button(brush_settings_frame, text="Pattern Settings", command=self._configure_pattern_brush, bg=btn_bg, fg=btn_fg, bd=1).pack(padx=4, pady=2, fill=tk.X)
        tk.Button(brush_settings_frame, text="Gradient Settings", command=self._open_gradient_settings, bg=btn_bg, fg=btn_fg, bd=1).pack(padx=4, pady=2, fill=tk.X)

        self.pattern_type_var = tk.StringVar(value=self.pattern_settings.get('type', 'checker'))
        tk.Label(brush_settings_frame, text="Pattern Style:", bg=bg_toolbar, fg=label_fg).pack(anchor=tk.W, padx=4, pady=(4, 0))
        pattern_menu = ttk.OptionMenu(brush_settings_frame, self.pattern_type_var, self.pattern_type_var.get(), 'checker', 'stripes', 'diagonal', 'dots', command=lambda _: self._sync_pattern_var())
        pattern_menu.pack(padx=4, pady=2, fill=tk.X)

        tk.Label(brush_settings_frame, text="Smart Eraser Tolerance", bg=bg_toolbar, fg=label_fg).pack(anchor=tk.W, padx=4, pady=(6, 0))
        self.smart_eraser_tol_var = tk.IntVar(value=35)
        tol_scale = tk.Scale(brush_settings_frame, from_=5, to=120, orient=tk.HORIZONTAL, variable=self.smart_eraser_tol_var, bg=slider_bg, fg=slider_fg, highlightthickness=0)
        tol_scale.pack(padx=4, pady=(0, 4), fill=tk.X)

    def close(self):
        try:
            self.top.grab_release()
            self.top.destroy()
        except Exception:
            pass


# Simple font cache to avoid reloading fonts repeatedly
_FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

def load_font(font_spec: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """Load a truetype font by family name or file path, with caching and Korean fallback.

    font_spec may be a family name like 'Arial' or an absolute path to a .ttf/.otf file.
    If the requested font can't be loaded, attempts a set of reasonable fallbacks
    including common Korean-supporting fonts. Returns an ImageFont instance.
    """
    key = (font_spec or 'default', size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    tried = []
    def try_load(name: str):
        try:
            f = ImageFont.truetype(name, size)
            _FONT_CACHE[key] = f
            return f
        except Exception:
            tried.append(name)
            return None

    # If a specific font path/name was provided, try it first
    if font_spec:
        # Try direct path or family name
        f = try_load(font_spec)
        if f:
            return f
        # Try with .ttf extension
        if not font_spec.lower().endswith('.ttf') and not font_spec.lower().endswith('.otf'):
            f = try_load(font_spec + '.ttf')
            if f:
                return f

    # Common fallbacks: Windows and cross-platform Korean fonts, then Arial
    for name in ("malgun.ttf", "Malgun Gothic", "NotoSansKR-Regular.ttf", "NotoSansKR.ttf", "NanumGothic.ttf", "arial.ttf"):
        f = try_load(name)
        if f:
            return f

    # Last resort: default PIL bitmap font
    f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def poster_edges_filter(img: Image.Image, poster_levels: int = 4, edge_width: int = 1) -> Image.Image:
    """Apply posterize + edge detection to create a poster edges look.

    - Posterize reduces colour levels.
    - Edge detection produces a black outline which is composited on top.
    """
    # Posterize
    try:
        poster = ImageOps.posterize(img.convert('RGB'), bits=max(1, min(8, int(math.log2(poster_levels)) if poster_levels>1 else 1)))
    except Exception:
        # fallback simple posterize via quantize
        poster = img.convert('RGB').quantize(colors=max(2, poster_levels)).convert('RGB')
    # Edge detection
    edges = img.convert('L').filter(ImageFilter.FIND_EDGES)
    # Enhance edges and threshold
    edges = edges.point(lambda p: 255 if p > 40 else 0).convert('L')
    # Thicken edges
    for _ in range(edge_width):
        edges = edges.filter(ImageFilter.MaxFilter(3))
    # Composite edges over posterized image
    poster = poster.convert('RGBA')
    edge_rgba = Image.new('RGBA', poster.size, (0, 0, 0, 0))
    edge_pixels = edge_rgba.load()
    src = edges.load()
    w, h = poster.size
    for y in range(h):
        for x in range(w):
            if src[x, y] == 255:
                edge_pixels[x, y] = (0, 0, 0, 255)
        # progress per row
        try:
            if _progress_callback is not None:
                _progress_callback(int((y + 1) / h * 100))
        except Exception:
            pass
    out = Image.alpha_composite(poster.convert('RGBA'), edge_rgba)
    return out


def expand_selection(mask: Image.Image, pixels: int = 5) -> Image.Image:
    """Expand a binary selection mask by `pixels` using dilation.

    Expects `mask` to be 'L' or '1' mode where selected=255.
    """
    if mask.mode != 'L':
        m = mask.convert('L')
    else:
        m = mask.copy()
    # Use a maximum filter to dilate selection
    try:
        for _ in range(max(0, pixels)):
            m = m.filter(ImageFilter.MaxFilter(3))
    except Exception:
        # fallback using numpy if available; try scipy's binary_dilation if installed
        arr = np.array(m)
        try:
            from scipy.ndimage import binary_dilation
            arr = binary_dilation(arr > 128, iterations=pixels).astype(np.uint8) * 255
            m = Image.fromarray(arr, mode='L')
        except Exception:
            # as a last resort run extra local MaxFilter passes (already attempted above)
            for _ in range(max(0, pixels)):
                m = m.filter(ImageFilter.MaxFilter(3))
    return m


def make_cutline(selection_mask: Image.Image, simplify_tol: float = 2.0) -> List[List[Tuple[int, int]]]:
    """Convert a binary selection mask into contour paths suitable for cutlines.

    Returns a list of contours, each contour is a list of (x,y) points.
    """
    # Ensure binary 'L'
    mask = selection_mask.convert('L')
    arr = np.array(mask)
    # find contours using OpenCV if available, otherwise a naive marching squares
    try:
        import cv2
        contours, _ = cv2.findContours((arr > 128).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        paths = []
        for c in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in c]
            paths.append(pts)
        return paths
    except Exception:
        # Simple (slow) contour trace: iterate boundary pixels and collect runs
        h, w = arr.shape
        visited = np.zeros_like(arr, dtype=bool)
        paths: List[List[Tuple[int, int]]] = []
        for y in range(h):
            for x in range(w):
                if arr[y, x] > 128 and not visited[y, x]:
                    # flood fill to get region
                    stack = [(x, y)]
                    region = []
                    while stack:
                        px, py = stack.pop()
                        if px < 0 or py < 0 or px >= w or py >= h:
                            continue
                        if visited[py, px] or arr[py, px] <= 128:
                            continue
                        visited[py, px] = True
                        region.append((px, py))
                        stack.extend([(px+1, py), (px-1, py), (px, py+1), (px, py-1)])
                    if region:
                        # compute convex hull-like path by taking bounding contour
                        xs = [p[0] for p in region]
                        ys = [p[1] for p in region]
                        bbox = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
                        paths.append(bbox)
        return paths



def contract_selection(mask: Image.Image, pixels: int = 3) -> Image.Image:
    """Contract a binary selection mask by repeatedly applying a minimum filter."""

    if mask.mode != 'L':
        contracted = mask.convert('L')
    else:
        contracted = mask.copy()
    for _ in range(max(0, pixels)):
        try:
            contracted = contracted.filter(ImageFilter.MinFilter(3))
        except Exception:
            arr = np.array(contracted)
            arr = np.pad(arr, 1, mode='edge')
            out = np.zeros_like(arr)
            for y in range(1, arr.shape[0] - 1):
                for x in range(1, arr.shape[1] - 1):
                    window = arr[y-1:y+2, x-1:x+2]
                    out[y, x] = window.min()
            arr = out[1:-1, 1:-1]
            contracted = Image.fromarray(arr.astype(np.uint8), mode='L')
    return contracted


def refine_selection_edges(mask: Image.Image,
                           smooth: int = 2,
                           feather: int = 5,
                           contrast: float = 0.0,
                           shift: int = 0) -> Image.Image:
    """Return a refined copy of *mask* applying smoothing, feather and expansion/contract."""

    refined = mask.convert('L') if mask.mode != 'L' else mask.copy()
    for _ in range(max(0, smooth)):
        try:
            refined = refined.filter(ImageFilter.MedianFilter(3))
        except Exception:
            break
    if feather > 0:
        refined = refined.filter(ImageFilter.GaussianBlur(radius=feather))
    if shift > 0:
        refined = expand_selection(refined, shift)
    elif shift < 0:
        refined = contract_selection(refined, abs(shift))
    if contrast:
        try:
            enhancer = ImageEnhance.Contrast(refined)
            refined = enhancer.enhance(max(0.1, 1.0 + contrast / 100.0))
        except Exception:
            pass
    return refined.point(lambda v: max(0, min(255, int(v))))


def apply_duotone(image: Image.Image,
                  shadows: Tuple[int, int, int],
                  highlights: Tuple[int, int, int]) -> Image.Image:
    """Apply a duotone colour mapping using the supplied shadow/highlight colours."""

    base = image.convert('RGBA')
    gray = base.convert('L')
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    sh = np.array(shadows, dtype=np.float32)
    hi = np.array(highlights, dtype=np.float32)
    mapped = sh + (hi - sh) * arr[..., None]
    alpha = np.asarray(base.split()[-1], dtype=np.float32)[..., None]
    out = np.concatenate([mapped, alpha], axis=2)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'RGBA')


def add_vignette(image: Image.Image, strength: float = 0.55, softness: float = 0.4) -> Image.Image:
    """Apply a radial vignette falloff to darken image edges."""

    base = image.convert('RGBA')
    arr = np.asarray(base, dtype=np.float32)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return base
    yy, xx = np.ogrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2) or 1.0
    norm = dist / max_dist
    mask = np.clip((norm - softness) / max(1e-3, 1.0 - softness), 0.0, 1.0)
    factor = 1.0 - mask * strength
    arr[..., :3] *= factor[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGBA')


def add_film_grain(image: Image.Image, amount: float = 0.08) -> Image.Image:
    """Overlay random grain onto the image to simulate film texture."""

    base = image.convert('RGBA')
    arr = np.asarray(base, dtype=np.float32)
    if arr.size == 0 or amount <= 0:
        return base
    noise = np.random.normal(0.0, 255.0 * amount, size=arr[..., :3].shape)
    arr[..., :3] = np.clip(arr[..., :3] + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), 'RGBA')


def apply_canva_preset(image: Image.Image, preset: str) -> Image.Image:
    """Apply a Canva-inspired stylistic preset by name."""

    preset = preset.lower()
    base = image.convert('RGBA')
    if preset == 'duotone':
        return apply_duotone(base, (40, 55, 145), (240, 220, 205))
    if preset == 'vignette':
        return add_vignette(base, strength=0.65, softness=0.35)
    if preset == 'glow':
        blurred = base.filter(ImageFilter.GaussianBlur(radius=12))
        arr = np.asarray(base, dtype=np.float32)
        glow = np.asarray(blurred, dtype=np.float32)
        arr[..., :3] = np.clip(arr[..., :3] + glow[..., :3] * 0.35, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), 'RGBA')
    if preset in ('film_grain', 'grain'):
        toned = ImageEnhance.Contrast(base).enhance(1.1)
        toned = ImageEnhance.Color(toned).enhance(0.9)
        return add_film_grain(toned, amount=0.12)
    if preset == 'cinematic':
        arr = np.asarray(base, dtype=np.float32)
        rgb = arr[..., :3] / 255.0
        lum = rgb.mean(axis=2, keepdims=True)
        shadows = np.clip((0.5 - lum) * 2.0, 0.0, 1.0)
        highlights = np.clip((lum - 0.5) * 2.0, 0.0, 1.0)
        rgb = np.clip(rgb * 1.08, 0, 1)
        rgb[..., 2] += 0.25 * shadows[..., 0]
        rgb[..., 0] += 0.18 * highlights[..., 0]
        rgb = np.clip(rgb, 0, 1)
        arr[..., :3] = rgb * 255.0
        result = Image.fromarray(arr.astype(np.uint8), 'RGBA')
        return add_vignette(result, strength=0.35, softness=0.55)
    if preset == 'neon':
        result = ImageEnhance.Color(base).enhance(1.6)
        result = ImageEnhance.Brightness(result).enhance(1.1)
        overlay = Image.new('RGBA', result.size, (60, 0, 120, 60))
        result = Image.alpha_composite(result, overlay)
        return result.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    if preset in ('hdr', 'dramatic'):
        boosted = ImageEnhance.Contrast(base).enhance(1.25)
        boosted = ImageEnhance.Brightness(boosted).enhance(1.05)
        return boosted.filter(ImageFilter.UnsharpMask(radius=2, percent=220, threshold=2))
    if preset == 'retro':
        faded = ImageEnhance.Color(base).enhance(0.85)
        faded = ImageEnhance.Brightness(faded).enhance(1.08)
        overlay = Image.new('RGBA', faded.size, (255, 200, 150, 40))
        return Image.alpha_composite(faded, overlay)
    if preset == 'summer':
        warmed = ImageEnhance.Color(base).enhance(1.2)
        warmed = ImageEnhance.Brightness(warmed).enhance(1.05)
        overlay = Image.new('RGBA', warmed.size, (255, 140, 40, 50))
        return Image.alpha_composite(warmed, overlay)
    if preset == 'winter':
        cooled = ImageEnhance.Color(base).enhance(0.95)
        cooled = ImageEnhance.Contrast(cooled).enhance(1.05)
        overlay = Image.new('RGBA', cooled.size, (40, 120, 255, 60))
        return Image.alpha_composite(cooled, overlay)
    if preset in ('background_blur', 'bg_blur'):
        blurred = base.filter(ImageFilter.GaussianBlur(radius=18))
        return ImageEnhance.Brightness(blurred).enhance(1.05)
    raise ValueError(f"Unknown Canva preset '{preset}'")


def apply_precise_lighting_color_adjustments(image: Image.Image,
                                             exposure: float = 0.0,
                                             highlights: float = 0.0,
                                             shadows: float = 0.0,
                                             whites: float = 0.0,
                                             blacks: float = 0.0,
                                             temperature: float = 0.0,
                                             tint: float = 0.0,
                                             vibrance: float = 0.0,
                                             saturation: float = 0.0,
                                             clarity: float = 0.0) -> Image.Image:
    """Apply a suite of tonal and colour adjustments to *image* and return the result."""

    base = image.convert('RGBA')
    arr = np.asarray(base, dtype=np.float32)
    if arr.size == 0:
        return base
    rgb = arr[..., :3] / 255.0
    alpha = arr[..., 3:4] / 255.0
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    rgb *= 2.0 ** float(exposure)

    def _apply_region(delta: float, mask: np.ndarray):
        nonlocal rgb
        if delta == 0:
            return
        amt = delta / 100.0
        if amt > 0:
            rgb += (1.0 - rgb) * amt * mask[..., None]
        else:
            rgb += rgb * amt * mask[..., None]

    highlight_mask = np.clip((lum - 0.55) * 3.0, 0.0, 1.0)
    shadow_mask = np.clip((0.45 - lum) * 3.0, 0.0, 1.0)
    whites_mask = np.clip((lum - 0.75) * 4.0, 0.0, 1.0)
    blacks_mask = np.clip((0.25 - lum) * 4.0, 0.0, 1.0)

    _apply_region(highlights, highlight_mask)
    _apply_region(shadows, shadow_mask)
    _apply_region(whites, whites_mask)
    _apply_region(blacks, blacks_mask)

    temp = temperature / 100.0
    rgb[..., 0] += temp * 0.12
    rgb[..., 2] -= temp * 0.12

    t = tint / 100.0
    rgb[..., 1] += t * 0.1
    rgb[..., 0] -= t * 0.05
    rgb[..., 2] += t * 0.05

    vibr = vibrance / 100.0
    if vibr:
        sat = np.max(rgb, axis=2) - np.min(rgb, axis=2)
        weight = 1.0 - np.clip(sat / 0.6, 0.0, 1.0)
        mean = rgb.mean(axis=2, keepdims=True)
        rgb = mean + (rgb - mean) * (1.0 + vibr * weight[..., None])

    sat_amt = saturation / 100.0
    if sat_amt:
        mean = rgb.mean(axis=2, keepdims=True)
        rgb = mean + (rgb - mean) * (1.0 + sat_amt)

    rgb = np.clip(rgb, 0.0, 1.0)
    arr[..., :3] = rgb * 255.0
    arr[..., 3:4] = np.clip(alpha, 0, 1) * 255.0
    result = Image.fromarray(arr.astype(np.uint8), 'RGBA')

    if clarity:
        clarity_amt = max(-100.0, min(100.0, clarity)) / 100.0
        if clarity_amt >= 0:
            sharp = result.filter(ImageFilter.UnsharpMask(radius=2, percent=int(200 * clarity_amt + 1), threshold=2))
            result = Image.blend(result, sharp, clarity_amt)
        else:
            blur = result.filter(ImageFilter.GaussianBlur(radius=2))
            result = Image.blend(result, blur, -clarity_amt)

    return result



# ----------------------------------------------------------------------

class VectorObject:
    """Base class for vector objects."""
    def __init__(self, x: float, y: float, fill_color: str = "#000000", 
                 stroke_color: str = "#000000", stroke_width: float = 1.0, 
                 opacity: float = 1.0, visible: bool = True):
        self.x = x
        self.y = y
        self.fill_color = fill_color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.opacity = opacity
        self.visible = visible

class VectorRectangle(VectorObject):
    """Vector rectangle object."""
    def __init__(self, x: float, y: float, width: float, height: float, 
                 fill_color: str = "#000000", stroke_color: str = "#000000", 
                 stroke_width: float = 1.0, opacity: float = 1.0, 
                 visible: bool = True, corner_radius: float = 0.0):
        super().__init__(x, y, fill_color, stroke_color, stroke_width, opacity, visible)
        self.width = width
        self.height = height
        self.corner_radius = corner_radius

class VectorCircle(VectorObject):
    """Vector circle object."""
    def __init__(self, x: float, y: float, radius: float, 
                 fill_color: str = "#000000", stroke_color: str = "#000000", 
                 stroke_width: float = 1.0, opacity: float = 1.0, 
                 visible: bool = True):
        super().__init__(x, y, fill_color, stroke_color, stroke_width, opacity, visible)
        self.radius = radius

class VectorEllipse(VectorObject):
    """Vector ellipse object."""
    def __init__(self, x: float, y: float, width: float, height: float, 
                 fill_color: str = "#000000", stroke_color: str = "#000000", 
                 stroke_width: float = 1.0, opacity: float = 1.0, 
                 visible: bool = True):
        super().__init__(x, y, fill_color, stroke_color, stroke_width, opacity, visible)
        self.width = width
        self.height = height

class VectorLine(VectorObject):
    """Vector line object."""
    def __init__(self, x: float, y: float, x2: float, y2: float, 
                 fill_color: str = "#000000", stroke_color: str = "#000000", 
                 stroke_width: float = 1.0, opacity: float = 1.0, 
                 visible: bool = True):
        super().__init__(x, y, fill_color, stroke_color, stroke_width, opacity, visible)
        self.x2 = x2
        self.y2 = y2

class VectorPath(VectorObject):
    """Vector path object with Bézier curves."""
    def __init__(self, x: float, y: float, points: List[Tuple[float, float]], 
                 fill_color: str = "#000000", stroke_color: str = "#000000", 
                 stroke_width: float = 1.0, opacity: float = 1.0, 
                 visible: bool = True, closed: bool = False):
        super().__init__(x, y, fill_color, stroke_color, stroke_width, opacity, visible)
        self.points = points
        self.closed = closed

class VectorText(VectorObject):
    """Vector text object."""
    def __init__(self, x: float, y: float, text: str, 
                 fill_color: str = "#000000", stroke_color: str = "#000000", 
                 stroke_width: float = 1.0, opacity: float = 1.0, 
