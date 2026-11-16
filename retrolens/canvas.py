"""Air drawing: point with the index finger to paint neon strokes, open palm to erase."""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from . import config, fx
from . import gestures as gs
from . import tracking as tk
from .tracking import Hand

STROKE_GAP = 0.15      # seconds; a longer pause starts a new stroke instead of joining
BRUSH = 9
GLOW_DOWNSCALE = 4


class AirCanvas:
    def __init__(self):
        self._ink = None                 # BGR stroke layer
        self._mask = None                # uint8 mask of painted pixels
        self._core = None                # pale centre line, kept separate so joints stay clean
        self._core_mask = None
        self._last: Dict[str, Tuple[np.ndarray, float]] = {}
        self._cursors: List[Tuple[str, Tuple[int, int], int]] = []
        self.color_index = 0
        self.empty = True

    @property
    def color(self):
        return config.BRUSH_COLORS[self.color_index][1]

    @property
    def color_name(self) -> str:
        return config.BRUSH_COLORS[self.color_index][0]

    def next_color(self) -> None:
        self.color_index = (self.color_index + 1) % len(config.BRUSH_COLORS)

    def clear(self) -> None:
        if self._ink is not None:
            for layer in (self._ink, self._mask, self._core, self._core_mask):
                layer[:] = 0
        self._last.clear()
        self.empty = True

    def _ensure(self, shape) -> None:
        if self._ink is None or self._ink.shape[:2] != shape[:2]:
            self._ink = np.zeros((shape[0], shape[1], 3), np.uint8)
            self._mask = np.zeros(shape[:2], np.uint8)
            self._core = np.zeros_like(self._ink)
            self._core_mask = np.zeros_like(self._mask)
            self._last.clear()
            self.empty = True

    def update(self, frame_shape, hands: List[Hand], poses: Dict[str, str], t: float) -> None:
        self._ensure(frame_shape)
        self._cursors = []
        for hand in hands:
            pose = poses.get(hand.key, gs.OTHER)
            if pose == gs.POINT:
                tip = hand.pts[tk.INDEX_TIP]
                p = (int(tip[0]), int(tip[1]))
                prev = self._last.get(hand.key)
                start = (int(prev[0][0]), int(prev[0][1])) if prev and t - prev[1] < STROKE_GAP else p
                # Coloured tube with a pale core, like a neon sign.
                cv2.line(self._ink, start, p, self.color, BRUSH, cv2.LINE_AA)
                cv2.line(self._mask, start, p, 255, BRUSH, cv2.LINE_AA)
                cv2.line(self._core, start, p, fx.lerp_color(self.color, config.WHITE, 0.75), 3, cv2.LINE_AA)
                cv2.line(self._core_mask, start, p, 255, 3, cv2.LINE_AA)
                self._last[hand.key] = (tip.copy(), t)
                self.empty = False
                self._cursors.append(("draw", p, BRUSH))
            elif pose == gs.PALM:
                centre = hand.palm
                p = (int(centre[0]), int(centre[1]))
                radius = int(hand.size * 0.9)
                if not self.empty:
                    for layer in (self._ink, self._mask, self._core, self._core_mask):
                        cv2.circle(layer, p, radius, (0, 0, 0), -1)
                self._cursors.append(("erase", p, radius))

    def render(self, frame: np.ndarray) -> None:
        if self._ink is None or self.empty:
            return
        h, w = frame.shape[:2]
        small = cv2.resize(self._ink, (w // GLOW_DOWNSCALE, h // GLOW_DOWNSCALE), interpolation=cv2.INTER_AREA)
        halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(frame, 1.0, halo, 1.6, 0, dst=frame)
        cv2.copyTo(self._ink, self._mask, frame)
        cv2.copyTo(self._core, self._core_mask, frame)

    def draw_cursors(self, frame: np.ndarray) -> None:
        for kind, p, radius in self._cursors:
            if kind == "draw":
                cv2.circle(frame, p, radius + 6, self.color, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, p, radius, config.WHITE, 1, cv2.LINE_AA)
                fx.text(frame, "ERASE", (p[0] - 26, p[1] + 5), 0.5, config.WHITE)
