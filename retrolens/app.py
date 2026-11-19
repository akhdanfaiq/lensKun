"""Application core: wires tracking, gestures, portal, canvas, capture and HUD together."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config, hud, theme
from . import gestures as gs
from .canvas import AirCanvas
from .capture import CaptureManager
from .filters import build_filters
from .portal import Portal
from .tracking import HandTracker

MODE_PORTAL, MODE_DRAW = "PORTAL", "DRAW"
TOAST_TIME = 1.8


class App:
    def __init__(self, mirror: bool = True, capture_dir=config.CAPTURE_DIR):
        self.tracker = HandTracker()
        self.gestures = gs.GestureEngine()
        self.filters = build_filters()
        self.portal = Portal()
        self.canvas = AirCanvas()
        self.capture = CaptureManager(capture_dir)

        self.mirror = mirror
        self.mode = MODE_PORTAL
        self.filter_index = 0
        self._prev_filter: Optional[int] = None
        self._switched_at = -1e9
        self.show_help = True
        self.fps = 0.0
        self._last_t: Optional[float] = None
        self._toasts: List[Tuple[str, float]] = []
        self._clean: Optional[np.ndarray] = None   # last frame without HUD (what gets saved)
        self._t = 0.0

    # --- actions (shared by gestures and keyboard) ---------------------------------
    def toast(self, message: str) -> None:
        self._toasts = [x for x in self._toasts if x[1] > self._t][-2:]
        self._toasts.append((message, self._t + TOAST_TIME))

    def step_filter(self, delta: int) -> None:
        self._prev_filter = self.filter_index
        self.filter_index = (self.filter_index + delta) % len(self.filters)
        self._switched_at = self._t

    def next_item(self, delta: int = 1) -> None:
        if self.mode == MODE_PORTAL:
            self.step_filter(delta)
        else:
            self.canvas.step_color(delta)

    def toggle_mode(self) -> None:
        self.mode = MODE_DRAW if self.mode == MODE_PORTAL else MODE_PORTAL
        self.toast(f"{self.mode.capitalize()} mode")

    def take_photo(self) -> None:
        self.capture.start_countdown(self._t)

    def toggle_recording(self) -> None:
        if self._clean is None:
            return
        was_recording = self.capture.recording
        path = self.capture.toggle_recording(self._clean, self._t)
        if path is None:
            self.toast("Recording failed")
        elif was_recording:
            self.toast(f"Video saved: {path.name}")
        else:
            self.toast("Recording started")

    def clear_canvas(self) -> None:
        self.canvas.clear()
        self.toast("Drawing cleared")

    def handle_key(self, key: int) -> bool:
        """Returns False when the app should quit."""
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("n"), ord("N")):
            self.next_item(1)
        elif key in (ord("p"), ord("P")):
            self.next_item(-1)
        elif key in (ord("d"), ord("D"), 9):
            self.toggle_mode()
        elif key == 32:
            self.take_photo()
        elif key in (ord("r"), ord("R")):
            self.toggle_recording()
        elif key in (ord("c"), ord("C")):
            self.clear_canvas()
        elif key in (ord("t"), ord("T")):
            self.toast(f"Theme: {theme.cycle().name}")
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
        return True

    # --- per frame --------------------------------------------------------------------
    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Takes a raw BGR camera frame, returns the frame to display."""
        self._t = t
        if self._last_t is not None and t > self._last_t:
            inst = 1.0 / (t - self._last_t)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._last_t = t

        if self.mirror:
            frame = cv2.flip(frame, 1)
        frame = np.ascontiguousarray(frame)

        hands = self.tracker.detect(frame, t)
        g = self.gestures.update(hands, t)
        if gs.EVENT_NEXT in g.events:
            self.next_item(1)
        if gs.EVENT_PHOTO in g.events:
            self.take_photo()

        if self.mode == MODE_PORTAL:
            self.portal.update(hands, t)
            transition = (t - self._switched_at) / config.FILTER_TRANSITION
            current = self.filters[self.filter_index]
            previous = self.filters[self._prev_filter] if (
                self._prev_filter is not None and transition < 1.0) else None
            mask = None
            if self.portal.active and (current.needs_mask or (previous and previous.needs_mask)):
                mask = self.tracker.segment()   # must run before the frame is painted on
            self.portal.render(frame, t, current, previous, transition, mask)
        else:
            self.canvas.update(frame.shape, hands, g.poses, t)
            self.canvas.render(frame)

        # Saved photos / videos get the effects but not the skeleton or HUD.
        saving = self.capture.recording or self.capture.countdown_left(t) is not None
        self._clean = frame.copy() if saving else frame
        saved = self.capture.update(self._clean, t)
        if saved is not None:
            self.toast(f"Photo saved: {saved.name}")
        elif saving and not self.capture.recording and self.capture.countdown_left(t) is None:
            self.toast("Photo failed")

        hud.draw_hands(frame, hands)
        if self.mode == MODE_DRAW:
            self.canvas.draw_cursors(frame)

        if self.mode == MODE_PORTAL:
            items, index, swatches = [f.name for f in self.filters], self.filter_index, None
        else:
            palette = self.canvas.palette
            items = [name for name, _ in palette]
            index, swatches = self.canvas.color_index % len(palette), [c for _, c in palette]
        hud.draw(frame, hud.HudState(
            mode=self.mode, fps=self.fps, hands=len(hands), items=items, index=index, swatches=swatches,
            recording=self.capture.recording, rec_elapsed=self.capture.record_elapsed(t),
            countdown=self.capture.countdown_left(t), flash=self.capture.flash_alpha(t),
            show_help=self.show_help, peace_progress=g.peace_progress, peace_anchor=g.peace_anchor,
            toasts=self._toasts,
        ), t)
        return frame

    def close(self) -> None:
        self.capture.close()
        self.tracker.close()
