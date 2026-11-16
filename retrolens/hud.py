"""Cyberpunk heads-up display: neon hand skeleton, info panels, carousel, overlays."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config, fx
from .tracking import HAND_CONNECTIONS, Hand

HELP_LINES = [
    ("GESTURES", None),
    ("Thumb + index, both hands", "open portal"),
    ("Thumb + pinky touch", "next filter / colour"),
    ("Index tips together", "next filter / colour"),
    ("Peace sign (hold)", "photo countdown"),
    ("Point (draw mode)", "draw"),
    ("Open palm (draw mode)", "erase"),
    ("KEYS", None),
    ("N / P", "next / previous"),
    ("D", "portal <-> draw mode"),
    ("SPACE", "photo"),
    ("R", "record video"),
    ("C", "clear drawing"),
    ("H", "toggle this help"),
    ("Q / ESC", "quit"),
]


@dataclass
class HudState:
    mode: str = "PORTAL"
    fps: float = 0.0
    hands: int = 0
    items: Sequence[str] = ()                 # filter names (portal) or colour names (draw)
    index: int = 0
    swatches: Optional[Sequence[Tuple[int, int, int]]] = None
    recording: bool = False
    rec_elapsed: float = 0.0
    countdown: Optional[float] = None
    flash: float = 0.0
    show_help: bool = True
    peace_progress: float = 0.0
    peace_anchor: Optional[Tuple[int, int]] = None
    toasts: List[Tuple[str, float]] = field(default_factory=list)   # (message, expires_at)


def draw_hands(frame: np.ndarray, hands: List[Hand]) -> None:
    for hand in hands:
        pts = hand.pts
        x0, y0 = pts.min(axis=0) - 24
        x1, y1 = pts.max(axis=0) + 24

        def draw(layer: np.ndarray, off: np.ndarray, pts=pts) -> None:
            local = np.round(pts - off).astype(np.int32)
            for a, b in HAND_CONNECTIONS:
                cv2.line(layer, tuple(map(int, local[a])), tuple(map(int, local[b])), config.CYAN, 2, cv2.LINE_AA)
            for i, p in enumerate(local):
                tip = i in (4, 8, 12, 16, 20)
                cv2.circle(layer, (int(p[0]), int(p[1])), 4 if tip else 2,
                           config.MAGENTA if tip else config.WHITE, -1, cv2.LINE_AA)

        fx.glow_region(frame, (x0, y0, x1, y1), draw, sigma=5, strength=0.9)


def _carousel(frame: np.ndarray, st: HudState, s: float) -> None:
    fh, fw = frame.shape[:2]
    n = len(st.items)
    if n == 0:
        return
    pw, ph = int(560 * s), int(58 * s)
    x0, y0 = (fw - pw) // 2, fh - ph - int(18 * s)
    fx.panel(frame, x0, y0, x0 + pw, y0 + ph, border=config.MAGENTA)

    cur = st.items[st.index]
    cur_scale = 0.85 * s
    cw, ch = fx.text_size(cur, cur_scale, 2)
    cy = y0 + (ph + ch) // 2 - int(4 * s)
    accent = st.swatches[st.index] if st.swatches else config.CYAN
    fx.text(frame, cur, ((fw - cw) // 2, cy), cur_scale, accent, 2)

    side = 0.48 * s
    prev, nxt = st.items[(st.index - 1) % n], st.items[(st.index + 1) % n]
    fx.text(frame, "< " + prev, (x0 + int(14 * s), cy - int(3 * s)), side, config.GREY)
    nw, _ = fx.text_size(nxt + " >", side)
    fx.text(frame, nxt + " >", (x0 + pw - nw - int(14 * s), cy - int(3 * s)), side, config.GREY)

    # position ticks along the bottom edge of the panel
    tick = max(4, int((pw - 40 * s) / n) - 3)
    total = n * (tick + 3) - 3
    tx = (fw - total) // 2
    ty = y0 + ph - int(8 * s)
    for i in range(n):
        colour = st.swatches[i] if st.swatches else config.CYAN
        if i != st.index:
            colour = fx.scale_color(colour, 0.35)
        cv2.rectangle(frame, (tx, ty), (tx + tick, ty + max(2, int(3 * s))), colour, -1)
        tx += tick + 3


def _info(frame: np.ndarray, st: HudState, s: float) -> None:
    x0, y0 = int(18 * s), int(18 * s)
    fx.panel(frame, x0, y0, x0 + int(330 * s), y0 + int(70 * s))
    fx.text(frame, "RETROLENS", (x0 + int(12 * s), y0 + int(28 * s)), 0.8 * s, config.CYAN, 2)
    fx.text(frame, "// 2.0", (x0 + int(172 * s), y0 + int(28 * s)), 0.55 * s, config.MAGENTA)
    status = f"MODE {st.mode}   FPS {st.fps:4.1f}   HANDS {st.hands}"
    fx.text(frame, status, (x0 + int(12 * s), y0 + int(54 * s)), 0.45 * s, config.WHITE)


def _help(frame: np.ndarray, st: HudState, s: float) -> None:
    fw = frame.shape[1]
    line = int(21 * s)
    pw = int(400 * s)
    x0, y0 = fw - pw - int(18 * s), int(18 * s)
    fx.panel(frame, x0, y0, x0 + pw, y0 + line * len(HELP_LINES) + int(18 * s), border=config.MAGENTA)
    y = y0 + int(24 * s)
    for left, right in HELP_LINES:
        if right is None:
            fx.text(frame, left, (x0 + int(12 * s), y), 0.5 * s, config.MAGENTA)
        else:
            fx.text(frame, left, (x0 + int(12 * s), y), 0.44 * s, config.WHITE)
            fx.text(frame, right, (x0 + int(232 * s), y), 0.44 * s, config.CYAN)
        y += line


def draw(frame: np.ndarray, st: HudState, t: float) -> None:
    fh, fw = frame.shape[:2]
    s = fh / 720.0

    fx.brackets(frame, int(8 * s), int(8 * s), fw - int(8 * s) - 1, fh - int(8 * s) - 1,
                config.CYAN, length=int(34 * s), thick=2)
    _info(frame, st, s)
    if st.show_help:
        _help(frame, st, s)
    else:
        hint = "[H] HELP"
        hw, _ = fx.text_size(hint, 0.45 * s)
        fx.text(frame, hint, (fw - hw - int(24 * s), int(40 * s)), 0.45 * s, config.GREY)
    _carousel(frame, st, s)

    if st.recording:
        rx, ry = int(24 * s), int(118 * s)
        if int(t * 2) % 2 == 0:
            cv2.circle(frame, (rx + int(8 * s), ry - int(6 * s)), int(8 * s), config.RED, -1, cv2.LINE_AA)
        mins, secs = divmod(int(st.rec_elapsed), 60)
        fx.text(frame, f"REC {mins:02d}:{secs:02d}", (rx + int(26 * s), ry), 0.6 * s, config.RED, 2)

    if st.peace_progress > 0 and st.peace_anchor is not None:
        r = int(34 * s)
        cv2.circle(frame, st.peace_anchor, r, fx.scale_color(config.YELLOW, 0.3), 2, cv2.LINE_AA)
        cv2.ellipse(frame, st.peace_anchor, (r, r), -90, 0, 360 * st.peace_progress,
                    config.YELLOW, 4, cv2.LINE_AA)

    if st.countdown is not None:
        num = str(max(1, math.ceil(st.countdown)))
        frac = st.countdown - math.floor(st.countdown)        # 1 -> 0 within each second
        scale = (4.0 + 1.5 * frac) * s
        tw, th = fx.text_size(num, scale, 6)
        centre = (fw // 2, fh // 2)
        r = int(120 * s)
        cv2.circle(frame, centre, r, fx.scale_color(config.CYAN, 0.35), 3, cv2.LINE_AA)
        cv2.ellipse(frame, centre, (r, r), -90, 0, 360 * (1 - frac), config.CYAN, 6, cv2.LINE_AA)
        fx.text(frame, num, (centre[0] - tw // 2, centre[1] + th // 2), scale, config.WHITE, 6)

    y = int(120 * s) if not st.recording else int(150 * s)
    for message, expires in st.toasts:
        if expires <= t:
            continue
        tw, th = fx.text_size(message, 0.55 * s)
        x = (fw - tw) // 2
        fx.panel(frame, x - int(14 * s), y - th - int(10 * s), x + tw + int(14 * s), y + int(10 * s),
                 border=config.YELLOW)
        fx.text(frame, message, (x, y), 0.55 * s, config.YELLOW)
        y += th + int(30 * s)

    if st.flash > 0:
        a = st.flash * 0.9
        cv2.convertScaleAbs(frame, dst=frame, alpha=1.0 - a, beta=255 * a)
