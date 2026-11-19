"""Heads-up display: hand overlay, info and help panels, item selector and capture overlays.
Everything is drawn in the active theme (see theme.py)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import fx, theme
from .theme import Theme
from .tracking import HAND_CONNECTIONS, Hand

FINGERTIPS = (4, 8, 12, 16, 20)

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
    ("T", "switch theme"),
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


# --- hands ------------------------------------------------------------------------
def draw_hands(frame: np.ndarray, hands: List[Hand]) -> None:
    th = theme.get()
    for hand in hands:
        if th.neon:
            _neon_hand(frame, hand, th)
        else:
            _clean_hand(frame, hand, th)


def _neon_hand(frame: np.ndarray, hand: Hand, th: Theme) -> None:
    pts = hand.pts
    x0, y0 = pts.min(axis=0) - 24
    x1, y1 = pts.max(axis=0) + 24

    def draw(layer: np.ndarray, off: np.ndarray) -> None:
        local = np.round(pts - off).astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(layer, tuple(map(int, local[a])), tuple(map(int, local[b])), th.accent, 2, cv2.LINE_AA)
        for i, p in enumerate(local):
            tip = i in FINGERTIPS
            cv2.circle(layer, (int(p[0]), int(p[1])), 4 if tip else 2,
                       th.accent2 if tip else th.text, -1, cv2.LINE_AA)

    fx.glow_region(frame, (x0, y0, x1, y1), draw, sigma=5, strength=0.9)


def _clean_hand(frame: np.ndarray, hand: Hand, th: Theme) -> None:
    """Faint thin skeleton plus small fingertip dots."""
    fh, fw = frame.shape[:2]
    pts = hand.pts
    x0, y0 = (np.floor(pts.min(axis=0)) - 4).astype(int)
    x1, y1 = (np.ceil(pts.max(axis=0)) + 5).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)
    if x1 - x0 > 2 and y1 - y0 > 2:
        region = frame[y0:y1, x0:x1]
        overlay = region.copy()
        local = np.round(pts - (x0, y0)).astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(overlay, tuple(map(int, local[a])), tuple(map(int, local[b])), th.text, 1, cv2.LINE_AA)
        frame[y0:y1, x0:x1] = cv2.addWeighted(overlay, 0.35, region, 0.65, 0)
    for i in FINGERTIPS:
        p = (int(pts[i][0]), int(pts[i][1]))
        cv2.circle(frame, p, 4, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 3, th.text, -1, cv2.LINE_AA)


# --- panels -----------------------------------------------------------------------
def _info(frame: np.ndarray, st: HudState, s: float, th: Theme) -> None:
    x0, y0, pad = int(18 * s), int(18 * s), int(12 * s)
    if th.neon:
        title_scale, title_thick = 0.8 * s, 2
        status = f"MODE {st.mode}   FPS {st.fps:4.1f}   HANDS {st.hands}"
    else:
        title_scale, title_thick = 0.6 * s, 1
        status = f"{st.mode.capitalize()} mode   {st.fps:.0f} fps"
    tw, _ = fx.text_size(th.title, title_scale, title_thick)
    sw, _ = fx.text_size(status, 0.45 * s)
    width = max(tw + (int(90 * s) if th.neon else 0), sw) + 2 * pad
    height = int((70 if th.neon else 58) * s)
    fx.panel(frame, x0, y0, x0 + width, y0 + height)
    title_y = y0 + int((28 if th.neon else 24) * s)
    fx.text(frame, th.title, (x0 + pad, title_y), title_scale, th.accent if th.neon else th.text, title_thick)
    if th.neon:
        fx.text(frame, "// 2.0", (x0 + pad + tw + int(16 * s), title_y), 0.55 * s, th.accent2)
    fx.text(frame, status, (x0 + pad, y0 + height - int(16 * s)), 0.45 * s, th.text if th.neon else th.muted)


def _help(frame: np.ndarray, s: float, th: Theme) -> None:
    fw = frame.shape[1]
    line = int(21 * s)
    pw = int(400 * s)
    x0, y0 = fw - pw - int(18 * s), int(18 * s)
    fx.panel(frame, x0, y0, x0 + pw, y0 + line * len(HELP_LINES) + int(18 * s), border=th.accent2)
    y = y0 + int(24 * s)
    for left, right in HELP_LINES:
        if right is None:
            fx.text(frame, theme.label(left), (x0 + int(12 * s), y), 0.5 * s,
                    th.accent2 if th.neon else th.text)
        else:
            fx.text(frame, left, (x0 + int(12 * s), y), 0.44 * s, th.text if th.neon else th.muted)
            fx.text(frame, right, (x0 + int(232 * s), y), 0.44 * s, th.accent if th.neon else th.text)
        y += line


def _carousel(frame: np.ndarray, st: HudState, s: float, th: Theme) -> None:
    n = len(st.items)
    if n == 0:
        return
    if th.carousel_neighbors:
        _carousel_full(frame, st, s, th)
    else:
        _carousel_compact(frame, st, s, th)


def _carousel_compact(frame: np.ndarray, st: HudState, s: float, th: Theme) -> None:
    """Small pill: [swatch] Name   3/18"""
    fh, fw = frame.shape[:2]
    name = theme.label(st.items[st.index])
    counter = f"{st.index + 1}/{len(st.items)}"
    scale = 0.65 * s
    tw, tht = fx.text_size(name, scale)
    cw, _ = fx.text_size(counter, 0.45 * s)
    sw = int(14 * s) if st.swatches else 0
    gap, pad = int(14 * s), int(18 * s)
    pw = pad + (sw + gap if sw else 0) + tw + gap + cw + pad
    ph = int(42 * s)
    x0, y0 = (fw - pw) // 2, fh - ph - int(20 * s)
    fx.panel(frame, x0, y0, x0 + pw, y0 + ph)
    cy = y0 + ph // 2
    x = x0 + pad
    if sw:
        cv2.circle(frame, (x + sw // 2, cy), sw // 2 + 1, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(frame, (x + sw // 2, cy), sw // 2, st.swatches[st.index], -1, cv2.LINE_AA)
        x += sw + gap
    fx.text(frame, name, (x, cy + tht // 2), scale, th.text)
    fx.text(frame, counter, (x + tw + gap, cy + tht // 2), 0.45 * s, th.muted)


def _carousel_full(frame: np.ndarray, st: HudState, s: float, th: Theme) -> None:
    """Wide bar: < previous   CURRENT   next >, with position ticks."""
    fh, fw = frame.shape[:2]
    n = len(st.items)
    pw, ph = int(560 * s), int(58 * s)
    x0, y0 = (fw - pw) // 2, fh - ph - int(18 * s)
    fx.panel(frame, x0, y0, x0 + pw, y0 + ph, border=th.accent2)

    cur = theme.label(st.items[st.index])
    cur_scale = 0.85 * s
    cw, ch = fx.text_size(cur, cur_scale, 2)
    cy = y0 + (ph + ch) // 2 - int(4 * s)
    colour = st.swatches[st.index] if st.swatches else th.accent
    fx.text(frame, cur, ((fw - cw) // 2, cy), cur_scale, colour, 2)

    side = 0.48 * s
    prev = theme.label(st.items[(st.index - 1) % n])
    nxt = theme.label(st.items[(st.index + 1) % n])
    fx.text(frame, "< " + prev, (x0 + int(14 * s), cy - int(3 * s)), side, th.muted)
    nw, _ = fx.text_size(nxt + " >", side)
    fx.text(frame, nxt + " >", (x0 + pw - nw - int(14 * s), cy - int(3 * s)), side, th.muted)

    tick = max(4, int((pw - 40 * s) / n) - 3)
    total = n * (tick + 3) - 3
    tx = (fw - total) // 2
    ty = y0 + ph - int(8 * s)
    for i in range(n):
        c = st.swatches[i] if st.swatches else th.accent
        if i != st.index:
            c = fx.scale_color(c, 0.35)
        cv2.rectangle(frame, (tx, ty), (tx + tick, ty + max(2, int(3 * s))), c, -1)
        tx += tick + 3


def _toast_text(message: str, th: Theme) -> str:
    if not th.uppercase:
        return message
    head, sep, tail = message.partition(": ")     # keep file names in their real case
    return head.upper() + sep + tail


# --- main entry ---------------------------------------------------------------------
def draw(frame: np.ndarray, st: HudState, t: float) -> None:
    th = theme.get()
    fh, fw = frame.shape[:2]
    s = fh / 720.0

    if th.brackets:
        fx.brackets(frame, int(8 * s), int(8 * s), fw - int(8 * s) - 1, fh - int(8 * s) - 1,
                    th.accent, length=int(34 * s), thick=2)
    _info(frame, st, s, th)
    if st.show_help:
        _help(frame, s, th)
    else:
        hint = theme.label("[H] HELP")
        hw, _ = fx.text_size(hint, 0.45 * s)
        fx.text(frame, hint, (fw - hw - int(24 * s), int(40 * s)), 0.45 * s, th.muted)
    _carousel(frame, st, s, th)

    if st.recording:
        rx, ry = int(24 * s), int(118 * s)
        if int(t * 2) % 2 == 0:
            cv2.circle(frame, (rx + int(8 * s), ry - int(6 * s)), int(7 * s), th.alert, -1, cv2.LINE_AA)
        mins, secs = divmod(int(st.rec_elapsed), 60)
        fx.text(frame, f"REC {mins:02d}:{secs:02d}", (rx + int(24 * s), ry), 0.55 * s, th.alert,
                2 if th.neon else 1)

    if st.peace_progress > 0 and st.peace_anchor is not None:
        r = int((34 if th.neon else 26) * s)
        cv2.circle(frame, st.peace_anchor, r, fx.scale_color(th.highlight, 0.3), 2, cv2.LINE_AA)
        cv2.ellipse(frame, st.peace_anchor, (r, r), -90, 0, 360 * st.peace_progress,
                    th.highlight, 4 if th.neon else 2, cv2.LINE_AA)

    if st.countdown is not None:
        num = str(max(1, math.ceil(st.countdown)))
        frac = st.countdown - math.floor(st.countdown)        # 1 -> 0 within each second
        if th.neon:
            scale, thick, ring = (4.0 + 1.5 * frac) * s, 6, 6
        else:
            scale, thick, ring = (3.4 + 0.4 * frac) * s, 3, 2
        tw, tht = fx.text_size(num, scale, thick)
        centre = (fw // 2, fh // 2)
        r = int((120 if th.neon else 100) * s)
        cv2.circle(frame, centre, r, fx.scale_color(th.accent, 0.35), 2 if not th.neon else 3, cv2.LINE_AA)
        cv2.ellipse(frame, centre, (r, r), -90, 0, 360 * (1 - frac), th.accent, ring, cv2.LINE_AA)
        fx.text(frame, num, (centre[0] - tw // 2, centre[1] + tht // 2), scale, th.text, thick)

    y = int(150 * s) if st.recording else int(120 * s)
    for message, expires in st.toasts:
        if expires <= t:
            continue
        message = _toast_text(message, th)
        tw, tht = fx.text_size(message, 0.55 * s)
        x = (fw - tw) // 2
        fx.panel(frame, x - int(14 * s), y - tht - int(10 * s), x + tw + int(14 * s), y + int(10 * s),
                 border=th.highlight)
        fx.text(frame, message, (x, y), 0.55 * s, th.highlight if th.neon else th.text)
        y += tht + int(30 * s)

    if st.flash > 0:
        a = st.flash * 0.9
        cv2.convertScaleAbs(frame, dst=frame, alpha=1.0 - a, beta=255 * a)
