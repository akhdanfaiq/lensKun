"""Portal filters. Every filter takes a BGR region + context and returns a new BGR array
of the same shape (the input must not be modified in place)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np


@dataclass
class FilterContext:
    t: float                                  # seconds, for animation
    x: int                                    # region origin inside the full frame
    y: int
    frame_w: int
    frame_h: int
    person_mask: Optional[np.ndarray] = None  # full-frame float32 0..1, or None


@dataclass
class Filter:
    name: str
    fn: Callable[[np.ndarray, FilterContext], np.ndarray]
    needs_mask: bool = False

    def apply(self, roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
        out = self.fn(roi, ctx)
        if out.shape != roi.shape:  # safety net, should never trigger
            out = cv2.resize(out, (roi.shape[1], roi.shape[0]))
        return out


def _gray(roi: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def _gray3(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _row_mask(h: int, ctx: FilterContext, period: int, on: int, scroll: float = 0.0) -> np.ndarray:
    """Scanline row selector anchored to frame coordinates so lines don't swim with the portal."""
    return ((np.arange(h) + ctx.y + int(ctx.t * scroll)) % period) < on


# --- classics (from the original project, polished) ---------------------------
def mono(roi, ctx):
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(_gray(roi))
    return _gray3(gray)


_DUO_LUT = np.zeros((256, 3), np.uint8)
_ramp = np.linspace(0.0, 1.0, 256)[:, None]
_DUO_LUT[:] = (1 - _ramp) * np.array([120, 10, 90]) + _ramp * np.array([40, 190, 255])


def dual_tone(roi, ctx):
    return _DUO_LUT[_gray(roi)]


def pixelate(roi, ctx):
    h, w = roi.shape[:2]
    block = max(6, ctx.frame_w // 80)
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def invert(roi, ctx):
    return cv2.bitwise_not(roi)


_SEPIA = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]], np.float32)


def sepia(roi, ctx):
    return cv2.convertScaleAbs(cv2.transform(roi.astype(np.float32), _SEPIA))


def blur(roi, ctx):
    return cv2.GaussianBlur(roi, (0, 0), 14)


def thermal(roi, ctx):
    return cv2.applyColorMap(cv2.GaussianBlur(_gray(roi), (9, 9), 0), cv2.COLORMAP_JET)


def sketch(roi, ctx):
    gray = _gray(roi)
    soft = cv2.GaussianBlur(cv2.bitwise_not(gray), (21, 21), 0)
    return _gray3(cv2.divide(gray, cv2.bitwise_not(soft), scale=256))


def glitch(roi, ctx):
    h, w = roi.shape[:2]
    rng = np.random.default_rng(int(ctx.t * 14))  # changes ~14x per second for a stutter feel
    shift = max(3, int(w * 0.02 * (1 + 1.5 * rng.random())))
    out = roi.copy()
    if w > 2 * shift:
        out[:, :-shift, 2] = roi[:, shift:, 2]
        out[:, shift:, 0] = roi[:, :-shift, 0]
    for _ in range(int(rng.integers(2, 6))):
        band = int(rng.integers(4, max(6, h // 8)))
        y0 = int(rng.integers(0, max(1, h - band)))
        dx = int(rng.integers(-(w // 8) - 1, w // 8 + 2))
        out[y0:y0 + band] = np.roll(out[y0:y0 + band], dx, axis=1)
    return out


def neon(roi, ctx):
    h, w = roi.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(_gray(roi), (5, 5), 0), 50, 130)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    # Rainbow that drifts over time and across the frame.
    hsv = np.empty((1, w, 3), np.uint8)
    hsv[0, :, 0] = ((np.arange(w) + ctx.x) * 0.12 + ctx.t * 40) % 180
    hsv[0, :, 1:] = 255
    colour = np.ascontiguousarray(np.broadcast_to(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (h, w, 3)))
    lines = cv2.bitwise_and(colour, colour, mask=edges)
    glow = cv2.GaussianBlur(lines, (0, 0), 5)
    out = cv2.convertScaleAbs(roi, alpha=0.15)
    out = cv2.addWeighted(out, 1.0, glow, 1.8, 0)
    return cv2.add(out, lines)


class Galaxy:
    """Replaces the background behind the person with a slowly drifting star field."""

    def __init__(self):
        self._tile = None
        self._size = (0, 0)

    def _build(self, w: int, h: int) -> None:
        rng = np.random.default_rng(7)
        noise = rng.random((h // 60 + 2, w // 60 + 2, 3)).astype(np.float32)
        neb = np.clip(cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)
        bg = np.empty((h, w, 3), np.float32)
        bg[..., 0] = 35 + 170 * neb[..., 0] ** 2
        bg[..., 1] = 6 + 70 * neb[..., 1] ** 3
        bg[..., 2] = 25 + 170 * neb[..., 2] ** 2
        bg = bg.astype(np.uint8)
        n = int(w * h * 0.0012)
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        bg[ys, xs] = rng.integers(140, 256, (n, 1))
        big = np.zeros_like(bg)
        for _ in range(int(w * h * 0.00012)):
            centre = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            tint = (255, int(rng.integers(170, 256)), int(rng.integers(150, 256)))
            cv2.circle(big, centre, int(rng.integers(1, 3)), tint, -1, cv2.LINE_AA)
        bg = cv2.add(bg, cv2.add(big, cv2.GaussianBlur(big, (0, 0), 3)))
        # bg | mirrored bg | bg  -> seamless horizontal scroll with period 2w
        self._tile = np.hstack([bg, bg[:, ::-1], bg])
        self._size = (w, h)

    def __call__(self, roi, ctx):
        if self._size != (ctx.frame_w, ctx.frame_h):
            self._build(ctx.frame_w, ctx.frame_h)
        h, w = roi.shape[:2]
        off = int(ctx.t * 25) % (2 * ctx.frame_w) + ctx.x
        stars = self._tile[ctx.y:ctx.y + h, off:off + w]
        if ctx.person_mask is None:  # no segmentation model: fall back to a double exposure
            return cv2.addWeighted(roi, 0.55, stars, 0.9, 0)
        alpha = ctx.person_mask[ctx.y:ctx.y + h, ctx.x:ctx.x + w]
        alpha = np.clip((alpha - 0.3) / 0.4, 0, 1)[..., None]
        return (roi * alpha + stars * (1 - alpha)).astype(np.uint8)


# --- new in 2.0 ----------------------------------------------------------------
def vhs(roi, ctx):
    h, w = roi.shape[:2]
    s = max(2, ctx.frame_w // 320)
    out = roi.copy()
    if w > 2 * s:  # chroma bleed
        out[:, s:, 2] = roi[:, :-s, 2]
        out[:, :-s, 0] = roi[:, s:, 0]
    out = cv2.addWeighted(out, 0.72, _gray3(_gray(roi)), 0.22, 14)  # washed-out tape colours
    rows = _row_mask(h, ctx, 3, 1)
    out[rows] = out[rows] * 0.72
    # tracking bar rolling down the frame
    bar = int((ctx.t * 90) % (ctx.frame_h + 80)) - 40 - ctx.y
    y0, y1 = max(0, bar), min(h, bar + 16)
    if y1 > y0:
        out[y0:y1] = cv2.convertScaleAbs(np.roll(out[y0:y1], 3 * s, axis=1), alpha=1.0, beta=45)
    grain = np.random.randint(0, 26, (h, w), dtype=np.uint8)
    return cv2.add(out, cv2.merge([grain, grain, grain]))


class MatrixRain:
    GLYPHS = "01ABCDEFHKLMNPRSTXYZ#$%&*+<>"
    CELL = 16

    def __init__(self):
        self._layer = None
        self._t = 0.0

    def _reset(self, w: int, h: int) -> None:
        self._layer = np.zeros((h, w), np.uint8)
        self._cols = w // self.CELL + 1
        self._rows = h // self.CELL + 1
        rng = np.random.default_rng()
        self._pos = rng.uniform(-self._rows, self._rows, self._cols)
        self._speed = rng.uniform(8, 24, self._cols)
        self._rng = rng

    def _step(self, t: float) -> None:
        dt = min(max(t - self._t, 0.0), 0.1)
        self._t = t
        self._layer = cv2.convertScaleAbs(self._layer, alpha=0.90)  # trails fade out
        before = np.floor(self._pos).astype(int)
        self._pos += self._speed * dt
        after = np.floor(self._pos).astype(int)
        for col in np.flatnonzero(after != before):
            row = after[col]
            if 0 <= row < self._rows:
                glyph = self.GLYPHS[int(self._rng.integers(len(self.GLYPHS)))]
                org = (int(col) * self.CELL, (int(row) + 1) * self.CELL - 3)
                cv2.putText(self._layer, glyph, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
            if row > self._rows + self._rng.integers(0, 20):
                self._pos[col] = -self._rng.uniform(0, 12)
                self._speed[col] = self._rng.uniform(8, 24)

    def __call__(self, roi, ctx):
        if self._layer is None or self._layer.shape != (ctx.frame_h, ctx.frame_w):
            self._reset(ctx.frame_w, ctx.frame_h)
        self._step(ctx.t)
        h, w = roi.shape[:2]
        gray = _gray(roi)
        rain = self._layer[ctx.y:ctx.y + h, ctx.x:ctx.x + w]
        faint = cv2.convertScaleAbs(rain, alpha=0.45)
        return cv2.merge([
            cv2.add(cv2.convertScaleAbs(gray, alpha=0.12), faint),
            cv2.add(cv2.convertScaleAbs(gray, alpha=0.55), rain),
            cv2.add(cv2.convertScaleAbs(gray, alpha=0.08), faint),
        ])


def hologram(roi, ctx):
    h, w = roi.shape[:2]
    gray = _gray(roi)
    edges = cv2.Canny(gray, 60, 140)
    out = cv2.merge([
        cv2.add(gray, edges),
        cv2.add(cv2.convertScaleAbs(gray, alpha=0.85), edges),
        cv2.convertScaleAbs(gray, alpha=0.22),
    ])
    flicker = 0.92 + 0.08 * math.sin(ctx.t * 37.0)
    out = cv2.convertScaleAbs(out, alpha=flicker)
    rows = _row_mask(h, ctx, 4, 2, scroll=30)
    out[rows] = out[rows] * 0.6
    sweep = int((ctx.t * 220) % ctx.frame_h) - ctx.y  # bright scan bar
    y0, y1 = max(0, sweep), min(h, sweep + 26)
    if y1 > y0:
        out[y0:y1] = cv2.convertScaleAbs(out[y0:y1], alpha=1.0, beta=55)
    return out


def cartoon(roi, ctx):
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    small = cv2.bilateralFilter(small, 7, 70, 70)
    small = (small // 40) * 40 + 20  # posterise
    colour = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    ink = cv2.adaptiveThreshold(cv2.medianBlur(_gray(roi), 5), 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
    return cv2.bitwise_and(colour, colour, mask=ink)


def mirror(roi, ctx):
    out = roi.copy()
    w = roi.shape[1]
    half = w // 2
    if half > 0:
        out[:, w - half:] = roi[:, :half][:, ::-1]
    return out


def _cyber_lut() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256)
    x = 0.5 * x + 0.5 * (0.5 - 0.5 * np.cos(np.pi * x))   # gentle S-curve
    b = x + 0.14 * (1 - x) + 0.05 * x                      # teal shadows
    g = x + 0.05 * (1 - x) - 0.12 * x
    r = x - 0.04 * (1 - x) + 0.14 * x                      # magenta highlights
    return (np.clip(np.stack([b, g, r], axis=1), 0, 1) * 255).astype(np.uint8)


_CYBER_LUT = _cyber_lut()


def cyberpunk(roi, ctx):
    out = np.empty_like(roi)
    for c in range(3):
        out[..., c] = _CYBER_LUT[:, c][roi[..., c]]
    bright = cv2.threshold(_gray(roi), 185, 255, cv2.THRESH_BINARY)[1]
    bloom = cv2.GaussianBlur(cv2.merge([bright, bright // 3, bright]), (0, 0), 9)
    return cv2.addWeighted(out, 1.0, bloom, 0.55, 0)


_NV_GAMMA = (np.linspace(0, 1, 256) ** 0.55 * 255).astype(np.uint8)


def night_vision(roi, ctx):
    h, w = roi.shape[:2]
    gray = _NV_GAMMA[_gray(roi)]
    grain = np.random.randint(0, 40, (h, w), dtype=np.uint8)
    green = cv2.add(gray, grain)
    out = cv2.merge([cv2.convertScaleAbs(green, alpha=0.22), green, cv2.convertScaleAbs(green, alpha=0.18)])
    vy = 1.0 - 0.55 * np.linspace(-1, 1, h, dtype=np.float32) ** 2
    vx = 1.0 - 0.55 * np.linspace(-1, 1, w, dtype=np.float32) ** 2
    out = (out * np.outer(vy, vx)[..., None]).astype(np.uint8)
    rows = _row_mask(h, ctx, 3, 1)
    out[rows] = out[rows] * 0.8
    return out


def build_filters() -> List[Filter]:
    return [
        Filter("MONO", mono),
        Filter("DUAL-TONE", dual_tone),
        Filter("PIXELATE", pixelate),
        Filter("INVERT", invert),
        Filter("SEPIA", sepia),
        Filter("BLUR", blur),
        Filter("THERMAL", thermal),
        Filter("SKETCH", sketch),
        Filter("GLITCH", glitch),
        Filter("NEON", neon),
        Filter("GALAXY", Galaxy(), needs_mask=True),
        Filter("VHS", vhs),
        Filter("MATRIX", MatrixRain()),
        Filter("HOLOGRAM", hologram),
        Filter("CARTOON", cartoon),
        Filter("MIRROR", mirror),
        Filter("CYBERPUNK", cyberpunk),
        Filter("NIGHT VISION", night_vision),
    ]
