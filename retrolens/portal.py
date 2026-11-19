"""The portal: a quad stretched between thumb + index fingertips of both hands."""
from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np

from . import config, fx, theme
from . import tracking as tk
from .filters import Filter, FilterContext
from .tracking import Hand


class Particles:
    """Fixed-size pool of sparks that drift away from the portal edge."""

    def __init__(self, capacity: int = 320):
        self.pos = np.zeros((capacity, 2), np.float32)
        self.vel = np.zeros((capacity, 2), np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.ttl = np.ones(capacity, np.float32)
        self._rng = np.random.default_rng()

    def emit(self, pos: np.ndarray, vel: np.ndarray) -> None:
        free = np.flatnonzero(self.life <= 0)[:len(pos)]
        n = len(free)
        if n == 0:
            return
        self.pos[free], self.vel[free] = pos[:n], vel[:n]
        self.ttl[free] = self._rng.uniform(0.35, 0.9, n)
        self.life[free] = self.ttl[free]

    def update(self, dt: float) -> None:
        self.pos += self.vel * dt
        self.life -= dt

    def draw(self, layer: np.ndarray, offset: np.ndarray, a, b) -> None:
        for i in np.flatnonzero(self.life > 0):
            k = float(self.life[i] / self.ttl[i])
            p = self.pos[i] - offset
            colour = fx.scale_color(fx.lerp_color(a, b, 1.0 - k), k)
            cv2.circle(layer, (int(p[0]), int(p[1])), 1 + int(2 * k), colour, -1, cv2.LINE_AA)


def _perimeter_point(poly: np.ndarray, u: float) -> np.ndarray:
    """Point at fraction u (0..1) along the closed polygon outline."""
    nxt = np.roll(poly, -1, axis=0)
    lengths = np.linalg.norm(nxt - poly, axis=1)
    target = (u % 1.0) * lengths.sum()
    for i, seg in enumerate(lengths):
        if target <= seg or i == len(lengths) - 1:
            return poly[i] + (nxt[i] - poly[i]) * (target / max(seg, 1e-6))
        target -= seg
    return poly[0]


class Portal:
    def __init__(self):
        self._poly: Optional[np.ndarray] = None   # last good quad, float32 (4, 2)
        self._last_seen = -1e9
        self._opened_at = 0.0
        self._last_t = 0.0
        self.active = False
        self.particles = Particles()
        self._rng = np.random.default_rng()

    def update(self, hands: List[Hand], t: float) -> None:
        quad = None
        if len(hands) >= 2:
            pts = np.array([h.pts[i] for h in hands[:2] for i in (tk.THUMB_TIP, tk.INDEX_TIP)], np.float32)
            centre = pts.mean(axis=0)
            # Sorting by angle around the centre always yields a non-self-intersecting quad.
            order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
            quad = pts[order]
            if cv2.contourArea(quad) < config.PORTAL_MIN_AREA:
                quad = None

        if quad is not None:
            if not self.active:
                self._opened_at = t
            self._poly, self._last_seen, self.active = quad, t, True
        elif t - self._last_seen > config.PORTAL_GRACE:
            self.active = False

    def render(self, frame: np.ndarray, t: float, current: Filter, previous: Optional[Filter],
               transition: float, person_mask: Optional[np.ndarray]) -> None:
        dt = min(max(t - self._last_t, 0.0), 0.1)
        self._last_t = t
        self.particles.update(dt)
        if not self.active or self._poly is None:
            return

        th = theme.get()
        fh, fw = frame.shape[:2]
        centre = self._poly.mean(axis=0)
        grow = fx.ease_out((t - self._opened_at) / config.PORTAL_OPEN_TIME)
        poly = centre + (self._poly - centre) * grow
        ipoly = np.round(poly).astype(np.int32)

        # --- filtered interior --------------------------------------------------
        x, y, bw, bh = cv2.boundingRect(ipoly)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + bw), min(fh, y + bh)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            roi = frame[y0:y1, x0:x1]
            ctx = FilterContext(t=t, x=x0, y=y0, frame_w=fw, frame_h=fh, person_mask=person_mask)
            out = current.apply(roi, ctx)
            if previous is not None and transition < 1.0:
                if out is roi or not out.flags.writeable:
                    out = out.copy()
                cut = int((x1 - x0) * fx.ease_out(transition))
                out[:, cut:] = previous.apply(roi, ctx)[:, cut:]
                lw = 2 if th.neon else 1
                out[:, max(0, cut - lw):cut + lw] = th.accent           # wipe line
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(mask, [ipoly - np.array([x0, y0])], 255, cv2.LINE_AA)
            np.copyto(roi, out, where=(mask > 127)[..., None])

        # --- frame ---------------------------------------------------------------
        if th.neon:
            self._neon_frame(frame, poly, ipoly, centre, (x, y, bw, bh), t, dt, th)
        else:
            cv2.polylines(frame, [ipoly], True, th.accent, 2, cv2.LINE_AA)
            for p in ipoly:
                cv2.circle(frame, (int(p[0]), int(p[1])), 4, th.accent, -1, cv2.LINE_AA)

    def _neon_frame(self, frame, poly, ipoly, centre, box, t, dt, th) -> None:
        """Glowing pulsing edge with sparks and energy dots running round the rim."""
        nxt = np.roll(poly, -1, axis=0)
        n = max(1, int(round(260 * dt)))
        edge = self._rng.integers(0, 4, n)
        k = self._rng.random((n, 1)).astype(np.float32)
        pos = poly[edge] + (nxt[edge] - poly[edge]) * k
        outward = pos - centre
        outward /= np.linalg.norm(outward, axis=1, keepdims=True) + 1e-6
        vel = outward * self._rng.uniform(25, 95, (n, 1)) + self._rng.normal(0, 30, (n, 2))
        self.particles.emit(pos.astype(np.float32), vel.astype(np.float32))

        pulse = 0.5 + 0.5 * math.sin(t * 2.2)
        edge_colour = fx.lerp_color(th.accent, th.accent2, pulse)
        spark_colour = fx.lerp_color(th.accent2, th.accent, pulse)
        x, y, bw, bh = box
        pad = 70

        def draw(layer: np.ndarray, off: np.ndarray) -> None:
            local = np.round(poly - off).astype(np.int32)
            cv2.polylines(layer, [local], True, edge_colour, 3, cv2.LINE_AA)
            self.particles.draw(layer, off, th.text, spark_colour)
            for i in range(6):
                p = _perimeter_point(poly, t * 0.22 + i / 6.0) - off
                cv2.circle(layer, (int(p[0]), int(p[1])), 4, th.text, -1, cv2.LINE_AA)
            for p in local:                                      # fingertip anchors
                cv2.circle(layer, (int(p[0]), int(p[1])), 7, spark_colour, 2, cv2.LINE_AA)
                cv2.circle(layer, (int(p[0]), int(p[1])), 3, th.text, -1, cv2.LINE_AA)

        fx.glow_region(frame, (x - pad, y - pad, x + bw + pad, y + bh + pad), draw, sigma=8, strength=1.3)
        cv2.polylines(frame, [ipoly], True, th.text, 1, cv2.LINE_AA)
