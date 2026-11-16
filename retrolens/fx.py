"""Small drawing helpers shared by the portal, canvas and HUD."""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX


def lerp_color(a: Sequence[int], b: Sequence[int], k: float) -> Tuple[int, int, int]:
    k = min(1.0, max(0.0, k))
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def scale_color(c: Sequence[int], k: float) -> Tuple[int, int, int]:
    return tuple(int(min(255, max(0, v * k))) for v in c[:3])


def ease_out(k: float) -> float:
    k = min(1.0, max(0.0, k))
    return 1.0 - (1.0 - k) ** 3


def glow_region(frame: np.ndarray, box: Tuple[int, int, int, int],
                draw: Callable[[np.ndarray, np.ndarray], None],
                sigma: float = 7.0, strength: float = 1.0) -> None:
    """Additive neon glow. `draw(layer, offset)` paints onto a black layer covering `box`
    (x0, y0, x1, y1); the layer is blurred and added to the frame, then added again sharp."""
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(fw, int(box[2])), min(fh, int(box[3]))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    draw(layer, np.array([x0, y0], np.float32))
    halo = cv2.GaussianBlur(layer, (0, 0), sigma)
    region = frame[y0:y1, x0:x1]
    region = cv2.addWeighted(region, 1.0, halo, strength, 0)
    frame[y0:y1, x0:x1] = cv2.add(region, layer)


def text(img: np.ndarray, s: str, org: Tuple[int, int], scale: float = 0.6,
         color: Sequence[int] = (255, 255, 255), thick: int = 1, shadow: bool = True) -> None:
    org = (int(org[0]), int(org[1]))
    if shadow:
        cv2.putText(img, s, (org[0] + 2, org[1] + 2), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_size(s: str, scale: float = 0.6, thick: int = 1) -> Tuple[int, int]:
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    return w, h


def panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
          border: Sequence[int] = (255, 255, 0), opacity: float = 0.62) -> None:
    """Dark translucent panel with a thin border and bright corner ticks."""
    fh, fw = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(fw, int(x1)), min(fh, int(y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    img[y0:y1, x0:x1] = cv2.convertScaleAbs(img[y0:y1, x0:x1], alpha=1.0 - opacity, beta=8)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), scale_color(border, 0.45), 1, cv2.LINE_AA)
    brackets(img, x0, y0, x1 - 1, y1 - 1, border, length=min(12, (x1 - x0) // 3, (y1 - y0) // 3), thick=2)


def brackets(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
             color: Sequence[int], length: int = 24, thick: int = 2) -> None:
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thick, cv2.LINE_AA)
