"""Visual themes. The default is set by config.THEME; press T in the app to switch.

To make your own theme, add a Theme(...) to THEMES below."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2

from . import config

Color = Tuple[int, int, int]   # BGR


@dataclass(frozen=True)
class Theme:
    name: str
    title: str                       # app name in the top-left panel
    font: int
    text: Color                      # main text
    muted: Color                     # secondary text
    accent: Color                    # portal edge, selected item, progress rings
    accent2: Color                   # second accent (the neon portal edge pulses between both)
    highlight: Color                 # peace-sign ring and notifications
    alert: Color                     # recording indicator
    panel_opacity: float             # how much panels darken what is behind them
    panel_border: Optional[Color]    # None = borderless panels
    brackets: bool                   # corner brackets on panels and screen edges
    neon: bool                       # glow, sparks, neon hand skeleton and neon brush strokes
    carousel_neighbors: bool         # show previous / next names next to the current one
    uppercase: bool                  # keep names in CAPITALS (False = "Night Vision")
    brush_colors: Sequence[Tuple[str, Color]]


MINIMAL = Theme(
    name="minimal",
    title="Retrolens",
    font=cv2.FONT_HERSHEY_SIMPLEX,
    text=(240, 240, 240),
    muted=(165, 165, 165),
    accent=(245, 245, 245),
    accent2=(245, 245, 245),
    highlight=(245, 245, 245),
    alert=(80, 80, 220),
    panel_opacity=0.45,
    panel_border=None,
    brackets=False,
    neon=False,
    carousel_neighbors=False,
    uppercase=False,
    brush_colors=[
        ("WHITE", (240, 240, 240)),
        ("GREY", (150, 150, 150)),
        ("RED", (90, 90, 215)),
        ("BLUE", (205, 155, 90)),
        ("GREEN", (130, 185, 110)),
        ("YELLOW", (115, 205, 235)),
    ],
)

CYBERPUNK = Theme(
    name="cyberpunk",
    title="RETROLENS",
    font=cv2.FONT_HERSHEY_DUPLEX,
    text=config.WHITE,
    muted=config.GREY,
    accent=config.CYAN,
    accent2=config.MAGENTA,
    highlight=config.YELLOW,
    alert=config.RED,
    panel_opacity=0.62,
    panel_border=config.CYAN,
    brackets=True,
    neon=True,
    carousel_neighbors=True,
    uppercase=True,
    brush_colors=[
        ("CYAN", config.CYAN),
        ("MAGENTA", config.MAGENTA),
        ("YELLOW", config.YELLOW),
        ("LIME", config.GREEN),
        ("ORANGE", config.ORANGE),
        ("WHITE", config.WHITE),
    ],
)

THEMES = [MINIMAL, CYBERPUNK]
_ACRONYMS = {"VHS"}

_index = next((i for i, th in enumerate(THEMES) if th.name == config.THEME), 0)


def get() -> Theme:
    return THEMES[_index]


def cycle() -> Theme:
    global _index
    _index = (_index + 1) % len(THEMES)
    return THEMES[_index]


def label(s: str) -> str:
    """Display form of a name: 'NIGHT VISION' stays as is, or becomes 'Night Vision'."""
    if get().uppercase:
        return s
    return " ".join(w if w in _ACRONYMS else w.capitalize() for w in s.split(" "))
