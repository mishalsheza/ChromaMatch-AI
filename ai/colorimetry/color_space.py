"""
color_space.py — the single sRGB → CIELAB conversion used by the whole app.

Skin pixels (face_color.py) and foundation hex codes (match.py) both go
through srgb_to_lab(), so they are always compared on the same scale.

Standard: sRGB (IEC 61966-2-1), D65 white point, CIE 1976 L*a*b*.
"""

from __future__ import annotations

import numpy as np

# sRGB (linear) → XYZ, D65
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_D65_WHITE = np.array([0.95047, 1.0, 1.08883])

_EPSILON = 216 / 24389   # ≈ 0.008856, CIE threshold for the cube-root branch
_KAPPA = 24389 / 27      # ≈ 903.3


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Undo the sRGB gamma curve. `c` is in [0, 1].

    Piecewise per the sRGB standard:
      c / 12.92                     if c <= 0.04045
      ((c + 0.055) / 1.055) ** 2.4  otherwise
    """
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in 0–255 (shape (..., 3), RGB order) to CIELAB."""
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    xyz = srgb_to_linear(rgb) @ _SRGB_TO_XYZ.T
    t = xyz / _D65_WHITE
    f = np.where(t > _EPSILON, np.cbrt(t), (_KAPPA * t + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """Same as srgb_to_lab, for OpenCV's BGR channel order."""
    bgr = np.asarray(bgr)
    return srgb_to_lab(bgr[..., ::-1])


def hex_to_lab(hex_color: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' (or 'RGB') to an (L, a, b) tuple. Raises ValueError if malformed."""
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        raise ValueError(f"Not a hex colour: {hex_color!r}")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    L, a, b = srgb_to_lab(np.array(rgb))
    return float(L), float(a), float(b)
