"""
match.py — foundation shade matching with CIEDE2000 (ΔE00).

Foundation hex codes are converted with the same sRGB → CIELAB function as the
skin pixels (ai/colorimetry/color_space.py), so both sides are on one scale.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ai.colorimetry.color_space import hex_to_lab
from ai.colorimetry.face_color import classify_undertone

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "foundations.json"


# ── CIEDE2000 ──

def delta_e_2000(lab1: Sequence[float], lab2: Sequence[float]) -> float:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal, 2005), kL = kC = kH = 1.

    Verified against all 34 pairs of Sharma's published test data
    (tests/fixtures/ciede2000_sharma_testdata.txt).
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C_bar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2.0
    G = 0.5 * (1 - math.sqrt(C_bar ** 7 / (C_bar ** 7 + 25 ** 7)))
    a1p, a2p = a1 * (1 + G), a2 * (1 + G)
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)

    def hue(b, ap):
        return 0.0 if (ap == 0 and b == 0) else math.degrees(math.atan2(b, ap)) % 360

    h1p, h2p = hue(b1, a1p), hue(b2, a2p)

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        if dhp > 180:
            dhp -= 360
        elif dhp < -180:
            dhp += 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2)

    Lp_bar = (L1 + L2) / 2
    Cp_bar = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hp_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp_bar = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hp_bar = (h1p + h2p + 360) / 2
    else:
        hp_bar = (h1p + h2p - 360) / 2

    T = (1
         - 0.17 * math.cos(math.radians(hp_bar - 30))
         + 0.24 * math.cos(math.radians(2 * hp_bar))
         + 0.32 * math.cos(math.radians(3 * hp_bar + 6))
         - 0.20 * math.cos(math.radians(4 * hp_bar - 63)))

    SL = 1 + 0.015 * (Lp_bar - 50) ** 2 / math.sqrt(20 + (Lp_bar - 50) ** 2)
    SC = 1 + 0.045 * Cp_bar
    SH = 1 + 0.015 * Cp_bar * T

    d_theta = 30 * math.exp(-(((hp_bar - 275) / 25) ** 2))
    RC = 2 * math.sqrt(Cp_bar ** 7 / (Cp_bar ** 7 + 25 ** 7))
    RT = -RC * math.sin(math.radians(2 * d_theta))

    return math.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )


# ── FOUNDATION DATABASE ──

def load_foundation_db(data_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load foundations.json and attach a precomputed LAB value to each shade.

    Shades with a missing or malformed hex code are skipped (not given a fake colour).
    """
    path = Path(data_path) if data_path else DEFAULT_DB_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shades, skipped = [], 0
    for item in data:
        try:
            item["lab"] = hex_to_lab(item["hex"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        shades.append(item)

    logger.info("Loaded %d foundation shades from %s (%d skipped: bad hex)", len(shades), path, skipped)
    return shades


def find_shade(foundations: List[Dict[str, Any]], brand: str, shade: str) -> Optional[Dict[str, Any]]:
    """Exact (case-insensitive) brand + shade lookup, or None."""
    brand_key, shade_key = (brand or "").strip().lower(), (shade or "").strip().lower()
    for item in foundations:
        if (item.get("brand") or "").strip().lower() == brand_key and \
                (item.get("shade") or "").strip().lower() == shade_key:
            return item
    return None


# ── MATCHING ──

def match_foundations(
    skin_lab: Sequence[float],
    foundations: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top_k shades with the smallest ΔE00 to the skin colour."""
    matches = []
    for foundation in foundations:
        lab = foundation.get("lab")
        if lab is None:
            continue
        matches.append({
            **foundation,
            "distance": delta_e_2000(skin_lab, lab),
            # Same undertone rule as the skin analysis (one definition app-wide)
            "undertone": classify_undertone(lab[1], lab[2]),
        })
    matches.sort(key=lambda m: m["distance"])
    return matches[:top_k]
