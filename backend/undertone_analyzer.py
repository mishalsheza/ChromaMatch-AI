import numpy as np
import math


# ─────────────────────────────────────────────
#  Accurate sRGB → CIE L*a*b* (no OpenCV rounding)
# ─────────────────────────────────────────────

def _linearize(c: float) -> float:
    """Inverse gamma correction for one sRGB channel (0–1 range)."""
    return (c / 12.92) if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    """
    Converts 8-bit sRGB → CIE L*a*b* (D65 illuminant, 2° observer).
    Uses the proper IEC 61966-2-1 pipeline instead of OpenCV's
    integer-quantised approximation, which loses ~0.5–1 unit of precision.
    """
    # 1. Normalise to [0, 1]
    r_lin = _linearize(r / 255.0)
    g_lin = _linearize(g / 255.0)
    b_lin = _linearize(b / 255.0)

    # 2. sRGB → XYZ (D65, IEC 61966-2-1 matrix)
    X = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    Y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    Z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # 3. Normalise by D65 white point
    X /= 0.95047
    Y /= 1.00000
    Z /= 1.08883

    # 4. XYZ → L*a*b* (CIE 1976)
    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    return L, a, b_val


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


# ─────────────────────────────────────────────
#  ITA skin-depth classifier
# ─────────────────────────────────────────────

def _ita_to_skin_type(ita: float) -> str:
    """
    Fitzpatrick-correlated ITA bands (Chardon et al. 1991).
    """
    if ita > 55:
        return "Very Light"
    elif ita > 41:
        return "Light"
    elif ita > 28:
        return "Intermediate"
    elif ita > 10:
        return "Tan"
    elif ita > -30:
        return "Brown"
    else:
        return "Dark"


def _l_to_cosmetic_depth(L: float) -> str:
    """
    Maps L* lightness to a five-band cosmetic depth label.
    Thresholds are empirically tighter than the original to avoid
    misclassifying medium-dark tones as Tan.
    """
    if L > 74:
        return "Fair"
    elif L > 60:
        return "Light"
    elif L > 44:
        return "Medium"
    elif L > 30:
        return "Tan"
    else:
        return "Deep"


# ─────────────────────────────────────────────
#  Undertone detection  (multi-signal approach)
# ─────────────────────────────────────────────

def _classify_undertone(L: float, a: float, b: float) -> str:
    """
    Uses three complementary signals so no single ratio can dominate.

    Signal 1 – b*/a* yellow-red ratio
        • High b*/a*  → yellow bias  → Warm
        • Low  b*/a*  → red/pink bias → Cool

    Signal 2 – Hue angle in a*b* plane
        • Hue > 60°   → more yellow  → Warm lean
        • Hue < 45°   → more red     → Cool lean

    Signal 3 – Absolute chroma gate
        • Chroma < 5  → near-achromatic; call Neutral regardless of ratio
          (avoids ratio blow-ups on very desaturated patches)

    Voting: each signal casts a vote; majority wins, with Neutral as tiebreak.
    """
    chroma = math.sqrt(a ** 2 + b ** 2)

    # --- Chroma gate (low-saturation override) ---
    if chroma < 5.0:
        return "Neutral"

    votes = []

    # --- Signal 1: b*/a* ratio ---
    # Guard against near-zero a* to avoid ratio explosion
    a_safe = a if abs(a) > 0.5 else (0.5 if a >= 0 else -0.5)
    ratio = b / a_safe

    if ratio > 1.65:
        votes.append("Warm")
    elif ratio < 1.10:
        votes.append("Cool")
    else:
        votes.append("Neutral")

    # --- Signal 2: Hue angle ---
    hue = math.degrees(math.atan2(b, a_safe)) % 360.0
    # Skin hues typically cluster 30°–70° in CIELAB
    if hue > 58:
        votes.append("Warm")
    elif hue < 44:
        votes.append("Cool")
    else:
        votes.append("Neutral")

    # --- Signal 3: Raw b* minus a* difference ---
    # Warm skin: b* noticeably exceeds a*
    # Cool skin: a* close to or exceeds b*
    diff = b - a
    if diff > 4.0:
        votes.append("Warm")
    elif diff < 0.5:
        votes.append("Cool")
    else:
        votes.append("Neutral")

    warm_count = votes.count("Warm")
    cool_count = votes.count("Cool")

    if warm_count > cool_count and warm_count >= 2:
        return "Warm"
    elif cool_count > warm_count and cool_count >= 2:
        return "Cool"
    else:
        return "Neutral"


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def analyze_skin_color(left_rgb: list, right_rgb: list) -> dict:
    """
    Full skin-colour analysis pipeline.

    Args:
        left_rgb  : [R, G, B] for the left cheek  (0–255 integers)
        right_rgb : [R, G, B] for the right cheek (0–255 integers)

    Returns:
        dict with keys: rgb, hex, lab, ita, skin_type,
                        cosmetic_depth, undertone, chroma
    """
    # 1. Average both cheeks
    avg_r = (left_rgb[0] + right_rgb[0]) / 2.0
    avg_g = (left_rgb[1] + right_rgb[1]) / 2.0
    avg_b = (left_rgb[2] + right_rgb[2]) / 2.0

    # 2. Accurate RGB → L*a*b*
    L, a, b = rgb_to_lab(int(avg_r), int(avg_g), int(avg_b))

    # 3. ITA°  (standard single-argument arctan form, b* never zero in practice)
    ita_rad = math.atan2(L - 50.0, b)
    ita_deg = math.degrees(ita_rad)

    # 4. Classifications
    skin_type      = _ita_to_skin_type(ita_deg)
    cosmetic_depth = _l_to_cosmetic_depth(L)
    undertone      = _classify_undertone(L, a, b)
    chroma         = math.sqrt(a ** 2 + b ** 2)

    return {
        "rgb": [int(avg_r), int(avg_g), int(avg_b)],
        "hex": rgb_to_hex(int(avg_r), int(avg_g), int(avg_b)),
        "lab": {
            "L": round(L, 2),
            "a": round(a, 2),
            "b": round(b, 2)
        },
        "ita":           round(ita_deg, 2),
        "skin_type":     skin_type,
        "cosmetic_depth": cosmetic_depth,
        "undertone":     undertone,
        "chroma":        round(chroma, 2),

        # Debug info (can strip in production)
        "_debug": {
            "votes_explanation": (
                "Multi-signal voting: b*/a* ratio + hue angle + b*-a* diff. "
                "Majority of 3 signals decides undertone."
            )
        }
    }