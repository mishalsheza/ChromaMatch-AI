"""
season.py — 12-season colour analysis by weighted scoring (deterministic).

Each of the 12 seasons in ai/data/season_color_reference.json is scored against
the measured profile (undertone, depth, clarity, contrast) and ranked. The
"classification_rules" table in that JSON is legacy and not used.

Known limitation: some input combinations tie at the top. The tie is broken by
the seasons' order in the JSON file, and it shows up as margin 0 / is_close_call=True.
"""

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "season_color_reference.json"


@lru_cache(maxsize=1)
def _load_reference_cached() -> Dict[str, Any]:
    with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_season_reference() -> Dict[str, Any]:
    """season_color_reference.json, read from disk once per process (callers get a copy)."""
    return copy.deepcopy(_load_reference_cached())


# ── HELPER: MAP FROM FACE COLOR TO SEASON INPUTS ──

def face_color_to_season_inputs(
    face_result: Dict[str, Any]
) -> Dict[str, str]:
    """
    Convert face color analysis output
    to season classification inputs.
    """

    depth = face_result.get(
        "depth",
        "medium"
    )

    undertone = face_result.get(
        "undertone",
        "neutral"
    )

    # The season rules split "neutral" into a warm-leaning and cool-leaning half,
    # using the same a*/b* direction as the undertone classifier.
    if undertone == "neutral":
        a = face_result.get("a", 0)
        b = face_result.get("b", 0)
        undertone = "neutral_warm" if b > a else "neutral_cool"

    clarity = face_result.get(
        "clarity",
        "medium"
    )

    if clarity not in ("clear", "muted"):
        clarity = "medium"

    contrast = face_result.get(
        "contrast",
        "medium"
    )

    if contrast not in [
        "low",
        "medium",
        "high"
    ]:
        contrast = "medium"

    return {
        "undertone": undertone,
        "depth": depth,
        "clarity": clarity,
        "contrast": contrast
    }


# ══════════════════════════════════════════════════════════════
# RANKED SCORING
#
# Every season is scored against the detected profile using its
# "skin_characteristics" in season_color_reference.json. The winner
# is the top score; the margin to #2 tells callers (and the narrative)
# whether this was a close call.
# ══════════════════════════════════════════════════════════════

# Compatibility scores between the detected undertone and each
# season's undertone characteristic. Diagonal = exact match = best.
UNDERTONE_COMPAT = {
    "cool":         {"cool": 3.0, "neutral_cool": 2.0, "neutral_warm": 0.0, "warm": 0.0, "olive": 0.0},
    "neutral_cool": {"cool": 2.0, "neutral_cool": 3.0, "neutral_warm": 1.0, "warm": 0.0, "olive": 0.5},
    "neutral_warm": {"cool": 0.0, "neutral_cool": 1.0, "neutral_warm": 3.0, "warm": 2.0, "olive": 1.5},
    "warm":         {"cool": 0.0, "neutral_cool": 0.0, "neutral_warm": 2.0, "warm": 3.0, "olive": 1.0},
    "olive":        {"cool": 0.0, "neutral_cool": 0.5, "neutral_warm": 1.5, "warm": 1.0, "olive": 3.0},
}

DEPTH_ORDER = ["light", "medium", "deep"]
CLOSE_CALL_MARGIN_THRESHOLD = 10.0


def _undertone_score(input_undertone: str, season_undertone_str: str) -> float:
    """Season undertone field can be a single value or slash-separated
    (e.g. deep_autumn's 'warm/olive') — score against each part, keep the best."""
    parts = [p.strip().lower() for p in season_undertone_str.split("/")]
    compat = UNDERTONE_COMPAT.get(input_undertone, {})
    return max((compat.get(p, 0.0) for p in parts), default=0.0)


def _bucket_set(season_value: str) -> set:
    """Handles season fields like 'light_to_medium' or 'medium_to_high'
    by splitting into the set of buckets they span."""
    s = season_value.lower()
    if "_to_" in s:
        return set(s.split("_to_"))
    return {s}


def _depth_score(input_depth: str, season_depth_str: str) -> float:
    buckets = _bucket_set(season_depth_str)
    if input_depth in buckets:
        return 2.0
    # partial credit if adjacent on the light-medium-deep scale
    try:
        input_idx = DEPTH_ORDER.index(input_depth)
        if any(abs(input_idx - DEPTH_ORDER.index(b)) == 1 for b in buckets if b in DEPTH_ORDER):
            return 0.5
    except ValueError:
        pass
    return 0.0


def _clarity_base(season_clarity_str: str) -> str:
    s = season_clarity_str.lower()
    if "clear" in s or "vivid" in s:
        return "clear"
    if "muted" in s or "soft" in s:
        return "muted"
    return "medium"


def _clarity_score(input_clarity: str, season_clarity_str: str) -> float:
    # Defensive: real pipeline only emits clear/muted/medium, but normalize
    # anything else down to "medium" rather than silently scoring zero.
    input_base = input_clarity if input_clarity in ("clear", "muted") else "medium"
    season_base = _clarity_base(season_clarity_str)
    if input_base == season_base:
        return 2.0
    if input_base == "medium":
        return 1.0
    return 0.0


def _contrast_score(input_contrast: str, season_contrast_str: str) -> float:
    buckets = _bucket_set(season_contrast_str)
    if input_contrast in buckets:
        return 2.0
    if input_contrast == "medium":
        return 1.0
    return 0.0


def score_all_seasons(
    undertone: str,
    depth: str,
    clarity: str = "medium",
    contrast: str = "medium",
) -> list[Dict[str, Any]]:
    """
    Score every season against the detected profile. Returns a list sorted best
    to worst, each with a 0-9 raw score, a 0-100 normalized score, and
    the field-by-field breakdown so you can see WHY it scored that way.
    """
    seasons = _load_reference_cached().get("seasons", {})
    MAX_SCORE = 9.0  # 3 (undertone) + 2 (depth) + 2 (clarity) + 2 (contrast)

    results = []
    for season_key, season_data in seasons.items():
        chars = season_data.get("skin_characteristics", {})

        u_score = _undertone_score(undertone, chars.get("undertone", ""))
        d_score = _depth_score(depth, chars.get("depth", ""))
        c_score = _clarity_score(clarity, chars.get("clarity", ""))
        ct_score = _contrast_score(contrast, chars.get("contrast", ""))

        total = u_score + d_score + c_score + ct_score

        # ── Penalize hard mismatches ──
        # A zero on undertone (the season-defining trait) is far worse than a
        # zero on a secondary trait like clarity — treat them differently
        # instead of flattening every mismatch to the same 50% cap.
        zero_count = sum(1 for s in (u_score, d_score, c_score, ct_score) if s == 0)

        if u_score == 0:
            total *= 0.25   # wrong temperature family — never a serious contender
        elif zero_count >= 2:
            total *= 0.55   # multiple secondary mismatches compound
        elif zero_count == 1:
            total *= 0.75   # single secondary mismatch, scaled down but ranking preserved

        results.append({
            "season_key": season_key,
            "season_label": season_data.get("label", season_key),
            "score": round(total, 2),
            "score_pct": round((total / MAX_SCORE) * 100, 1),
            "breakdown": {
                "undertone": u_score,
                "depth": d_score,
                "clarity": c_score,
                "contrast": ct_score,
            },
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def get_ranked_season_recommendations(
    face_result: Dict[str, Any],
    top_n: int = 3,
) -> Dict[str, Any]:
    """
    Convenience wrapper: takes raw face_result (same input shape as
    get_season_recommendations), returns the top N ranked seasons plus
    a computed margin between #1 and #2 so callers (e.g. groq_writer.py)
    can tell how confident the classification actually is.
    """
    inputs = face_color_to_season_inputs(face_result)
    ranked = score_all_seasons(**inputs)

    top = ranked[:top_n]
    margin = top[0]["score"] - top[1]["score"] if len(top) >= 2 else top[0]["score"]
    margin_pct = top[0]["score_pct"] - top[1]["score_pct"] if len(top) >= 2 else 100.0
    is_close_call = margin_pct < CLOSE_CALL_MARGIN_THRESHOLD

    return {
        "top_seasons": top,
        "margin": round(margin, 2),
        "margin_pct": round(margin_pct, 1),
        "is_close_call":  is_close_call,
        "winner_key": top[0]["season_key"],
    }
