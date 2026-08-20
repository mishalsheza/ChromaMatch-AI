"""
season.py — 12-season color analysis classification

Uses the unified season_color_reference.json
(classification_rules + full rich per-season data)
for deterministic classification.
"""

import json
import os
from typing import Dict, Any


# ── CLASSIFICATION FUNCTION ──

def classify_season(
    undertone: str,
    depth: str,
    clarity: str = "medium",
    contrast: str = "medium"
) -> Dict[str, Any]:
    """
    Classify skin profile into one of 12 seasons.

    Args:
        undertone:
            "warm" | "cool" | "neutral_warm" |
            "neutral_cool" | "olive"

        depth:
            "light" | "medium" | "deep"

        clarity:
            "clear" | "muted" | "soft" | "medium"

        contrast:
            "low" | "medium" | "high"

    Returns:
        Season data with jewelry, lipstick, blush recommendations,
        plus the full rich season entry under "raw_data".
    """

    ref_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "season_color_reference.json"
    )

    ref_path = os.path.abspath(ref_path)

    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Season reference not found: {ref_path}"
        )

    with open(ref_path, "r") as f:
        data = json.load(f)

    # Step 1: Find matching season using classification rules
    rules = data.get("classification_rules", [])

    matched_season = None

    for rule in rules:
        conditions = rule.get("if", {})
        match = True

        for key, values in conditions.items():

            if key == "undertone":
                if undertone not in values:
                    match = False
                    break

            elif key == "depth":
                if depth not in values:
                    match = False
                    break

            elif key == "clarity":
                if clarity not in values:
                    match = False
                    break

            elif key == "contrast":
                if contrast not in values:
                    match = False
                    break

        if match:
            matched_season = rule.get("season")
            break

    # Fallback if no rule matched
    if matched_season is None:
        matched_season = "true_summer"

    # Step 2: Get the full season entry
    seasons = data.get("seasons", {})

    season_data = seasons.get(
        matched_season,
        {}
    )

    if not season_data:
        print(
            f"⚠️ classify_season: matched key "
            f"'{matched_season}' not found in "
            f"season_color_reference.json. "
            f"Available keys: {list(seasons.keys())}"
        )

    # Step 3: Build response
    # season_key is the authoritative exact key.
    # Other modules should use this field instead of
    # re-deriving the season key from season_label.

    return {
        "season_key": matched_season,

        "season_label": season_data.get(
            "label",
            matched_season
        ),

        "family": season_data.get(
            "family",
            "neutral"
        ),

        "profile": season_data.get(
            "profile",
            {}
        ),

        "jewelry": season_data.get(
            "jewelry_simple",
            ["silver", "gold"]
        ),

        "lipstick_family": season_data.get(
            "lipstick_family",
            {
                "name": "neutral",
                "hexes": ["#C0C0C0"]
            }
        ),

        "blush_family": season_data.get(
            "blush_family",
            {
                "name": "neutral",
                "hexes": ["#D0B0A0"]
            }
        ),

        "raw_data": season_data
    }


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

    if undertone == "warm":
        undertone = "warm"

    elif undertone == "cool":
        undertone = "cool"

    elif undertone == "neutral":
        a = face_result.get("a", 0)
        b = face_result.get("b", 0)

        if b > a:
            undertone = "neutral_warm"
        else:
            undertone = "neutral_cool"

    elif undertone == "olive":
        undertone = "olive"

    clarity = face_result.get(
        "clarity",
        "medium"
    )

    if clarity == "clear":
        clarity = "clear"

    elif clarity == "muted":
        clarity = "muted"

    else:
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


# ── CONVENIENCE FUNCTION ──

def get_season_recommendations(
    face_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get full season recommendations
    from face color analysis.
    """

    inputs = face_color_to_season_inputs(
        face_result
    )

    return classify_season(
        undertone=inputs["undertone"],
        depth=inputs["depth"],
        clarity=inputs["clarity"],
        contrast=inputs["contrast"]
    )


# ── TEST FUNCTION ──

if __name__ == "__main__":

    test_cases = [
        {
            "depth": "light",
            "undertone": "warm",
            "clarity": "clear",
            "contrast": "high"
        },
        {
            "depth": "medium",
            "undertone": "cool",
            "clarity": "muted",
            "contrast": "low"
        },
        {
            "depth": "deep",
            "undertone": "neutral_cool",
            "clarity": "clear",
            "contrast": "high"
        },
        {
            "depth": "medium",
            "undertone": "olive",
            "clarity": "muted",
            "contrast": "low"
        }
    ]

    print("=" * 60)
    print("Season Classification Tests")
    print("=" * 60)

    for test in test_cases:

        result = classify_season(**test)

        print(f"\n📊 Input: {test}")

        print(
            f" Season key: "
            f"{result['season_key']}"
        )

        print(
            f" Season label: "
            f"{result['season_label']}"
        )

        print(
            f" Family: "
            f"{result['family']}"
        )

        print(
            f" Jewelry: "
            f"{result['jewelry']}"
        )

        print(
            f" Lipstick: "
            f"{result['lipstick_family']['name']}"
        )

        print(
            f" Blush: "
            f"{result['blush_family']['name']}"
        )

        print(
            f" Raw data has "
            f"{len(result['raw_data'])} "
            f"top-level fields "
            f"(should include lipstick.best, "
            f"dress_occasions, etc.)"
        )
# ══════════════════════════════════════════════════════════════
# RANKED SCORING — additive, does not replace classify_season()
#
# classify_season() above stays exactly as-is (first-match-wins,
# proven correct). This adds a SEPARATE scoring function that ranks
# all 12 seasons by how well they fit the input, so Groq can see
# "this was close" instead of only ever seeing one confident winner.
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
    Score every season against the detected profile, instead of stopping
    at the first classification_rules match. Returns a list sorted best
    to worst, each with a 0-9 raw score, a 0-100 normalized score, and
    the field-by-field breakdown so you can see WHY it scored that way.
    """
    ref_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "season_color_reference.json"
    ))
    with open(ref_path, "r") as f:
        data = json.load(f)

    seasons = data.get("seasons", {})
    MAX_SCORE = 9.0  # 3 (undertone) + 2 (depth) + 2 (clarity) + 2 (contrast)

    results = []
    for season_key, season_data in seasons.items():
        chars = season_data.get("skin_characteristics", {})

        u_score = _undertone_score(undertone, chars.get("undertone", ""))
        d_score = _depth_score(depth, chars.get("depth", ""))
        c_score = _clarity_score(clarity, chars.get("clarity", ""))
        ct_score = _contrast_score(contrast, chars.get("contrast", ""))

        total = u_score + d_score + c_score + ct_score

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
    TOP_SCORE_CONFIDENCE_THRESHOLD = 92.0
    is_close_call = top[0]["score_pct"] < TOP_SCORE_CONFIDENCE_THRESHOLD


    return {
        "top_seasons": top,
        "margin": round(margin, 2),
        "margin_pct": round(margin_pct, 1),
        "is_close_call":  is_close_call,
        "winner_key": top[0]["season_key"],
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Ranked Scoring Tests")
    print("=" * 60)

    test_cases = [
        {"depth": "medium", "undertone": "warm", "clarity": "clear", "contrast": "high"},
        {"depth": "medium", "undertone": "warm", "clarity": "clear", "contrast": "medium"},
        {"depth": "light", "undertone": "cool", "clarity": "soft", "contrast": "low"},
    ]

    for test in test_cases:
        ranked = score_all_seasons(**test)
        print(f"\n📊 Input: {test}")
        for r in ranked[:4]:
            print(f"  {r['score_pct']:5.1f}%  {r['season_label']:15s}  "
                  f"(u={r['breakdown']['undertone']}, d={r['breakdown']['depth']}, "
                  f"c={r['breakdown']['clarity']}, ct={r['breakdown']['contrast']})")