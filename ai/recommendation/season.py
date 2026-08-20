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