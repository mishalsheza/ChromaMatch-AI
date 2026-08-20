"""
test_ranked_writer.py — standalone test for the ranked Groq writer.

Tests get_ai_recommendations_ranked() in isolation using fake face_result
data for two scenarios: a close-call case (should hedge between two
seasons) and a clear-cut case (should commit confidently to one). No
real photo needed, no app.py involved.

Usage:
    python test_ranked_writer.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai.recommendation.season import get_ranked_season_recommendations
from ai.recommendation.groq_writer import get_ai_recommendations_ranked


# Load the full seasons data once, same as app.py will
REF_PATH = Path("ai/data/season_color_reference.json")
with open(REF_PATH, "r") as f:
    SEASONS_FULL_DATA = json.load(f).get("seasons", {})


# Fake foundation matches — just enough shape for the prompt to use
FAKE_FOUNDATIONS = [
    {"brand": "colourpop", "shade": "Medium Warm 20", "distance": 3.2},
    {"brand": "fenty", "shade": "260", "distance": 4.1},
    {"brand": "deciem", "shade": "2.1Y", "distance": 4.8},
]


def run_scenario(label: str, face_result: dict):
    print("\n" + "=" * 70)
    print(f"SCENARIO: {label}")
    print("=" * 70)
    print(f"Input face_result: {face_result}")

    ranked = get_ranked_season_recommendations(face_result, top_n=3)

    print(f"\nTop seasons:")
    for s in ranked["top_seasons"]:
        print(f"  {s['score_pct']:5.1f}%  {s['season_label']}")
    print(f"Margin: {ranked['margin_pct']:.1f}% — is_close_call: {ranked['is_close_call']}")

    result = get_ai_recommendations_ranked(
        face_result, FAKE_FOUNDATIONS, ranked, SEASONS_FULL_DATA
    )

    print("\n--- Groq output ---\n")
    print(result["summary"])
    print("\n" + "-" * 70)


if __name__ == "__main__":
    # Scenario 1: close call — mirrors the real True Spring / Bright Spring
    # ambiguity from tonight's debugging (clear clarity, medium contrast
    # after the confidence-gate downgrade).
    run_scenario(
        "Close call (True Spring vs Bright Spring)",
        {
            "depth": "medium",
            "undertone": "warm",
            "clarity": "clear",
            "contrast": "medium",
            "confidence": 0.78,
            "ita_degrees": 22.5,
        },
    )

    # Scenario 2: clear-cut — should commit confidently, no hedging language.
    run_scenario(
        "Clear-cut (Deep Winter)",
        {
            "depth": "deep",
            "undertone": "neutral_cool",
            "clarity": "clear",
            "contrast": "high",
            "confidence": 0.91,
            "ita_degrees": -35.0,
        },
    )
    run_scenario(
    "Genuinely ambiguous (olive undertone)",
    {
        "depth": "medium",
        "undertone": "olive",
        "clarity": "muted",
        "contrast": "medium",
        "confidence": 0.75,
        "ita_degrees": 15.0,
    },
)