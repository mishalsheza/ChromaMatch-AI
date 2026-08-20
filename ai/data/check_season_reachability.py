"""
check_season_reachability.py — Verify every season in season_color_reference.json
is actually reachable by season.py's classification_rules.

Run this after any change to classification_rules or to face_color.py's
output fields. It brute-forces every combination of the four classifier
inputs and confirms each season key gets hit at least once, using the
SAME matching logic as season.py (copied inline so this stays a pure
static check with no import-time side effects).

Usage:
    python check_season_reachability.py
Exit code 0 = all seasons reachable. Exit code 1 = one or more unreachable.
"""

import itertools
import json
import os
import sys
from typing import Dict, Any, List

UNDERTONES = ["warm", "cool", "neutral_warm", "neutral_cool", "olive"]
DEPTHS = ["light", "medium", "deep"]
CLARITIES = ["clear", "muted", "medium"]
CONTRASTS = ["low", "medium", "high"]


def match_rule(rule: Dict[str, Any], undertone: str, depth: str, clarity: str, contrast: str) -> bool:
    """Mirrors season.py's classify_season matching logic exactly."""
    conditions = rule.get("if", {})
    for key, values in conditions.items():
        if key == "undertone":
            if undertone not in values:
                return False
        elif key == "depth":
            if depth not in values:
                return False
        elif key == "clarity":
            if clarity not in values and clarity != "medium":
                return False
        elif key == "contrast":
            if contrast not in values:
                return False
    return True


def classify(rules: List[Dict[str, Any]], undertone: str, depth: str, clarity: str, contrast: str) -> str:
    for rule in rules:
        if match_rule(rule, undertone, depth, clarity, contrast):
            return rule.get("season")
    return "true_summer"  # fallback, matches season.py


def main() -> int:
    ref_path = os.path.join(os.path.dirname(__file__), "season_color_reference.json")
    if not os.path.exists(ref_path):
        # allow passing a path as first arg for convenience
        if len(sys.argv) > 1:
            ref_path = sys.argv[1]
        else:
            print(f"❌ Could not find season_color_reference.json at {ref_path}")
            print("   Pass the path explicitly: python check_season_reachability.py <path>")
            return 1

    with open(ref_path, "r") as f:
        data = json.load(f)

    rules = data.get("classification_rules", [])
    all_seasons = set(data.get("seasons", {}).keys())

    if not rules:
        print("❌ No classification_rules found in reference file.")
        return 1
    if not all_seasons:
        print("❌ No seasons found in reference file.")
        return 1

    reached: Dict[str, int] = {season: 0 for season in all_seasons}
    reached_by: Dict[str, List[str]] = {season: [] for season in all_seasons}
    unknown_hits = 0
    total = 0

    for undertone, depth, clarity, contrast in itertools.product(
        UNDERTONES, DEPTHS, CLARITIES, CONTRASTS
    ):
        total += 1
        season = classify(rules, undertone, depth, clarity, contrast)
        if season not in reached:
            unknown_hits += 1
            continue
        reached[season] += 1
        if len(reached_by[season]) < 3:
            reached_by[season].append(
                f"undertone={undertone}, depth={depth}, clarity={clarity}, contrast={contrast}"
            )

    print("=" * 70)
    print(f"Season Reachability Check — {total} input combinations tested")
    print("=" * 70)

    unreachable = [s for s, count in reached.items() if count == 0]

    for season in sorted(all_seasons):
        count = reached[season]
        status = "✅" if count > 0 else "❌ UNREACHABLE"
        print(f"\n{status}  {season}  ({count} combos)")
        for example in reached_by[season]:
            print(f"    e.g. {example}")

    if unknown_hits:
        print(f"\n⚠️  {unknown_hits} combinations matched a season key not present "
              f"in `seasons` (typo in classification_rules?)")

    print("\n" + "=" * 70)
    if unreachable:
        print(f"❌ FAIL: {len(unreachable)} season(s) unreachable: {', '.join(sorted(unreachable))}")
        print("   Check classification_rules ordering — a broader rule earlier")
        print("   in the list may be matching before a more specific one below it.")
        return 1
    else:
        print(f"✅ PASS: all {len(all_seasons)} seasons are reachable.")
        return 0


if __name__ == "__main__":
    sys.exit(main())