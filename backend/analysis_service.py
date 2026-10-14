"""
analysis_service.py — runs the ShadeSense pipeline on one decoded image.

    face colour (CIELAB) → skin observations → foundation matching (ΔE00)
    → season ranking → narrative text (Groq, or a template fallback)

Colour science computes every result; the LLM only rewrites those results as prose and
never receives the photo. Routes in app.py stay thin and call into this module.
"""

import logging
from typing import Any, Dict, List

import cv2
import numpy as np

from ai.colorimetry.face_color import (
    UNDERTONE_RELIABILITY,
    UNDERTONE_RELIABILITY_NOTE,
    analyze_face_color,
)
from ai.colorimetry.skin_observations import detect_skin_observations
from ai.recommendation.groq_writer import get_ai_recommendations_ranked
from ai.recommendation.match import match_foundations
from ai.recommendation.season import get_ranked_season_recommendations, load_season_reference

logger = logging.getLogger(__name__)

# Larger photos are downscaled first: landmarks and colour medians don't need 12 MP,
# and it keeps per-request CPU time bounded.
MAX_ANALYSIS_SIDE = 1600
TOP_FOUNDATIONS = 5


def decode_image(data: bytes):
    """Decode uploaded bytes to a BGR array (None if not a supported image). Nothing touches disk."""
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def downscale(img: np.ndarray, max_side: int = MAX_ANALYSIS_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _analysis_block(face: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "depth": face["depth"],
        "undertone": face["undertone"],
        # Undertone barely beat chance against human labels: clients must show it as a guess
        "undertone_reliability": UNDERTONE_RELIABILITY,
        "undertone_note": UNDERTONE_RELIABILITY_NOTE,
        "clarity": face["clarity"],
        "contrast": face["contrast"],
        "lab": {"L": face["L"], "a": face["a"], "b": face["b"]},
        "ita_degrees": face["ita_degrees"],
        "ita_bucket": face["ita_bucket"],
        "confidence": face["confidence"],
        "landmark_source": face["landmark_source"],
        "skin_regions_used": face["skin_regions_used"],
        "per_region_values": face["per_region_values"],
    }


def _public_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{**m, "lab": list(m["lab"])} for m in matches]


def run_analysis(img: np.ndarray, foundations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Full analysis for one image. Raises ValueError when no usable face is found."""
    face = analyze_face_color(downscale(img))

    try:
        observations = detect_skin_observations(face["per_region_values"])
    except Exception:
        logger.exception("Skin observations failed; continuing without them")
        observations = {}

    matches = match_foundations((face["L"], face["a"], face["b"]), foundations, top_k=TOP_FOUNDATIONS)

    seasons_full = load_season_reference()["seasons"]
    ranked = get_ranked_season_recommendations(face, top_n=3)
    winner = seasons_full[ranked["winner_key"]]

    ai = get_ai_recommendations_ranked(face, matches, ranked, seasons_full, observations)
    structured = ai["structured"]
    details = structured.get("season", {})

    return {
        "success": True,
        "method": "colorimetry_v2",
        "ai_used": ai["source"] == "groq",
        "ai_source": ai["source"],          # "groq" or "template"
        "analysis": _analysis_block(face),
        "skin_observations": observations,
        "foundations": _public_matches(matches),
        "season": {
            "key": ranked["winner_key"],
            "label": winner.get("label", ranked["winner_key"]),
            "family": winner.get("family", "neutral"),
            "jewelry": details.get("jewelry", []),
            "lipstick": details.get("lipstick", []),
            "blush": details.get("blush", []),
            "best_colors": details.get("best_colors", {}),
            "worst_colors": details.get("worst_colors", []),
        },
        "recommendations": ai["summary"],
        "ai_recommendations": {"summary": ai["summary"]},
        "structured_recommendations": {**structured, "ranked_seasons": ranked},
    }


def run_batch_item(img: np.ndarray, foundations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Lightweight per-image result for the batch endpoint (no LLM call)."""
    face = analyze_face_color(downscale(img))
    matches = match_foundations((face["L"], face["a"], face["b"]), foundations, top_k=1)
    ranked = get_ranked_season_recommendations(face, top_n=1)
    return {
        "depth": face["depth"],
        "undertone": face["undertone"],
        "undertone_reliability": UNDERTONE_RELIABILITY,
        "season": ranked["top_seasons"][0]["season_label"],
        "top_match": _public_matches(matches)[0] if matches else None,
    }
