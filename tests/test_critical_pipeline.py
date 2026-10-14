import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai.recommendation.groq_writer as groq_writer
from ai.recommendation.groq_writer import get_ai_recommendations_ranked

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "ai/data/season_color_reference.json").read_text())
SEASONS = REFERENCE["seasons"]
# Part of the git-ignored dataset: present on the author's machine, absent on a fresh clone.
REAL_FACE_IMAGE = ROOT / "ai/data/raw/medium_warm/27_1_2_20170116175930683.jpg.chip.jpg"


def ranked_input():
    return {
        "top_seasons": [
            {
                "season_key": "deep_winter",
                "season_label": "Deep Winter",
                "score": 8.0,
                "score_pct": 88.9,
                "breakdown": {"undertone": 3.0, "depth": 2.0, "clarity": 2.0, "contrast": 1.0},
            },
            {
                "season_key": "true_winter",
                "season_label": "True Winter (Cool Winter)",
                "score": 7.0,
                "score_pct": 77.8,
                "breakdown": {"undertone": 3.0, "depth": 2.0, "clarity": 2.0, "contrast": 0.0},
            },
        ],
        "margin": 1.0,
        "margin_pct": 11.1,
        "is_close_call": False,
        "winner_key": "deep_winter",
    }


def face_input():
    return {
        "depth": "deep",
        "undertone": "cool",
        "clarity": "muted",
        "confidence": 0.8,
        "ita_degrees": -20.0,
    }


def fake_client(text, calls=None):
    def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_ranked_writer_successful_groq_response(monkeypatch):
    calls = []
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(groq_writer, "Groq", lambda **kwargs: fake_client("### Your Skin & Season\n\nA grounded summary.", calls))
    result = get_ai_recommendations_ranked(
        face_input(),
        [{"brand": "colourpop", "shade": "Deep 115", "distance": 2.4}],
        ranked_input(),
        SEASONS,
    )
    assert result["source"] == "groq"
    assert result["summary"] == "### Your Skin & Season\n\nA grounded summary."
    assert result["structured"]["season"]["label"] == "Deep Winter"
    # The LLM only receives text derived from measurements: never image data
    messages = calls[0]["messages"]
    assert all(isinstance(m["content"], str) for m in messages)   # text only, no image parts
    assert "data:image" not in "".join(m["content"] for m in messages)


def test_ranked_writer_groq_failure_reports_template(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def broken(**kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(groq_writer, "Groq", broken)
    result = get_ai_recommendations_ranked(
        face_input(),
        [{"brand": "colourpop", "shade": "Deep 115", "distance": 4.2}],
        ranked_input(),
        SEASONS,
    )
    assert result["source"] == "template"
    assert "Deep Winter" in result["summary"]
    assert result["structured"]["season"]["label"] == "Deep Winter"
    assert result["ranked_seasons"] == ranked_input()


def test_ranked_writer_without_key_reports_template():
    # conftest blanks GROQ_API_KEY and makes any Groq client construction fail
    result = get_ai_recommendations_ranked(face_input(), [], ranked_input(), SEASONS)
    assert result["source"] == "template"


@pytest.mark.skipif(not REAL_FACE_IMAGE.exists(), reason="dataset image not present (git-ignored)")
def test_analyze_endpoint_real_image_response_shape():
    """End-to-end on a real face photo: MediaPipe, colour, matching, seasons, template text."""
    import backend.app as app_module

    app_module.limiter.reset()
    response = app_module.app.test_client().post(
        "/api/analyze",
        data={"image": (BytesIO(REAL_FACE_IMAGE.read_bytes()), REAL_FACE_IMAGE.name)},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["ai_source"] == "template" and payload["ai_used"] is False
    assert {"analysis", "foundations", "season", "recommendations", "structured_recommendations", "skin_observations"} <= payload.keys()
    assert {"depth", "undertone", "clarity", "contrast", "confidence", "lab"} <= payload["analysis"].keys()
    assert payload["analysis"]["landmark_source"] == "mediapipe"
    assert len(payload["foundations"]) == 5
    assert payload["foundations"][0]["distance"] <= payload["foundations"][-1]["distance"]
