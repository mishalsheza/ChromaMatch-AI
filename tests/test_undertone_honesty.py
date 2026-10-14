"""
Undertone agreed with human labels only ~26% of the time (≈ chance, see README "Evaluation"),
so every place it reaches the user must present it as a low-confidence best guess.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import ai.recommendation.groq_writer as groq_writer
from ai.colorimetry.face_color import UNDERTONE_RELIABILITY, UNDERTONE_RELIABILITY_NOTE
from ai.recommendation.season import get_ranked_season_recommendations

ROOT = Path(__file__).resolve().parents[1]
SEASONS = json.loads((ROOT / "ai/data/season_color_reference.json").read_text())["seasons"]
FACE = {"depth": "medium", "undertone": "neutral", "clarity": "muted", "contrast": "medium",
        "confidence": 0.7, "ita_degrees": 30.0, "a": 12.0, "b": 15.0}
MATCHES = [{"brand": "colourpop", "shade": "Medium 75", "distance": 1.8}]


def _ranked():
    return get_ranked_season_recommendations(FACE, top_n=3)


def test_reliability_is_flagged_low():
    assert UNDERTONE_RELIABILITY == "low"
    assert "best guess" in UNDERTONE_RELIABILITY_NOTE and "chance" in UNDERTONE_RELIABILITY_NOTE


def test_system_prompt_forbids_stating_undertone_as_fact():
    prompt = groq_writer.NARRATIVE_SYSTEM_PROMPT
    assert "LOW-CONFIDENCE BEST GUESS" in prompt
    assert "NEVER state the undertone as a fact" in prompt


def test_user_prompt_marks_undertone_and_season_as_uncertain():
    prompt = groq_writer.build_ranked_ai_prompt(FACE, MATCHES, _ranked(), SEASONS)
    assert "Undertone: neutral  (reliability: low; a best guess, NOT a fact" in prompt
    assert "depends heavily on the undertone guess" in prompt
    assert "not on the undertone label" in prompt          # foundation match is colour-based
    assert "NOT how likely the labels are right" in prompt  # 'confidence' is measurement quality


def test_template_text_hedges_undertone_and_season():
    result = groq_writer.get_ai_recommendations_ranked(FACE, MATCHES, _ranked(), SEASONS)
    assert result["source"] == "template"
    summary = result["summary"]
    assert "best guess for your undertone" in summary
    assert "depends heavily on the undertone guess" in summary
    assert "neutral** undertone" not in summary            # never "a **neutral** undertone" as fact
    assert result["structured"]["skin"]["undertone_reliability"] == "low"


def test_llm_receives_the_hedging_instructions(monkeypatch):
    calls = []
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="text"))])

    def create(**kwargs):
        calls.append(kwargs)
        return reply

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(groq_writer, "Groq", lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    groq_writer.get_ai_recommendations_ranked(FACE, MATCHES, _ranked(), SEASONS)
    system, user = calls[0]["messages"]
    assert "NEVER state the undertone as a fact" in system["content"]
    assert "a best guess, NOT a fact" in user["content"]


def test_frontend_never_presents_undertone_as_fact():
    html = (ROOT / "ShadeSense.html").read_text()
    assert "Undertone (best guess)" in html
    assert "with ${undertone} undertone" not in html
    assert "'</b> Undertone'" not in html
    assert "Confidence: ' + confidence" not in html          # it's measurement quality
