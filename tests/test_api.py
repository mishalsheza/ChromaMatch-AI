"""API behaviour and hardening. Uses a generated image and a mocked face analysis, so it
runs on a fresh clone without the (git-ignored) dataset and without any network calls."""

from io import BytesIO
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import ai.recommendation.groq_writer as groq_writer
import backend.analysis_service as service
import backend.app as app_module

FACE_RESULT = {
    "L": 62.0, "a": 13.0, "b": 19.0,
    "ita_degrees": 32.0, "ita_bucket": "intermediate",
    "depth": "medium", "undertone": "neutral", "clarity": "muted", "contrast": "medium",
    "confidence": 0.7, "landmark_source": "mediapipe",
    "skin_regions_used": ["chin", "forehead", "left_cheek", "right_cheek"],
    "white_balance_applied": False,
    "per_region_values": {
        name: {"L": 62.0, "a": 13.0, "b": 19.0}
        for name in ["forehead", "left_cheek", "right_cheek", "chin"]
    },
}


def png_bytes(size=(64, 64)) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((*size, 3), (120, 140, 170), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(monkeypatch):
    app_module.limiter.reset()
    monkeypatch.setattr(service, "analyze_face_color", lambda img: dict(FACE_RESULT))
    return app_module.app.test_client()


def post_image(client, url="/api/analyze", data=None, **extra):
    form = {"image": (BytesIO(data if data is not None else png_bytes()), "face.png"), **extra}
    return client.post(url, data=form, content_type="multipart/form-data")


# ── analyze ──

def test_analyze_without_llm_reports_template(client):
    res = post_image(client)
    assert res.status_code == 200
    body = res.get_json()
    assert body["ai_source"] == "template" and body["ai_used"] is False
    assert body["analysis"]["contrast"] == "medium"
    assert body["analysis"]["undertone_reliability"] == "low"
    assert "best guess" in body["analysis"]["undertone_note"]
    assert len(body["foundations"]) == 5
    assert body["season"]["label"]


def test_analyze_with_llm_reports_groq(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="### Summary\n\nText."))])
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    monkeypatch.setattr(groq_writer, "Groq", lambda **kw: fake)
    body = post_image(client).get_json()
    assert body["ai_source"] == "groq" and body["ai_used"] is True
    assert body["recommendations"] == "### Summary\n\nText."


def test_analyze_missing_image(client):
    res = client.post("/api/analyze", data={}, content_type="multipart/form-data")
    assert res.status_code == 400
    assert res.get_json()["error"] == "No image provided"


def test_analyze_corrupt_image(client):
    res = post_image(client, data=b"definitely not an image")
    assert res.status_code == 400


def test_no_face_is_a_400_with_a_friendly_message(client, monkeypatch):
    def no_face(img):
        raise ValueError("No face detected. Try a well-lit, front-facing photo.")

    monkeypatch.setattr(service, "analyze_face_color", no_face)
    res = post_image(client)
    assert res.status_code == 400
    assert "No face detected" in res.get_json()["error"]


def test_unexpected_errors_do_not_leak_details(client, monkeypatch):
    def boom(img):
        raise RuntimeError("secret internal path /Users/someone/x.py")

    monkeypatch.setattr(service, "analyze_face_color", boom)
    res = post_image(client)
    assert res.status_code == 500
    assert res.get_json()["error"] == "Internal server error"


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 1024)
    res = post_image(client, data=b"x" * 4096)
    assert res.status_code == 413


def test_large_images_are_downscaled():
    big = np.zeros((4000, 3000, 3), dtype=np.uint8)
    assert max(service.downscale(big).shape[:2]) == service.MAX_ANALYSIS_SIDE


# ── rate limiting, CORS, health ──

def test_rate_limit_on_analyze(client):
    limit = int(app_module.RATE_LIMIT_ANALYZE.split()[0])
    statuses = [client.post("/api/analyze").status_code for _ in range(limit + 1)]
    assert statuses[:limit] == [400] * limit
    assert statuses[-1] == 429


def test_cors_allows_only_configured_origins(client):
    allowed = app_module.CORS_ORIGINS[0]
    res = client.get("/api/health", headers={"Origin": allowed})
    assert res.headers.get("Access-Control-Allow-Origin") == allowed
    res = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in res.headers


def test_health_reports_components(client):
    body = client.get("/api/health").get_json()
    assert body["foundations_loaded"] > 1000
    assert body["seasons_loaded"] == 12
    assert body["llm_configured"] is False
    assert body["status"] in {"ok", "degraded"}


# ── batch ──

def test_batch_is_capped(client):
    files = [(BytesIO(png_bytes()), f"{i}.png") for i in range(app_module.MAX_BATCH_IMAGES + 1)]
    res = client.post("/api/analyze-batch", data={"images": files}, content_type="multipart/form-data")
    assert res.status_code == 400


def test_batch_returns_one_result_per_image(client):
    files = [(BytesIO(png_bytes()), "a.png"), (BytesIO(b"junk"), "b.png")]
    body = client.post("/api/analyze-batch", data={"images": files}, content_type="multipart/form-data").get_json()
    assert [r["filename"] for r in body["results"]] == ["a.png", "b.png"]
    assert "season" in body["results"][0] and "error" in body["results"][1]


# ── try-on ──

def test_tryon_uses_the_shade_database(client):
    shade = app_module.FOUNDATIONS[0]
    res = post_image(client, "/api/tryon", brand=shade["brand"], shade=shade["shade"], intensity="0.7")
    assert res.status_code == 200
    body = res.get_json()
    assert body["hex"] == shade["hex"] and body["image"]


def test_tryon_rejects_unknown_shade_and_bad_intensity(client):
    shade = app_module.FOUNDATIONS[0]
    assert post_image(client, "/api/tryon", brand="nope", shade="nope").status_code == 400
    assert post_image(client, "/api/tryon", brand=shade["brand"], shade=shade["shade"], intensity="abc").status_code == 400
    assert post_image(client, "/api/tryon", brand=shade["brand"], shade=shade["shade"], intensity="5").status_code == 400
