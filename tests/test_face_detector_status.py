"""A missing MediaPipe must never degrade results silently: warn at start-up, say so in
/api/health, and make scripts/eval.py refuse to run unless --allow-haar is given."""

import importlib.util
import logging
from pathlib import Path

import ai.colorimetry.face_color as face_color
import backend.app as app_module

DEGRADED = {
    "mediapipe_ready": False,
    "mediapipe_error": "MediaPipe is not installed in this Python environment (No module named 'mediapipe')",
    "degraded_reason": "MediaPipe is not installed in this Python environment (No module named 'mediapipe'). "
                       + face_color.HAAR_FALLBACK_WARNING,
}

_spec = importlib.util.spec_from_file_location("eval_script", Path(__file__).resolve().parents[1] / "scripts/eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def test_missing_mediapipe_is_detected_and_logged(monkeypatch, caplog):
    def not_installed():
        raise ImportError("No module named 'mediapipe'")

    # Fresh, uninitialised detector state (monkeypatch restores the real one afterwards)
    monkeypatch.setattr(face_color, "_landmarker", None)
    monkeypatch.setattr(face_color, "_landmarker_error", None)
    monkeypatch.setattr(face_color, "_import_mediapipe_tasks", not_installed)

    with caplog.at_level(logging.WARNING, logger=face_color.__name__):
        status = face_color.landmarker_status()

    assert status["mediapipe_ready"] is False
    assert "not installed" in status["degraded_reason"]
    assert "LESS ACCURATE" in status["degraded_reason"]
    assert any(r.levelno == logging.WARNING and "not installed" in r.getMessage() for r in caplog.records)


def test_server_warns_at_startup_when_degraded(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "landmarker_status", lambda: DEGRADED)
    with caplog.at_level(logging.WARNING, logger="shadesense"):
        app_module.check_face_detector()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("DEGRADED" in m and "not installed" in m for m in warnings)


def test_health_reports_degraded_with_reason(monkeypatch):
    monkeypatch.setattr(app_module, "landmarker_status", lambda: DEGRADED)
    res = app_module.app.test_client().get("/api/health")
    body = res.get_json()
    assert res.status_code == 200          # still serving, but not at full quality
    assert body["status"] == "degraded"
    assert body["face_landmarker_ready"] is False
    assert "not installed" in body["degraded_reasons"][0]


def test_health_ok_has_no_reasons():
    body = app_module.app.test_client().get("/api/health").get_json()
    if body["face_landmarker_ready"]:
        assert body["status"] == "ok" and body["degraded_reasons"] == []


def _empty_labels(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("image,depth,undertone,labeler,notes\n")
    return str(labels)


def test_eval_refuses_without_mediapipe(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ev, "landmarker_status", lambda: DEGRADED)
    for command in ("run", "stability", "topup"):
        assert ev.main([command, "--labels" if command != "stability" else "--images",
                        _empty_labels(tmp_path) if command != "stability" else str(tmp_path)]) == 2
        assert "Refusing to run" in capsys.readouterr().err


def test_eval_allow_haar_runs_with_a_warning(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ev, "landmarker_status", lambda: DEGRADED)
    code = ev.main(["run", "--labels", _empty_labels(tmp_path), "--allow-haar"])
    captured = capsys.readouterr()
    assert code == 1                                   # got past the guard, stopped at "no labelled rows"
    assert "No labelled rows" in captured.out
    assert "--allow-haar" in captured.err and "WARNING" in captured.err


def test_sample_does_not_need_the_detector(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "landmarker_status", lambda: DEGRADED)
    labels = tmp_path / "labels.csv"
    labels.write_text("existing")
    # Refuses only because labels exist (no --force), not because of MediaPipe
    assert ev.main(["sample", "--labels", str(labels), "--source", str(tmp_path)]) == 1
