"""
app.py — ShadeSense Flask API.

  POST /api/analyze        one photo → skin colour, foundation matches, season, narrative
  POST /api/tryon          one photo + brand/shade → preview image (experimental)
  POST /api/analyze-batch  a few photos → compact results (testing aid, no LLM)
  GET  /api/health         readiness of each component
  GET  /                   the frontend (ShadeSense.html)

Uploads are decoded in memory and never written to disk. All settings come from
environment variables (see backend/.env.example).
"""

import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv  # noqa: E402

# Real environment variables win over the file; the file is optional.
load_dotenv(ROOT_DIR / "backend" / ".env")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


# ── CONFIG ──
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = _env_int("PORT", 5001)
CORS_ORIGINS = _env_list(
    "SHADESENSE_CORS_ORIGINS",
    "http://localhost:5500,http://127.0.0.1:5500,http://localhost:5501,http://127.0.0.1:5501",
)
MAX_UPLOAD_MB = _env_int("SHADESENSE_MAX_UPLOAD_MB", 10)
MAX_IMAGE_PIXELS = _env_int("SHADESENSE_MAX_IMAGE_PIXELS", 40_000_000)   # ≈ 7300×5500
MAX_BATCH_IMAGES = _env_int("SHADESENSE_MAX_BATCH_IMAGES", 5)
RATE_LIMIT_ANALYZE = os.environ.get("SHADESENSE_RATE_LIMIT_ANALYZE", "10 per minute;200 per day")
RATE_LIMIT_BATCH = os.environ.get("SHADESENSE_RATE_LIMIT_BATCH", "2 per minute")
RATE_LIMIT_STORAGE = os.environ.get("SHADESENSE_RATE_LIMIT_STORAGE", "memory://")

# Decompression-bomb guard: OpenCV refuses to decode images above this many pixels.
# Must be set before cv2 is imported.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(MAX_IMAGE_PIXELS))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402
from flask_limiter import Limiter  # noqa: E402
from flask_limiter.util import get_remote_address  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

# Imported eagerly: if a core module is broken, the server fails at start-up
# instead of answering every request with a 500.
from ai.colorimetry.face_color import landmarker_status  # noqa: E402
from ai.recommendation.match import find_shade, load_foundation_db  # noqa: E402
from ai.recommendation.season import load_season_reference  # noqa: E402
from backend.analysis_service import decode_image, run_analysis, run_batch_item  # noqa: E402
from backend.foundation_tryon import FoundationTryOn  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("shadesense")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024   # larger uploads → 413
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})
limiter = Limiter(get_remote_address, app=app, storage_uri=RATE_LIMIT_STORAGE, headers_enabled=True)

def check_face_detector() -> dict:
    """Load the face detector now, at start-up, so a missing MediaPipe is reported
    immediately instead of silently degrading results on the first request."""
    status = landmarker_status()
    if status["mediapipe_ready"]:
        logger.info("Face detector: MediaPipe Face Landmarker ready")
    else:
        logger.warning("Face detector DEGRADED: %s", status["degraded_reason"])
    return status


check_face_detector()
FOUNDATIONS = load_foundation_db()
SEASON_COUNT = len(load_season_reference().get("seasons", {}))
TRYON = FoundationTryOn()


def _error(message: str, status: int):
    return jsonify({"success": False, "error": message}), status


def _uploaded_image(field: str = "image"):
    """Return (image, None) or (None, error_response)."""
    if field not in request.files:
        return None, _error("No image provided", 400)
    img = decode_image(request.files[field].read())
    if img is None:
        return None, _error("Unsupported or corrupt image (use JPEG, PNG or WebP)", 400)
    return img, None


# ── ROUTES ──

@app.route("/")
def frontend():
    return send_from_directory(ROOT_DIR, "ShadeSense.html")


@app.route("/api")
def api_index():
    return jsonify({
        "name": "ShadeSense API",
        "endpoints": {
            "POST /api/analyze": "Main analysis endpoint",
            "POST /api/tryon": "Try-on preview (experimental)",
            "POST /api/analyze-batch": f"Up to {MAX_BATCH_IMAGES} images, no narrative (testing)",
            "GET /api/health": "Component readiness",
        },
    })


@app.route("/api/health", methods=["GET"])
@limiter.exempt
def health():
    face = landmarker_status()
    checks = {
        "foundations_loaded": len(FOUNDATIONS),
        "seasons_loaded": SEASON_COUNT,
        "face_landmarker_ready": face["mediapipe_ready"],   # False → Haar fallback only
        "llm_configured": bool(os.environ.get("GROQ_API_KEY")),  # False → template text
    }
    ready = checks["foundations_loaded"] > 0 and checks["seasons_loaded"] > 0
    reasons = [] if face["mediapipe_ready"] else [face["degraded_reason"]]
    status = "unavailable" if not ready else ("degraded" if reasons else "ok")
    return jsonify({"status": status, "degraded_reasons": reasons, **checks}), (200 if ready else 503)


@app.route("/api/analyze", methods=["POST"])
@limiter.limit(RATE_LIMIT_ANALYZE)
def analyze():
    img, err = _uploaded_image()
    if err:
        return err
    try:
        return jsonify(run_analysis(img, FOUNDATIONS))
    except ValueError as e:   # our own user-facing messages (e.g. no face found)
        return _error(str(e), 400)


@app.route("/api/tryon", methods=["POST"])
@limiter.limit(RATE_LIMIT_ANALYZE)
def try_foundation():
    img, err = _uploaded_image()
    if err:
        return err
    try:
        intensity = float(request.form.get("intensity", 0.7))
    except ValueError:
        return _error("intensity must be a number between 0 and 1", 400)
    if not 0.0 <= intensity <= 1.0:
        return _error("intensity must be a number between 0 and 1", 400)

    shade = find_shade(FOUNDATIONS, request.form.get("brand", ""), request.form.get("shade", ""))
    if shade is None:
        return _error("Unknown brand/shade", 400)

    rgb = [int(shade["hex"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return jsonify({
        "success": True,
        "image": TRYON.apply_foundation(img, rgb, intensity),
        "brand": shade["brand"],
        "shade": shade["shade"],
        "hex": shade["hex"],
    })


@app.route("/api/analyze-batch", methods=["POST"])
@limiter.limit(RATE_LIMIT_BATCH)
def analyze_batch():
    files = request.files.getlist("images")
    if not files:
        return _error("No images provided", 400)
    if len(files) > MAX_BATCH_IMAGES:
        return _error(f"At most {MAX_BATCH_IMAGES} images per batch", 400)

    results = []
    for file in files:
        img = decode_image(file.read())
        if img is None:
            results.append({"filename": file.filename, "error": "Unsupported or corrupt image"})
            continue
        try:
            results.append({"filename": file.filename, **run_batch_item(img, FOUNDATIONS)})
        except ValueError as e:
            results.append({"filename": file.filename, "error": str(e)})
    return jsonify({"success": True, "results": results})


# ── ERROR HANDLING ──

@app.errorhandler(413)
def too_large(_e):
    return _error(f"Image too large (max {MAX_UPLOAD_MB} MB)", 413)


@app.errorhandler(429)
def rate_limited(e):
    return _error(f"Too many requests ({e.description}). Please wait a moment.", 429)


@app.errorhandler(HTTPException)
def http_error(e):
    return _error(e.description or e.name, e.code or 500)


@app.errorhandler(Exception)
def unexpected_error(_e):
    # Details go to the server log, never to the client.
    logger.exception("Unhandled error on %s %s", request.method, request.path)
    return _error("Internal server error", 500)


if __name__ == "__main__":
    logger.info("ShadeSense API on http://%s:%d (%d foundation shades)", HOST, PORT, len(FOUNDATIONS))
    # Flask's built-in server is for local development. For deployment use a WSGI
    # server, e.g. `gunicorn -w 2 -b 0.0.0.0:5001 backend.app:app` (and set
    # SHADESENSE_RATE_LIMIT_STORAGE to Redis so limits are shared between workers).
    app.run(host=HOST, port=PORT, debug=False)
