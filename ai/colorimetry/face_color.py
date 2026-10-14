"""
face_color.py — measure skin colour from a face photo (deterministic, no ML training).

Pipeline
  1. MediaPipe Face Landmarker → 478 landmarks (468 mesh + 10 iris); a Haar cascade
     face box is a coarse fallback, used only if MediaPipe can't run (see landmarker_status)
  2. Skin regions = small polygons on the forehead, both cheeks and the chin
  3. Per region: drop very dark / blown-out pixels (HSV V channel), keep the dominant
     k-means colour cluster, take its median in CIELAB
  4. Overall skin colour = median of the skin regions
  5. Derived labels: depth (ITA), undertone (a*/b*), clarity (chroma), contrast (L* spread)

Under-eye and perioral regions are measured too, but only for skin_observations.py.
They are expected to differ from the rest of the face, so they are NOT part of the
overall skin colour.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .color_space import bgr_to_lab

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── CONFIG ──
# White balance is OFF by default, on purpose. The gray-world method assumes the
# average colour of the whole photo is neutral gray. In a face close-up the "whole
# photo" is mostly skin, so gray-world removes the very signal we want to measure:
# b* collapses towards 0 (which makes ITA swing to ±90°) and lightness is forced to
# mid-gray, so light faces read as deep and deep faces read as light. Measured on
# the dataset, depth agreement fell from 57% to 24% with it on. Only enable it for
# experiments, e.g. with a calibration target in frame.
APPLY_WHITE_BALANCE = os.environ.get("SHADESENSE_APPLY_WHITE_BALANCE", "0").strip().lower() in {"1", "true", "yes"}

MODEL_PATH = Path(os.environ.get("FACE_LANDMARKER_MODEL", REPO_ROOT / "backend" / "face_landmarker.task"))
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

# ── REGIONS ──
# MediaPipe Face Mesh indices. Naming follows MediaPipe: "left" = the subject's left
# (the right-hand side of a non-mirrored photo).
#
# Derived from the official canonical face mesh (ai/data/mediapipe/canonical_face_model.obj):
# each region is the set of mesh vertices inside an anatomical box, with a margin
# from MediaPipe's official eye / eyebrow / lips / nose / face-oval landmark sets.
# Right-side regions are exact mirrors of the left. tests/test_face_regions.py checks
# all of this, and scripts/visualize_regions.py draws them on a photo.
# An entry (i, j) means "the midpoint of landmarks i and j". The mesh is sparse on the
# forehead, and this keeps the polygon halfway between the mid-forehead and hairline rows.
SKIN_REGIONS = {
    # from the glabella (9) up to halfway towards the hairline, clear of the brows
    "forehead": (9, 69, 108, 151, 337, 299, (67, 69), (109, 108), (10, 151), (338, 337), (297, 299)),
    # cheek: below the under-eye band, outside the nose, above the mouth corners
    "left_cheek": (266, 280, 330, 411, 425, 426, 427),
    "right_cheek": (36, 50, 101, 187, 205, 206, 207),
    # chin: below the lower lip, inside the face oval
    "chin": (18, 83, 199, 200, 201, 208, 313, 421, 428),
}
OBSERVATION_REGIONS = {
    # band between the ring just below the eyelid contour and the next mesh row down
    "left_under_eye": (252, 253, 254, 449, 450, 451),
    "right_under_eye": (22, 23, 24, 229, 230, 231),
    # patch just outside the mouth corners
    "left_perioral": (273, 410, 422, 432, 436),
    "right_perioral": (43, 186, 202, 212, 216),
}
REGION_LANDMARKS = {**SKIN_REGIONS, **OBSERVATION_REGIONS}

# Shrink each polygon towards its centre so it stays clear of hair, brows and edges.
REGION_INSET = 0.85

# Haar fallback: region boxes as fractions (x0, y0, x1, y1) of the detected face box.
# Much coarser than landmarks; results from this path get a confidence penalty.
HAAR_REGION_BOXES = {
    "forehead": (0.36, 0.06, 0.64, 0.18),
    "left_cheek": (0.62, 0.50, 0.80, 0.66),
    "right_cheek": (0.20, 0.50, 0.38, 0.66),
    "chin": (0.42, 0.86, 0.58, 0.95),
    "left_under_eye": (0.58, 0.40, 0.74, 0.46),
    "right_under_eye": (0.26, 0.40, 0.42, 0.46),
    "left_perioral": (0.62, 0.72, 0.70, 0.80),
    "right_perioral": (0.30, 0.72, 0.38, 0.80),
}
HAAR_CONFIDENCE_PENALTY = 0.5

# ── THRESHOLDS ──
HSV_VALUE_MIN = 35     # drop deep shadows
HSV_VALUE_MAX = 245    # drop specular highlights
ITA_BUCKETS = (        # Chardon et al. (1991) ITA skin categories
    (55.0, "very_light"),
    (41.0, "light"),
    (28.0, "intermediate"),
    (10.0, "tan"),
    (-30.0, "brown"),
)
CHROMA_CLEAR_THRESHOLD = 22.0

# Olive = LOW a* (little redness) + clearly present b* (yellow-green)
OLIVE_A_MAX = 6.0
OLIVE_B_MIN = 14.0
OLIVE_B_TO_A_RATIO = 2.2

# Undertone from the b*/a* ratio (yellow vs red)
WARM_RATIO_MIN = 1.8
COOL_RATIO_MAX = 0.6

# Against hand-labelled photos (scripts/eval.py run) the undertone label agreed with
# people about as often as chance, called most faces "neutral" and never "olive";
# depth agreed far better. So undertone is surfaced everywhere as a low-confidence best
# guess (API, UI, LLM prompt). The thresholds above are unchanged.
# See README → "Evaluation & limitations".
UNDERTONE_RELIABILITY = "low"
UNDERTONE_RELIABILITY_NOTE = (
    "Undertone is a best guess. In testing against people's own judgements it agreed "
    "about as often as chance and leaned heavily towards 'neutral', so check it in "
    "person (for example, gold vs silver jewellery) before relying on it."
)

# "Contrast" here = L* spread across the skin regions (a proxy, see README)
CONTRAST_HIGH_THRESHOLD = 8.0
CONTRAST_LOW_THRESHOLD = 3.0
MIN_CONFIDENCE_FOR_CONTRAST = 0.5

# A skin region this far (CIE76 ΔE) from the median of the skin regions is treated as
# "not skin", e.g. bangs over the forehead (measured ΔE 32–49), and dropped. Normal
# left/right lighting differences measured ΔE ≈ 10–17, so they stay in.
REGION_OUTLIER_DELTA_E = 20.0

# Confidence heuristic: a region is "fully trusted" at this many usable pixels
FULL_CONFIDENCE_PIXELS = 100
DISPERSION_SCALE = 10.0


# ── CLASSIFIERS (pure functions) ──

def classify_undertone(a_value: float, b_value: float) -> str:
    """Classify undertone from LAB chroma direction: olive / warm / cool / neutral."""
    if a_value <= OLIVE_A_MAX and b_value >= OLIVE_B_MIN and b_value >= a_value * OLIVE_B_TO_A_RATIO:
        return "olive"

    # a* at or below ~0 means no measurable redness: the ratio is unbounded in b*'s direction
    if a_value <= 0.01:
        ratio = math.inf if b_value > 0 else (-math.inf if b_value < 0 else 0.0)
    else:
        ratio = b_value / a_value

    if ratio >= WARM_RATIO_MIN:
        return "warm"
    if ratio <= COOL_RATIO_MAX:
        return "cool"
    return "neutral"


def classify_contrast(region_values: dict) -> str:
    """Proxy for seasonal 'contrast' using L* spread across skin regions."""
    if len(region_values) < 2:
        return "medium"
    l_values = [v["L"] for v in region_values.values()]
    spread = max(l_values) - min(l_values)
    if spread >= CONTRAST_HIGH_THRESHOLD:
        return "high"
    if spread <= CONTRAST_LOW_THRESHOLD:
        return "low"
    return "medium"


def ita_degrees(lightness: float, b_value: float) -> float:
    """Individual Typology Angle: atan((L* - 50) / b*) in degrees."""
    if abs(b_value) < 1e-6:
        return 90.0 if lightness >= 50.0 else -90.0
    return math.degrees(math.atan((lightness - 50.0) / b_value))


def map_ita_depth(ita: float) -> tuple[str, str]:
    """Map ITA to a detailed bucket and the app's 3-level depth."""
    bucket = next((name for threshold, name in ITA_BUCKETS if ita >= threshold), "dark")

    if bucket in {"very_light", "light"}:
        depth = "light"
    elif bucket in {"intermediate", "tan"}:
        depth = "medium"
    else:
        depth = "deep"
    return bucket, depth


def classify_clarity(a_value: float, b_value: float) -> str:
    """LAB chroma >= 22 is 'clear', below is 'muted'."""
    chroma = math.hypot(a_value, b_value)
    return "clear" if chroma >= CHROMA_CLEAR_THRESHOLD else "muted"


# ── FACE DETECTION ──

_landmarker = None
_landmarker_error: str | None = None
_landmarker_lock = threading.Lock()   # MediaPipe graphs aren't documented as thread-safe
_haar_cascade = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_model() -> Path:
    """Return the model path, downloading (and checksum-verifying) it if missing."""
    if not MODEL_PATH.exists():
        logger.info("Downloading MediaPipe Face Landmarker model to %s", MODEL_PATH)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = MODEL_PATH.with_suffix(".download")
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        if _sha256(tmp_path) != MODEL_SHA256:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("Downloaded face_landmarker.task failed its SHA-256 check")
        tmp_path.replace(MODEL_PATH)
    return MODEL_PATH


def _import_mediapipe_tasks():
    """Separate function so tests can simulate MediaPipe not being installed."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    return mp_python, vision


HAAR_FALLBACK_WARNING = (
    "Falling back to the Haar cascade: face regions are coarse rectangles, so colour "
    "results are LESS ACCURATE and won't match evaluated numbers."
)


def _get_landmarker():
    """Create the MediaPipe Face Landmarker on first use (or at server start-up)."""
    global _landmarker, _landmarker_error
    if _landmarker is not None or _landmarker_error is not None:
        return _landmarker
    with _landmarker_lock:
        if _landmarker is None and _landmarker_error is None:
            try:
                mp_python, vision = _import_mediapipe_tasks()
            except ImportError as e:
                _landmarker_error = f"MediaPipe is not installed in this Python environment ({e})"
            else:
                try:
                    options = vision.FaceLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model())),
                        num_faces=1,
                    )
                    _landmarker = vision.FaceLandmarker.create_from_options(options)
                except Exception as e:  # bad/missing model file, no network for the download…
                    _landmarker_error = f"MediaPipe Face Landmarker could not be loaded ({e})"
            if _landmarker_error:
                logger.warning("%s. %s", _landmarker_error, HAAR_FALLBACK_WARNING)
    return _landmarker


def landmarker_status() -> dict:
    """Is the primary face detector usable? Used by the health check and scripts/eval.py."""
    ready = _get_landmarker() is not None
    return {
        "mediapipe_ready": ready,
        "mediapipe_error": _landmarker_error,
        "degraded_reason": None if ready else f"{_landmarker_error}. {HAAR_FALLBACK_WARNING}",
    }


def _get_landmarks_from_tasks_api(image: np.ndarray):
    """Return MediaPipe landmarks (normalised x/y) for the first face, or None."""
    landmarker = _get_landmarker()
    if landmarker is None:
        return None
    import mediapipe as mp

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    with _landmarker_lock:
        result = landmarker.detect(mp_image)
    return result.face_landmarks[0] if result.face_landmarks else None


def _detect_face_box_haar(image: np.ndarray):
    """Largest Haar-cascade face box (x, y, w, h), or None."""
    global _haar_cascade
    if _haar_cascade is None:
        _haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if _haar_cascade.empty():
            logger.warning("Haar cascade file not found; Haar fallback disabled")
    if _haar_cascade.empty():
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _haar_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


# ── REGION GEOMETRY ──

def _inset(points: np.ndarray, factor: float) -> np.ndarray:
    centre = points.mean(axis=0)
    return centre + (points - centre) * factor


def landmark_polygons(landmarks, width: int, height: int) -> dict[str, np.ndarray]:
    """Pixel-space polygons (convex hulls) for every region from MediaPipe landmarks."""
    polygons = {}
    def point(entry):
        ids = entry if isinstance(entry, tuple) else (entry,)
        return [np.mean([landmarks[i].x for i in ids]) * width, np.mean([landmarks[i].y for i in ids]) * height]

    for name, entries in REGION_LANDMARKS.items():
        pts = np.array([point(e) for e in entries])
        hull = cv2.convexHull(_inset(pts, REGION_INSET).astype(np.float32)).reshape(-1, 2)
        polygons[name] = np.round(hull).astype(np.int32)
    return polygons


def haar_polygons(face_box: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    """Rectangles for every region, placed relative to the Haar face box."""
    x, y, w, h = face_box
    polygons = {}
    for name, (fx0, fy0, fx1, fy1) in HAAR_REGION_BOXES.items():
        x0, y0, x1, y1 = x + fx0 * w, y + fy0 * h, x + fx1 * w, y + fy1 * h
        polygons[name] = np.round([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype(np.int32)
    return polygons


def detect_region_polygons(frame: np.ndarray, use_haar_fallback: bool = True):
    """Return (polygons, source) where source is 'mediapipe' or 'haar'. Raises ValueError if no face."""
    height, width = frame.shape[:2]
    landmarks = _get_landmarks_from_tasks_api(frame)
    if landmarks is not None:
        return landmark_polygons(landmarks, width, height), "mediapipe"
    if use_haar_fallback:
        face_box = _detect_face_box_haar(frame)
        if face_box is not None:
            return haar_polygons(face_box), "haar"
    raise ValueError("No face detected. Try a well-lit, front-facing photo.")


# ── PIXEL SAMPLING ──

def _region_pixels(frame: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """BGR pixels inside `polygon`, minus deep shadows and highlights."""
    height, width = frame.shape[:2]
    polygon = np.clip(polygon, [0, 0], [width - 1, height - 1]).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    pixels = frame[mask > 0]
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    valid = (hsv[:, 2] >= HSV_VALUE_MIN) & (hsv[:, 2] <= HSV_VALUE_MAX)
    return pixels[valid]


def _dominant_lab(pixels: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Median LAB of the largest k-means cluster, and that cluster's share of pixels."""
    if len(pixels) < 3:
        raise ValueError("Region has too few usable pixels")
    sample = np.float32(pixels)
    cluster_count = max(1, min(k, len(sample)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)  # deterministic k-means initialisation → repeatable results
    _, labels, _ = cv2.kmeans(sample, cluster_count, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=cluster_count)
    kept = pixels[labels.ravel() == int(np.argmax(counts))]
    return np.median(bgr_to_lab(kept), axis=0), float(len(kept) / len(pixels))


def apply_white_balance(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance. Disabled by default; see APPLY_WHITE_BALANCE above."""
    img = image.astype(np.float32)
    means = np.maximum(img.reshape(-1, 3).mean(axis=0), 1.0)
    return np.clip(img * (128.0 / means), 0, 255).astype(np.uint8)


def _as_frame(image_or_frames: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """Accept one BGR frame, or several same-sized frames (averaged)."""
    if isinstance(image_or_frames, np.ndarray):
        return image_or_frames
    frames = list(image_or_frames)
    if not frames:
        raise ValueError("At least one image frame is required")
    if len({tuple(frame.shape) for frame in frames}) != 1:
        raise ValueError("All image frames must have the same shape")
    return np.mean(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def _median_lab(values: Iterable[dict]) -> tuple[float, float, float]:
    arr = np.array([[v["L"], v["a"], v["b"]] for v in values])
    L, a, b = np.median(arr, axis=0)
    return float(L), float(a), float(b)


def _drop_outlier_regions(skin_values: dict) -> dict:
    """Remove regions whose colour is far from the others (hair, shadow, glasses frame)."""
    if len(skin_values) < 3:
        return skin_values
    centre = np.array(_median_lab(skin_values.values()))
    kept = {
        name: v for name, v in skin_values.items()
        if np.linalg.norm(np.array([v["L"], v["a"], v["b"]]) - centre) <= REGION_OUTLIER_DELTA_E
    }
    return kept if len(kept) >= 2 else skin_values


# ── MAIN ENTRY POINT ──

def analyze_face_color(
    image_or_frames: np.ndarray | Sequence[np.ndarray],
    *,
    k: int = 3,
    use_haar_fallback: bool = True,
    apply_wb: bool | None = None,
) -> dict:
    """Measure skin colour from BGR image(s). Raises ValueError if no usable face is found."""
    frame = _as_frame(image_or_frames)
    if apply_wb is None:
        apply_wb = APPLY_WHITE_BALANCE
    if apply_wb:
        frame = apply_white_balance(frame)

    polygons, source = detect_region_polygons(frame, use_haar_fallback)

    region_values, region_confidences = {}, {}
    for name, polygon in polygons.items():
        pixels = _region_pixels(frame, polygon)
        if len(pixels) < 3:
            continue
        lab, dominant_fraction = _dominant_lab(pixels, k)
        region_values[name] = {"L": float(lab[0]), "a": float(lab[1]), "b": float(lab[2])}
        region_confidences[name] = min(1.0, len(pixels) / FULL_CONFIDENCE_PIXELS) * dominant_fraction

    skin_values = _drop_outlier_regions({n: v for n, v in region_values.items() if n in SKIN_REGIONS})
    if len(skin_values) < 2:
        raise ValueError("Not enough visible skin. Try a well-lit, front-facing photo without heavy shadows.")

    L_value, a_value, b_value = _median_lab(skin_values.values())
    ita = ita_degrees(L_value, b_value)
    ita_bucket, depth = map_ita_depth(ita)

    skin_confidences = [region_confidences[n] for n in skin_values]
    avg_confidence = float(np.mean(skin_confidences))
    # Unreliable regions → the L* spread is more likely noise than real contrast
    contrast = classify_contrast(skin_values) if avg_confidence >= MIN_CONFIDENCE_FOR_CONTRAST else "medium"

    skin_lab = np.array([[v["L"], v["a"], v["b"]] for v in skin_values.values()])
    dispersion = float(np.mean(np.std(skin_lab, axis=0)))
    confidence = avg_confidence / (1.0 + dispersion / DISPERSION_SCALE)
    if source == "haar":
        confidence *= HAAR_CONFIDENCE_PENALTY
    confidence = max(0.0, min(1.0, confidence))

    logger.debug("skin LAB=(%.1f, %.1f, %.1f) ITA=%.1f source=%s regions=%s",
                 L_value, a_value, b_value, ita, source, sorted(region_values))

    return {
        "L": L_value,
        "a": a_value,
        "b": b_value,
        "ita_degrees": float(ita),
        "ita_bucket": ita_bucket,
        "depth": depth,
        "undertone": classify_undertone(a_value, b_value),
        "clarity": classify_clarity(a_value, b_value),
        "contrast": contrast,
        "confidence": confidence,
        "landmark_source": source,
        "skin_regions_used": sorted(skin_values),
        "white_balance_applied": bool(apply_wb),
        "per_region_values": region_values,
    }
