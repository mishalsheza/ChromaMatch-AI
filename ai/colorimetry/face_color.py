"""
face_color.py — Automatic face color extraction
Compatible with MediaPipe 0.10.9+ (Tasks API)
"""

from __future__ import annotations

import math
import os
import urllib.request
from typing import Iterable, Sequence

import cv2
import numpy as np

# ── MEDIAPIPE TASKS API (new) ──
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Download model if not exists
    MODEL_PATH = "face_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading MediaPipe Face Landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("✅ Model downloaded!")
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    TASKS_API_AVAILABLE = True
    
except Exception as e:
    TASKS_API_AVAILABLE = False
    print(f"⚠️ MediaPipe Tasks API not available: {e}")
    print("   Falling back to Haar Cascade")

# ── FALLBACK: Haar Cascade ──
try:
    from face_detector_v2 import FaceDetector
    HAAR_AVAILABLE = True
except ImportError:
    HAAR_AVAILABLE = False


# ── FIXED LANDMARK INDICES ──
REGION_LANDMARKS = {
    "left_cheek": (117, 118, 119, 120, 121, 122, 123, 124),
    "right_cheek": (346, 347, 348, 349, 350, 351, 352, 353),
    "forehead": (10, 338, 297, 332, 284, 251, 389, 356, 127, 234, 93, 67, 109, 108),
    "jaw_chin": (150, 149, 176, 148, 152, 377, 400, 378, 379, 136, 172, 397, 365),
}

# ── EXPLICIT THRESHOLDS ──
HSV_VALUE_MIN = 35
HSV_VALUE_MAX = 245
ITA_BUCKETS = (
    (55.0, "very_light"),
    (41.0, "light"),
    (28.0, "intermediate"),
    (10.0, "tan"),
    (-30.0, "brown"),
)
CHROMA_CLEAR_THRESHOLD = 22.0

# ── FIXED: True olive detection ──
# Real olive skin = LOW a* (subdued redness) + elevated b* (yellow-green)
OLIVE_A_MAX = 6.0          # Low a* means NOT red/pink
OLIVE_B_MIN = 14.0         # Need real yellow-green presence
OLIVE_B_TO_A_RATIO = 2.2   # b* must strongly dominate a*

# ── CONTRAST THRESHOLDS ──
CONTRAST_HIGH_THRESHOLD = 8.0   # L* spread >= 8 = high contrast
CONTRAST_LOW_THRESHOLD = 3.0    # L* spread <= 3 = low contrast

UNDERTONE_DELTA_THRESHOLD = 2.0


def classify_undertone(a_value: float, b_value: float) -> str:
    """
    Classify undertone from LAB chroma direction.
    
    IMPORTANT: This function NEVER returns bare "neutral" — it always returns
    one of: "olive", "warm", "cool", "neutral_warm", "neutral_cool"
    """
    # ── FIXED: Olive detection ──
    # Olive: a* is LOW (skin looks muted, not pink/ruddy) 
    # AND b* is high with strong ratio
    if a_value <= OLIVE_A_MAX and b_value >= OLIVE_B_MIN and b_value >= a_value * OLIVE_B_TO_A_RATIO:
        return "olive"

    # ── WARM / COOL DETECTION ──
    delta = b_value - a_value
    
    if delta >= 5.0:
        return "warm"
    elif delta <= -5.0:
        return "cool"
    elif delta >= UNDERTONE_DELTA_THRESHOLD:
        return "neutral_warm"
    elif delta <= -UNDERTONE_DELTA_THRESHOLD:
        return "neutral_cool"
    else:
        # True neutral: lean on which side of zero
        return "neutral_warm" if b_value >= a_value else "neutral_cool"


def classify_contrast(region_values: dict) -> str:
    """
    Proxy for seasonal 'contrast' using within-skin L* spread across regions.
    """
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
    """Return Individual Typology Angle using CIELAB L* and b*."""
    if abs(b_value) < 1e-6:
        return 90.0 if lightness >= 50.0 else -90.0
    return math.degrees(math.atan((lightness - 50.0) / b_value))


def map_ita_depth(ita: float) -> tuple[str, str]:
    """Map published ITA bands to a detailed bucket and app depth category."""
    for threshold, bucket in ITA_BUCKETS:
        if ita >= threshold:
            break
    else:
        bucket = "dark"

    if bucket in {"very_light", "light"}:
        depth = "light"
    elif bucket in {"intermediate", "tan"}:
        depth = "medium"
    else:
        depth = "deep"
    return bucket, depth


def classify_clarity(a_value: float, b_value: float) -> str:
    """Use LAB chroma; >=22 is clear and below 22 is muted."""
    chroma = math.hypot(a_value, b_value)
    return "clear" if chroma >= CHROMA_CLEAR_THRESHOLD else "muted"


def _as_frame(image_or_frames: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """Convert input to a single BGR frame."""
    if isinstance(image_or_frames, np.ndarray):
        return image_or_frames
    frames = list(image_or_frames)
    if not frames:
        raise ValueError("At least one image frame is required")
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1:
        raise ValueError("All image frames must have the same shape")
    return np.mean(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def _get_landmarks_from_tasks_api(image):
    """Get face landmarks using MediaPipe Tasks API."""
    if not TASKS_API_AVAILABLE:
        return None
    
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = FACE_LANDMARKER.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
        
        return detection_result.face_landmarks[0]
        
    except Exception as e:
        print(f"⚠️ MediaPipe detection error: {e}")
        return None


def _get_landmarks_from_haar(image):
    """Get face regions using Haar Cascade (fallback)."""
    if not HAAR_AVAILABLE:
        return None
    
    try:
        detector = FaceDetector()
        region_colors, success, error, face_box = detector.detect_face_regions(image)
        
        if not success:
            return None
        
        class FakeLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        class FakeLandmarks:
            def __init__(self):
                self.landmark = []
                h, w = image.shape[:2]
                
                if face_box:
                    x, y, fw, fh = face_box
                    center_x = (x + fw/2) / w
                    center_y = (y + fh/2) / h
                    
                    landmark_positions = {
                        "left_cheek": (center_x - 0.15, center_y + 0.05),
                        "right_cheek": (center_x + 0.15, center_y + 0.05),
                        "forehead": (center_x, center_y - 0.3),
                        "jaw_chin": (center_x, center_y + 0.3),
                    }
                    
                    for region, (lx, ly) in landmark_positions.items():
                        for idx in REGION_LANDMARKS[region]:
                            self.landmark.append(FakeLandmark(lx, ly))
        
        return FakeLandmarks()
        
    except Exception as e:
        print(f"⚠️ Haar fallback error: {e}")
        return None


def _region_pixels(frame: np.ndarray, landmarks, indices: Iterable[int]) -> np.ndarray:
    """Extract pixels from a region using landmark indices."""
    height, width = frame.shape[:2]
    
    if hasattr(landmarks, 'landmark'):
        points = np.array([
            [int(landmarks.landmark[index].x * width), int(landmarks.landmark[index].y * height)]
            for index in indices
        ], dtype=np.int32)
    else:
        points = np.array([
            [int(landmarks[index].x * width), int(landmarks[index].y * height)]
            for index in indices
        ], dtype=np.int32)
    
    x_min, y_min = np.maximum(points.min(axis=0) - 2, [0, 0])
    x_max, y_max = np.minimum(points.max(axis=0) + 3, [width, height])
    crop = frame[y_min:y_max, x_min:x_max]
    
    if crop.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    
    polygon = points - np.array([x_min, y_min])
    hull = cv2.convexHull(polygon) 
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)   
    pixels = crop[mask > 0]
    
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    valid = (hsv[:, 2] >= HSV_VALUE_MIN) & (hsv[:, 2] <= HSV_VALUE_MAX)
    return pixels[valid]


def _dominant_lab(pixels: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Extract dominant LAB color from pixels using k-means."""
    if len(pixels) < 3:
        raise ValueError("Region has too few usable pixels")
    
    sample = np.float32(pixels)
    cluster_count = max(1, min(k, len(sample)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(sample, cluster_count, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    
    counts = np.bincount(labels.ravel(), minlength=cluster_count)
    dominant = int(np.argmax(counts))
    kept = pixels[labels.ravel() == dominant]
    
    lab = cv2.cvtColor(kept.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0), float(len(kept) / len(pixels))


def analyze_face_color(
    image_or_frames: np.ndarray | Sequence[np.ndarray],
    *,
    k: int = 3,
    use_haar_fallback: bool = True,
) -> dict:
    """
    Extract automatic face LAB color and classification fields from BGR image(s).
    """
    frame = _as_frame(image_or_frames)
    
    landmarks = _get_landmarks_from_tasks_api(frame)
    
    if landmarks is None and use_haar_fallback:
        landmarks = _get_landmarks_from_haar(frame)
    
    if landmarks is None:
        raise ValueError("No face detected (MediaPipe and Haar fallback both failed)")
    
    region_values = {}
    region_confidences = {}
    
    for region_name, indices in REGION_LANDMARKS.items():
        pixels = _region_pixels(frame, landmarks, indices)
        if len(pixels) > 0:
            lab, dominant_fraction = _dominant_lab(pixels, k)
            region_values[region_name] = {
                "L": float(lab[0]), "a": float(lab[1]), "b": float(lab[2])
            }
            region_confidences[region_name] = min(1.0, len(pixels) / 100.0) * dominant_fraction
    
    if not region_values:
        raise ValueError("No valid skin regions detected")
    
    lab_values = np.array([[v["L"], v["a"], v["b"]] for v in region_values.values()])
    median_lab = np.median(lab_values, axis=0)
    L_value, a_value, b_value = map(float, median_lab)
    
    ita = ita_degrees(L_value, b_value)
    ita_bucket, depth = map_ita_depth(ita)
    undertone = classify_undertone(a_value, b_value)
    clarity = classify_clarity(a_value, b_value)
    contrast = classify_contrast(region_values)  # ← NEW

    # ── TEMP DEBUG — remove after diagnosing ──
    chroma = math.hypot(a_value, b_value)
    l_spread = max(v["L"] for v in region_values.values()) - min(v["L"] for v in region_values.values())
    print(f"🔍 DEBUG chroma={chroma:.2f} (threshold={CHROMA_CLEAR_THRESHOLD}) -> clarity={clarity}")
    print(f"🔍 DEBUG L*-spread={l_spread:.2f} (high>={CONTRAST_HIGH_THRESHOLD}) -> contrast={contrast}")
    for region, conf in region_confidences.items():
        print(f"🔍 DEBUG region={region} confidence={conf:.3f}")
    # ── END TEMP DEBUG ──
    for region, vals in region_values.items():
        print(f"🔍 DEBUG region={region} L={vals['L']:.1f}")
    
    avg_confidence = float(np.mean(list(region_confidences.values()))) if region_confidences else 0.0
    contrast = classify_contrast(region_values)

# If per-region detection is unreliable, the L*-spread is more likely lighting
# noise than genuine seasonal contrast — don't let it drive the season result.
    MIN_CONFIDENCE_FOR_CONTRAST = 0.5
    if avg_confidence < MIN_CONFIDENCE_FOR_CONTRAST:
        contrast = "medium"
    dispersion = float(np.mean(np.std(lab_values, axis=0))) if len(lab_values) > 1 else 0.0
    confidence = max(0.0, min(1.0, avg_confidence * (1.0 / (1.0 + dispersion / 10.0))))
    
    return {
        "L": L_value,
        "a": a_value,
        "b": b_value,
        "ita_degrees": float(ita),
        "ita_bucket": ita_bucket,
        "depth": depth,
        "undertone": undertone,
        "clarity": clarity,
        "contrast": contrast,  # ← NEW
        "confidence": confidence,
        "per_region_values": region_values,
    }