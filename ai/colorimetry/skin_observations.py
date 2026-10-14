"""
skin_observations.py — deterministic, cosmetic-only observations from regional LAB values.

Compares small regions (see face_color.OBSERVATION_REGIONS) against nearby skin:
  - under-eye vs cheeks: darker? which colour cast?
  - perioral (beside the mouth corners) vs chin
  - cheek redness vs forehead (a* difference)
  - tone evenness across all regions

These are colour-space differences, NOT diagnoses. Thresholds are hand-picked and
have not been validated against labelled data (see README "Limitations").
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── THRESHOLDS (ΔL* or Δa* in CIELAB units) ──
UNDER_EYE_THRESHOLDS = {
    'pronounced': 6.0,
    'mild': 3.0,
    'slight': 1.5,
}

PERIORAL_THRESHOLDS = {
    'pronounced': 4.0,
    'mild': 2.0,
}

REDNESS_THRESHOLD = 4.0
REDNESS_MILD_THRESHOLD = 2.0
UNEVEN_TONE_L_STD = 5.0

# Under-eye colour cast: a*/b* shift relative to the cheeks
BLUE_PURPLE_A_MIN, BLUE_PURPLE_B_MAX = 2.0, -1.0
BROWN_A_MIN, BROWN_B_MIN = 3.0, 3.0

REQUIRED_REGIONS = ('left_cheek', 'right_cheek', 'forehead', 'chin')


def _get_lab_values(region_data: Any) -> Optional[np.ndarray]:
    """Accept {'L','a','b'} dicts or 3-element sequences; return an array [L, a, b] or None."""
    if isinstance(region_data, dict):
        if all(k in region_data for k in ('L', 'a', 'b')):
            return np.array([region_data['L'], region_data['a'], region_data['b']], dtype=float)
        return None
    try:
        values = np.asarray(region_data, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return values[:3] if values.size >= 3 else None


def _severity(delta: float, thresholds: Dict[str, float]) -> Optional[str]:
    """Highest severity whose threshold `delta` reaches (thresholds ordered high → low)."""
    for name, threshold in thresholds.items():
        if delta >= threshold:
            return name
    return None


def detect_skin_observations(regions_lab: Dict[str, Any]) -> Dict[str, Any]:
    """Detect cosmetic observations from per-region LAB values."""
    observations = {
        'under_eye': None,
        'perioral': None,
        'redness': None,
        'tone_evenness': None,
        'has_observations': False,
    }

    lab = {name: v for name, v in ((n, _get_lab_values(d)) for n, d in regions_lab.items()) if v is not None}
    missing = [r for r in REQUIRED_REGIONS if r not in lab]
    if missing:
        logger.debug("Skipping skin observations, missing regions: %s", missing)
        return observations

    cheeks = (lab['left_cheek'] + lab['right_cheek']) / 2

    # ── 1. Under-eye darkness ──
    if 'left_under_eye' in lab and 'right_under_eye' in lab:
        under_eye = (lab['left_under_eye'] + lab['right_under_eye']) / 2
        severity = _severity(cheeks[0] - under_eye[0], UNDER_EYE_THRESHOLDS)
        if severity:
            a_delta, b_delta = under_eye[1] - cheeks[1], under_eye[2] - cheeks[2]
            if a_delta > BLUE_PURPLE_A_MIN and b_delta < BLUE_PURPLE_B_MAX:
                cast = 'blue-purple'
            elif a_delta > BROWN_A_MIN and b_delta > BROWN_B_MIN:
                cast = 'brown'
            else:
                cast = 'neutral'
            observations['under_eye'] = {'severity': severity, 'cast': cast}

    # ── 2. Perioral (beside the mouth corners) vs chin ──
    if 'left_perioral' in lab and 'right_perioral' in lab:
        perioral_L = (lab['left_perioral'][0] + lab['right_perioral'][0]) / 2
        severity = _severity(lab['chin'][0] - perioral_L, PERIORAL_THRESHOLDS)
        if severity:
            observations['perioral'] = {'severity': severity}

    # ── 3. Cheek redness vs forehead ──
    redness = cheeks[1] - lab['forehead'][1]
    if redness > REDNESS_THRESHOLD:
        observations['redness'] = {'severity': 'pronounced'}
    elif redness > REDNESS_MILD_THRESHOLD:
        observations['redness'] = {'severity': 'mild'}

    # ── 4. Tone evenness ──
    L_values = [v[0] for v in lab.values()]
    if len(L_values) >= 4:
        uneven = float(np.std(L_values)) > UNEVEN_TONE_L_STD
        observations['tone_evenness'] = 'uneven' if uneven else 'even'

    observations['has_observations'] = bool(
        observations['under_eye'] or observations['perioral'] or observations['redness']
        or observations['tone_evenness'] == 'uneven'
    )
    logger.debug("Skin observations: %s", observations)
    return observations
