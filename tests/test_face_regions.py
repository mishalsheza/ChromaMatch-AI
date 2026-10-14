"""
The skin-region landmark indices are checked against MediaPipe's *official* data:
  - the canonical face mesh (ai/data/mediapipe/canonical_face_model.obj, from the MediaPipe repo)
  - the eye / eyebrow / lips / nose / face-oval landmark sets shipped in the mediapipe package

Canonical coordinates: +x = subject's left, +y = up, units ≈ cm.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ai.colorimetry.face_color import OBSERVATION_REGIONS, REGION_LANDMARKS, SKIN_REGIONS

ROOT = Path(__file__).resolve().parents[1]
MESH = np.array([
    [float(v) for v in line.split()[1:4]]
    for line in (ROOT / "ai/data/mediapipe/canonical_face_model.obj").read_text().splitlines()
    if line.startswith("v ")
])

connections = pytest.importorskip("mediapipe.tasks.python.vision").FaceLandmarksConnections


def _landmark_set(name):
    return {i for edge in getattr(connections, f"FACE_LANDMARKS_{name}") for i in (edge.start, edge.end)}


FEATURES = {
    name: _landmark_set(name)
    for name in ["LEFT_EYE", "RIGHT_EYE", "LEFT_EYEBROW", "RIGHT_EYEBROW", "LIPS", "NOSE"]
}


def _points(entries):
    """Canonical (x, y) for region entries; (i, j) means the midpoint of i and j."""
    pts = []
    for e in entries:
        ids = e if isinstance(e, tuple) else (e,)
        pts.append(MESH[list(ids), :2].mean(axis=0))
    return np.array(pts)


def _ids(entries):
    return {i for e in entries for i in (e if isinstance(e, tuple) else (e,))}


def _feature_xy(name):
    return MESH[sorted(FEATURES[name]), :2]


def test_mesh_has_468_vertices():
    assert MESH.shape == (468, 3)


def test_all_indices_valid():
    for name, entries in REGION_LANDMARKS.items():
        assert all(0 <= i < 468 for i in _ids(entries)), name
        assert len(entries) >= 5, f"{name}: too few points for a usable polygon"


def test_regions_are_not_runs_of_consecutive_indices():
    """The old bug: index lists like 145..159 that mix eye, chin and edge points."""
    for name, entries in REGION_LANDMARKS.items():
        ids = sorted(_ids(entries))
        longest = run = 1
        for a, b in zip(ids, ids[1:], strict=False):
            run = run + 1 if b == a + 1 else 1
            longest = max(longest, run)
        assert longest <= 3, name


@pytest.mark.parametrize("region", sorted(REGION_LANDMARKS))
def test_regions_avoid_eyes_brows_lips(region):
    """No region uses a feature landmark, and no feature landmark lies inside a region's hull."""
    entries = REGION_LANDMARKS[region]
    hull = cv2.convexHull(_points(entries).astype(np.float32))
    for feature, ids in FEATURES.items():
        assert not (_ids(entries) & ids), f"{region} uses {feature} landmarks"
        for x, y in MESH[sorted(ids), :2]:
            assert cv2.pointPolygonTest(hull, (float(x), float(y)), False) < 0, f"{feature} point inside {region}"


def test_anatomical_positions():
    c = {name: _points(entries) for name, entries in REGION_LANDMARKS.items()}
    eye_bottom = min(_feature_xy("LEFT_EYE")[:, 1].min(), _feature_xy("RIGHT_EYE")[:, 1].min())
    lips = _feature_xy("LIPS")
    face_top = MESH[10, 1]

    # Forehead: centred, between the brows' inner ends and the hairline row
    assert abs(c["forehead"][:, 0].mean()) < 0.1
    assert c["forehead"][:, 1].min() > MESH[9, 1] - 0.01   # nothing below the glabella
    assert c["forehead"][:, 1].max() < face_top - 0.5       # clear of the hairline row

    # Under-eye: below the eye contour, above the cheek
    for side in ("left", "right"):
        ue = c[f"{side}_under_eye"]
        assert ue[:, 1].max() < eye_bottom
        assert ue[:, 1].min() > c[f"{side}_cheek"][:, 1].max()

    # Cheeks: lateral to the nose, between the under-eye band and the mouth corners
    assert c["left_cheek"][:, 0].min() > 2.0 and c["right_cheek"][:, 0].max() < -2.0
    for side in ("left", "right"):
        assert c[f"{side}_cheek"][:, 1].min() > lips[:, 1].max()

    # Chin: below the lower lip
    assert c["chin"][:, 1].max() < lips[:, 1].min()
    assert abs(c["chin"][:, 0].mean()) < 0.1

    # Perioral: just outside the mouth corners
    assert c["left_perioral"][:, 0].min() > lips[:, 0].max()
    assert c["right_perioral"][:, 0].max() < lips[:, 0].min()


@pytest.mark.parametrize("left", [n for n in REGION_LANDMARKS if n.startswith("left_")])
def test_left_and_right_regions_are_mirror_images(left):
    right = left.replace("left_", "right_")
    mirrored = _points(REGION_LANDMARKS[left]) * [-1, 1]
    np.testing.assert_allclose(sorted(map(tuple, mirrored)), sorted(map(tuple, _points(REGION_LANDMARKS[right]))), atol=1e-3)


def test_skin_and_observation_regions_are_separate():
    assert set(SKIN_REGIONS).isdisjoint(OBSERVATION_REGIONS)
    assert set(SKIN_REGIONS) == {"forehead", "left_cheek", "right_cheek", "chin"}
