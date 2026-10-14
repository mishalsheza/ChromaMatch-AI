"""sRGB → CIELAB conversion and CIEDE2000, checked against published reference values."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ai.colorimetry.color_space import hex_to_lab, srgb_to_lab, srgb_to_linear
from ai.recommendation.match import delta_e_2000

FIXTURES = Path(__file__).parent / "fixtures"


def test_srgb_linearisation_is_piecewise():
    # Below the 0.04045 knee the curve is linear: c / 12.92
    assert srgb_to_linear(0.04) == pytest.approx(0.04 / 12.92)
    # Above it: ((c + 0.055) / 1.055) ** 2.4 (the old bug used c ** 2.4)
    assert srgb_to_linear(0.5) == pytest.approx(((0.5 + 0.055) / 1.055) ** 2.4)
    assert srgb_to_linear(0.5) == pytest.approx(0.21404, abs=1e-5)


@pytest.mark.parametrize("hex_color, expected", [
    ("#FFFFFF", (100.0, 0.0, 0.0)),            # D65 white
    ("#000000", (0.0, 0.0, 0.0)),
    ("#808080", (53.585, 0.0, 0.0)),           # sRGB 128 grey → L* ≈ 53.59
    ("#FF0000", (53.24, 80.09, 67.20)),        # sRGB primaries (standard D65 values)
    ("#00FF00", (87.73, -86.18, 83.18)),
    ("#0000FF", (32.30, 79.19, -107.86)),
])
def test_hex_to_lab_reference_values(hex_color, expected):
    assert hex_to_lab(hex_color) == pytest.approx(expected, abs=0.02)


def test_matches_opencv_on_skin_tones():
    """Independent cross-check against OpenCV's float sRGB→Lab conversion.

    OpenCV uses lookup-table approximations (outputs are quantised to 1/64), so it
    agrees only to about 0.2; exact values are pinned by the reference test above.
    """
    rgb = np.array([[198, 156, 123], [141, 85, 36], [241, 194, 125], [92, 58, 40]], dtype=np.uint8)
    ours = srgb_to_lab(rgb)
    opencv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    np.testing.assert_allclose(ours, opencv, atol=0.25)


def test_short_hex_and_bad_hex():
    assert hex_to_lab("#fff") == pytest.approx((100.0, 0.0, 0.0), abs=0.02)
    with pytest.raises(ValueError):
        hex_to_lab("#12345")
    with pytest.raises(ValueError):
        hex_to_lab("not-a-colour")


def _sharma_pairs():
    rows = []
    for line in (FIXTURES / "ciede2000_sharma_testdata.txt").read_text().splitlines():
        if line.strip():
            values = [float(v) for v in line.split()]
            rows.append((tuple(values[0:3]), tuple(values[3:6]), values[6]))
    return rows


def test_delta_e_2000_all_sharma_reference_pairs():
    """All 34 pairs from Sharma, Wu & Dalal (2005), the standard CIEDE2000 test set."""
    pairs = _sharma_pairs()
    assert len(pairs) == 34
    for lab1, lab2, expected in pairs:
        assert delta_e_2000(lab1, lab2) == pytest.approx(expected, abs=1e-4)
        assert delta_e_2000(lab2, lab1) == pytest.approx(expected, abs=1e-4)  # symmetric


def test_delta_e_identical_colours_is_zero():
    assert delta_e_2000((60.0, 12.0, 20.0), (60.0, 12.0, 20.0)) == 0.0
