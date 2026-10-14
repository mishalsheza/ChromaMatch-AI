import math
import unittest
from unittest.mock import patch

import numpy as np

import ai.colorimetry.face_color as face_color
from ai.colorimetry.face_color import (
    CHROMA_CLEAR_THRESHOLD,
    classify_clarity,
    classify_undertone,
    ita_degrees,
    map_ita_depth,
)


class FaceColorMathTests(unittest.TestCase):
    def test_ita_formula_matches_definition(self):
        expected = math.degrees(math.atan((60.0 - 50.0) / 20.0))
        self.assertAlmostEqual(ita_degrees(60.0, 20.0), expected)

    def test_ita_depth_buckets_collapse_to_three_depths(self):
        self.assertEqual(map_ita_depth(56.0), ("very_light", "light"))
        self.assertEqual(map_ita_depth(45.0), ("light", "light"))
        self.assertEqual(map_ita_depth(30.0), ("intermediate", "medium"))
        self.assertEqual(map_ita_depth(15.0), ("tan", "medium"))
        self.assertEqual(map_ita_depth(-5.0), ("brown", "deep"))
        self.assertEqual(map_ita_depth(-31.0), ("dark", "deep"))

    def test_undertone_uses_b_to_a_ratio(self):
        # warm: b*/a* >= 1.8, cool: <= 0.6, neutral in between
        self.assertEqual(classify_undertone(10.0, 18.0), "warm")
        self.assertEqual(classify_undertone(10.0, 6.0), "cool")
        self.assertEqual(classify_undertone(24.0, 29.0), "neutral")
        self.assertEqual(classify_undertone(10.0, 17.9), "neutral")
        self.assertEqual(classify_undertone(10.0, 6.1), "neutral")

    def test_undertone_with_zero_or_negative_a(self):
        # No measurable redness + yellow → warm (used to come out "cool" when a* < 0)
        self.assertEqual(classify_undertone(-0.5, 10.0), "warm")
        self.assertEqual(classify_undertone(0.0, -3.0), "cool")

    def test_classify_undertone_olive(self):
        self.assertEqual(classify_undertone(6.0, 14.0), "olive")
        self.assertEqual(classify_undertone(5.0, 14.0), "olive")
        self.assertEqual(classify_undertone(6.0, 15.0), "olive")
        self.assertEqual(classify_undertone(4.0, 16.0), "olive")

    def test_classify_undertone_not_olive(self):
        self.assertNotEqual(classify_undertone(6.1, 14.0), "olive")
        self.assertNotEqual(classify_undertone(6.0, 13.9), "olive")
        self.assertNotEqual(classify_undertone(7.0, 15.0), "olive")

    def test_classify_undertone_boundary_exact(self):
        self.assertEqual(classify_undertone(6.0, 14.0), "olive")
        # The ratio boundary alone is insufficient: b must also be >= 14.0.
        self.assertNotEqual(classify_undertone(6.0, 13.2), "olive")

    def test_clarity_uses_lab_chroma_threshold(self):
        self.assertEqual(classify_clarity(16.0, 16.0), "clear")
        self.assertEqual(classify_clarity(10.0, 10.0), "muted")
        self.assertEqual(CHROMA_CLEAR_THRESHOLD, 22.0)


class FaceColorPipelineTests(unittest.TestCase):
    def _skin_image(self, bgr=(100, 130, 160)):
        return np.full((160, 160, 3), bgr, dtype=np.uint8)

    def test_haar_fallback_regions_are_relative_to_the_face_box(self):
        image = self._skin_image()
        with patch.object(face_color, "_get_landmarks_from_tasks_api", return_value=None), \
                patch.object(face_color, "_detect_face_box_haar", return_value=(20, 20, 120, 120)):
            result = face_color.analyze_face_color(image)

        self.assertEqual(result["landmark_source"], "haar")
        self.assertEqual(set(result["skin_regions_used"]), set(face_color.SKIN_REGIONS))
        # Uniform image → measured colour equals the image colour (BGR 100,130,160 → RGB 160,130,100)
        expected = face_color.bgr_to_lab(np.array([100, 130, 160]))
        self.assertAlmostEqual(result["L"], expected[0], places=3)
        self.assertAlmostEqual(result["b"], expected[2], places=3)

    def test_haar_boxes_scale_with_face_size(self):
        small = face_color.haar_polygons((100, 100, 50, 50))["left_cheek"]
        large = face_color.haar_polygons((0, 0, 200, 200))["left_cheek"]
        self.assertGreater(np.ptp(large[:, 0]), 3 * np.ptp(small[:, 0]))
        self.assertTrue((small[:, 0] >= 100).all() and (small[:, 0] <= 150).all())

    def test_no_face_raises_value_error(self):
        with patch.object(face_color, "_get_landmarks_from_tasks_api", return_value=None), \
                patch.object(face_color, "_detect_face_box_haar", return_value=None):
            with self.assertRaises(ValueError):
                face_color.analyze_face_color(self._skin_image())

    def test_white_balance_is_off_by_default(self):
        self.assertFalse(face_color.APPLY_WHITE_BALANCE)
        with patch.object(face_color, "_get_landmarks_from_tasks_api", return_value=None), \
                patch.object(face_color, "_detect_face_box_haar", return_value=(20, 20, 120, 120)), \
                patch.object(face_color, "apply_white_balance", side_effect=AssertionError("WB ran")):
            result = face_color.analyze_face_color(self._skin_image())
        self.assertFalse(result["white_balance_applied"])

    def test_outlier_region_is_dropped(self):
        regions = {
            "forehead": {"L": 20.0, "a": 4.0, "b": 6.0},     # hair over the forehead
            "left_cheek": {"L": 65.0, "a": 14.0, "b": 20.0},
            "right_cheek": {"L": 68.0, "a": 13.0, "b": 19.0},
            "chin": {"L": 63.0, "a": 15.0, "b": 21.0},
        }
        kept = face_color._drop_outlier_regions(regions)
        self.assertEqual(set(kept), {"left_cheek", "right_cheek", "chin"})


if __name__ == "__main__":
    unittest.main()
