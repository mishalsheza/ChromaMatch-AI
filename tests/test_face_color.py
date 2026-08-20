import math
import unittest

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


    def test_undertone_thresholds_are_explicit(self):
        self.assertEqual(classify_undertone(24.0, 29.0), "warm")
        self.assertEqual(classify_undertone(29.0, 24.0), "cool")
        self.assertEqual(classify_undertone(10.0, 11.0), "neutral_warm")
        self.assertEqual(classify_undertone(11.0, 10.0), "neutral_cool")
        self.assertEqual(classify_undertone(8.0, 11.0), "olive")


    def test_clarity_uses_lab_chroma_threshold(self):
        self.assertEqual(classify_clarity(16.0, 16.0), "clear")
        self.assertEqual(classify_clarity(10.0, 10.0), "muted")
        self.assertEqual(CHROMA_CLEAR_THRESHOLD, 22.0)


if __name__ == "__main__":
    unittest.main()
