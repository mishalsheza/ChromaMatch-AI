import itertools
import unittest

from ai.recommendation.season import (
    CLOSE_CALL_MARGIN_THRESHOLD,
    get_ranked_season_recommendations,
    load_season_reference,
    score_all_seasons,
)

ALL_INPUTS = list(itertools.product(
    ["warm", "cool", "neutral_warm", "neutral_cool", "olive"],
    ["light", "medium", "deep"],
    ["clear", "muted", "medium"],
    ["low", "medium", "high"],
))


class SeasonRankingTests(unittest.TestCase):
    def test_close_call_uses_top_two_margin(self):
        result = get_ranked_season_recommendations(
            {
                "undertone": "warm",
                "depth": "light",
                "clarity": "clear",
                "contrast": "medium",
            },
            top_n=3,
        )
        self.assertLess(result["margin_pct"], CLOSE_CALL_MARGIN_THRESHOLD)
        self.assertTrue(result["is_close_call"])

    def test_wider_margin_is_not_close_call(self):
        result = get_ranked_season_recommendations(
            {
                "undertone": "warm",
                "depth": "medium",
                "clarity": "clear",
                "contrast": "medium",
            },
            top_n=3,
        )
        self.assertGreaterEqual(result["margin_pct"], CLOSE_CALL_MARGIN_THRESHOLD)
        self.assertFalse(result["is_close_call"])

    def test_every_season_can_win(self):
        winners = {score_all_seasons(*combo)[0]["season_key"] for combo in ALL_INPUTS}
        self.assertEqual(winners, set(load_season_reference()["seasons"]))

    def test_ties_at_the_top_are_flagged_as_close_calls(self):
        """Known limitation: ties are broken by JSON order, so they must be reported as close calls."""
        ties = 0
        for u, d, c, ct in ALL_INPUTS:
            result = get_ranked_season_recommendations(
                {"undertone": u, "depth": d, "clarity": c, "contrast": ct}, top_n=3
            )
            if result["margin"] == 0:
                ties += 1
                self.assertTrue(result["is_close_call"])
        self.assertGreater(ties, 0)  # documented in README; update both if the scoring changes

    def test_neutral_is_split_by_a_b_direction(self):
        warmish = get_ranked_season_recommendations({"undertone": "neutral", "a": 10, "b": 14, "depth": "medium"})
        coolish = get_ranked_season_recommendations({"undertone": "neutral", "a": 14, "b": 10, "depth": "medium"})
        self.assertNotEqual(warmish["winner_key"], coolish["winner_key"])


if __name__ == "__main__":
    unittest.main()
