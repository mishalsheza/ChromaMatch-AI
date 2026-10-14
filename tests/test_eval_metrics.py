"""Metric helpers in scripts/eval.py (synthetic data only; no images, no network)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("eval_script", Path(__file__).resolve().parents[1] / "scripts/eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def test_confusion_matrix_rows_are_truth():
    truth = ["warm", "warm", "cool", "olive"]
    pred = ["warm", "cool", "cool", "warm"]
    m = ev.confusion_matrix(truth, pred, ["warm", "neutral", "cool", "olive"])
    assert m == [
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ]


def test_accuracy_and_majority_baseline():
    truth = ["deep", "deep", "deep", "light"]
    assert ev.accuracy(truth, ["deep", "light", "deep", "light"]) == 0.75
    assert ev.majority_baseline(truth) == ("deep", 0.75)


def test_wilson_interval_is_wide_for_small_samples():
    lo, hi = ev.wilson_interval(20, 40)
    assert lo == pytest.approx(0.352, abs=0.002)
    assert hi == pytest.approx(0.648, abs=0.002)
    assert ev.wilson_interval(0, 10)[0] == 0.0


def test_perturb_lighting_identity_and_direction():
    img = np.full((4, 4, 3), (90, 140, 200), dtype=np.uint8)  # BGR
    assert np.array_equal(ev.perturb_lighting(img, (1.0, 1.0, 1.0)), img)
    warmer = ev.perturb_lighting(img, (1.1, 1.0, 0.88))
    assert warmer[0, 0, 2] > img[0, 0, 2] and warmer[0, 0, 0] < img[0, 0, 0]  # more red, less blue


def test_per_class_recall_and_precision():
    truth = ["olive", "olive", "warm", "warm", "warm", "cool"]
    pred = ["olive", "warm", "warm", "warm", "olive", "cool"]
    m = ev.per_class_metrics(truth, pred, ["warm", "neutral", "cool", "olive"])
    assert (m["olive"]["n"], m["olive"]["correct"], m["olive"]["recall"]) == (2, 1, 0.5)
    assert m["olive"]["precision"] == 0.5          # predicted olive twice, right once
    assert m["warm"]["recall"] == pytest.approx(2 / 3)
    assert m["neutral"]["n"] == 0 and np.isnan(m["neutral"]["recall"])


def test_balanced_accuracy_is_not_fooled_by_the_majority_class():
    truth = ["cool"] * 9 + ["olive"]
    always_cool = ["cool"] * 10
    assert ev.accuracy(truth, always_cool) == 0.9
    assert ev.balanced_accuracy(truth, always_cool, ev.UNDERTONES) == 0.5   # = 1/k for k=2 present classes


def test_classes_needing_more():
    from collections import Counter
    counts = Counter({"cool": 20, "warm": 8, "neutral": 5, "unsure": 3})
    assert ev.classes_needing_more(counts, ev.UNDERTONES, 8) == {"neutral": 3, "olive": 8}


def test_select_topup_targets_short_classes_and_mixes_in_random_images():
    import random
    pool = {f"img{i}": ("olive" if i < 6 else "cool") for i in range(40)}
    picks = ev.select_topup(pool, {"olive": 2}, random.Random(0), oversample=2.0)
    targeted = [p for p, sel in picks if sel == "targeted:olive"]
    randoms = [p for p, sel in picks if sel == "random"]
    assert len(targeted) == 4 and all(pool[p] == "olive" for p in targeted)
    assert len(randoms) == 4                                   # blinding: as many random as targeted
    assert len({p for p, _ in picks}) == len(picks)            # no duplicates


def test_select_topup_when_too_few_candidates_exist():
    import random
    pool = {"a": "olive", "b": "cool", "c": "cool"}
    picks = ev.select_topup(pool, {"olive": 8}, random.Random(0), random_extra=0)
    assert picks == [("a", "targeted:olive")]
