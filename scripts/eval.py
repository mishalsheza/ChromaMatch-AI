#!/usr/bin/env python
"""
Honest evaluation for ShadeSense: hand labels, baselines, and a lighting-stability check.

Commands
  sample     Copy a balanced random sample of images to ai/data/eval_images/ under neutral
             names (img_001.jpg …) so you can label them blind, and write eval/labels.csv.
  topup      After labelling: add images until every undertone class has at least
             --min-per-class labels. Stays blind (see cmd_topup).
  run        Compare the pipeline with your hand labels: overall and per-class accuracy
             (recall), precision, balanced accuracy, confusion matrices, and baselines.
  stability  Re-run the pipeline on brightness / colour-temperature variations of the same
             photo and report how much the outputs move.

Examples
  python scripts/eval.py sample --source ai/data/raw --n 40
  python scripts/eval.py topup --min-per-class 8
  python scripts/eval.py run --labels eval/labels.csv
  python scripts/eval.py stability --images ai/data/eval_images --limit 40

No network or LLM calls. Everything here is the deterministic colour pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import logging
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.colorimetry.color_space import srgb_to_linear  # noqa: E402
from ai.colorimetry.face_color import analyze_face_color, landmarker_status  # noqa: E402
from ai.recommendation.match import delta_e_2000  # noqa: E402

DEPTHS = ["light", "medium", "deep"]
UNDERTONES = ["warm", "neutral", "cool", "olive"]
LABEL_COLUMNS = ["image", "depth", "undertone", "labeler", "notes"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
NOT_LABELS = {"", "unsure"}

# The manifest maps img_###.jpg back to the source file and records HOW each image was
# chosen. It lives next to the (git-ignored) images, not in eval/, because the source
# paths contain the old heuristic folder names: don't open it while labelling.
MANIFEST_NAME = "manifest.csv"
MANIFEST_COLUMNS = ["image", "original_path", "selection"]
# "initial"/"random" rows are an unbiased sample → recall and overall accuracy use only these.
# "targeted:<class>" rows were picked because the pipeline predicted <class> → fine for that
# class's precision, but they would inflate recall, so they're excluded from it.
UNBIASED_SELECTIONS = {"initial", "random"}

# Lighting perturbations, applied in *linear* RGB (physically: exposure and illuminant colour).
# Gains are (R, G, B).
PERTURBATIONS = {
    "darker_-0.5EV": (0.71, 0.71, 0.71),
    "brighter_+0.5EV": (1.41, 1.41, 1.41),
    "warmer_light": (1.10, 1.00, 0.88),
    "cooler_light": (0.90, 1.00, 1.12),
    "much_warmer_light": (1.20, 1.00, 0.78),
    "much_cooler_light": (0.80, 1.00, 1.25),
}


# ── metrics (pure functions; unit-tested in tests/test_eval_metrics.py) ──

def confusion_matrix(truth: list[str], pred: list[str], classes: list[str]) -> list[list[int]]:
    """Rows = true class, columns = predicted class. Predictions outside `classes` are ignored."""
    index = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for t, p in zip(truth, pred, strict=True):
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1
    return matrix


def accuracy(truth: list[str], pred: list[str]) -> float:
    return sum(t == p for t, p in zip(truth, pred, strict=True)) / len(truth) if truth else float("nan")


def majority_baseline(truth: list[str]) -> tuple[str, float]:
    """Accuracy of always predicting the most common true label."""
    label, count = Counter(truth).most_common(1)[0]
    return label, count / len(truth)


def per_class_metrics(truth: list[str], pred: list[str], classes: list[str]) -> dict[str, dict]:
    """Per class: n (true count), correct, recall (= per-class accuracy), predicted count, precision."""
    out = {}
    for c in classes:
        n = sum(t == c for t in truth)
        predicted = sum(p == c for p in pred)
        correct = sum(t == c and p == c for t, p in zip(truth, pred, strict=True))
        out[c] = {
            "n": n,
            "correct": correct,
            "recall": correct / n if n else float("nan"),
            "predicted": predicted,
            "precision": correct / predicted if predicted else float("nan"),
        }
    return out


def balanced_accuracy(truth: list[str], pred: list[str], classes: list[str]) -> float:
    """Mean recall over the classes present. Unlike plain accuracy it can't be inflated by
    doing well only on the most common class: any constant guess scores 1/k."""
    recalls = [m["recall"] for m in per_class_metrics(truth, pred, classes).values() if m["n"]]
    return sum(recalls) / len(recalls) if recalls else float("nan")


def classes_needing_more(counts: Counter, classes: list[str], minimum: int) -> dict[str, int]:
    """How many more labelled examples each class needs to reach `minimum`."""
    return {c: minimum - counts.get(c, 0) for c in classes if counts.get(c, 0) < minimum}


def select_topup(
    pool_predictions: dict[str, str],
    needed: dict[str, int],
    rng: random.Random,
    oversample: float = 2.0,
    random_extra: int | None = None,
) -> list[tuple[str, str]]:
    """Pick (path, selection) pairs: for each short class, random images the pipeline predicts
    as that class (oversampled, since some will turn out to be another class), plus purely
    random images so the targeted ones can't be told apart. Returned in shuffled order."""
    chosen: list[tuple[str, str]] = []
    taken: set[str] = set()
    for cls, missing in needed.items():
        candidates = sorted(p for p, pred in pool_predictions.items() if pred == cls and p not in taken)
        picks = rng.sample(candidates, min(len(candidates), math.ceil(missing * oversample)))
        chosen += [(p, f"targeted:{cls}") for p in picks]
        taken.update(picks)
    n_random = len(chosen) if random_extra is None else random_extra
    rest = sorted(p for p in pool_predictions if p not in taken)
    chosen += [(p, "random") for p in rng.sample(rest, min(n_random, len(rest)))]
    rng.shuffle(chosen)
    return chosen


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion. With n≈40 it is wide, and that's the honest part."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def perturb_lighting(image_bgr: np.ndarray, gains_rgb: tuple[float, float, float]) -> np.ndarray:
    """Scale each channel in linear light, then re-encode to sRGB."""
    linear = srgb_to_linear(image_bgr[..., ::-1] / 255.0) * np.array(gains_rgb)
    linear = np.clip(linear, 0, 1)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return np.round(srgb[..., ::-1] * 255).astype(np.uint8)


# ── helpers ──

def require_mediapipe(allow_haar: bool) -> bool:
    """Evaluation numbers are only meaningful with MediaPipe landmarks. Without it the
    pipeline silently uses coarse Haar boxes, so refuse unless explicitly allowed."""
    status = landmarker_status()
    if status["mediapipe_ready"]:
        return True
    if allow_haar:
        print(f"WARNING: {status['degraded_reason']}\n"
              "Continuing because --allow-haar was given. Don't report these numbers as the pipeline's accuracy.\n",
              file=sys.stderr)
        return True
    print(f"ERROR: {status['degraded_reason']}\n"
          "Refusing to run: results would not reflect the real pipeline. Run with the project's .venv\n"
          "(`source .venv/bin/activate`, `pip install -r requirements-dev.txt`), or pass --allow-haar\n"
          "if you really want Haar-fallback numbers.", file=sys.stderr)
    return False


def _analyze(image: np.ndarray) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return analyze_face_color(image)


def _find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _read_manifest(image_dir: Path) -> list[dict]:
    path = image_dir / MANIFEST_NAME
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_manifest(image_dir: Path, rows: list[dict]) -> None:
    with open(image_dir / MANIFEST_NAME, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def _fmt_pct(x: float) -> str:
    return "  n/a" if math.isnan(x) else f"{x:5.0%}"


def _format_matrix(matrix: list[list[int]], classes: list[str]) -> str:
    width = max(len(c) for c in classes) + 2
    lines = ["true \\ predicted".ljust(width + 6) + "".join(c[:8].rjust(9) for c in classes)]
    for c, row in zip(classes, matrix, strict=True):
        lines.append(c.ljust(width + 6) + "".join(str(v).rjust(9) for v in row))
    return "\n".join(lines)


# ── commands ──

def cmd_sample(args) -> int:
    """Stratified sample across the source sub-folders, copied under anonymous names."""
    source, out_dir, labels_path = Path(args.source), Path(args.out), Path(args.labels)
    if labels_path.exists() and not args.force:
        print(f"{labels_path} already exists; use --force to overwrite (this would discard your labels).")
        return 1
    groups = {d: _find_images(d) for d in sorted(source.iterdir()) if d.is_dir()}
    groups = {d: imgs for d, imgs in groups.items() if imgs}
    if not groups:
        print(f"No images found under {source}")
        return 1

    rng = random.Random(args.seed)
    # Group folders by depth prefix (light/medium/deep) so the sample is balanced across tones.
    # Folder names are only used to spread the sample; they are NOT used as labels.
    by_depth: dict[str, list[Path]] = {}
    for folder, imgs in groups.items():
        by_depth.setdefault(folder.name.split("_")[0], []).extend(imgs)
    per_group = math.ceil(args.n / len(by_depth))
    chosen: list[Path] = []
    for imgs in by_depth.values():
        chosen += rng.sample(imgs, min(per_group, len(imgs)))
    rng.shuffle(chosen)
    chosen = chosen[: args.n]

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with open(labels_path, "w", newline="") as lf:
        labels = csv.writer(lf)
        labels.writerow(LABEL_COLUMNS)
        for i, src in enumerate(chosen, 1):
            name = f"img_{i:03d}{src.suffix.lower()}"
            shutil.copy(src, out_dir / name)
            labels.writerow([name, "", "", "", ""])
            manifest.append({"image": name, "original_path": _rel(src), "selection": "initial"})
    _write_manifest(out_dir, manifest)
    print(f"Copied {len(chosen)} images to {out_dir}")
    print(f"Fill in {labels_path} by hand (see eval/README.md).")
    return 0


def cmd_topup(args) -> int:
    """Add blind images until each undertone class has --min-per-class labels.

    Only the total number added is printed. Which images were targeted at which class
    is recorded in the hidden manifest, so the labeller can't be anchored by it.
    """
    labels_path, image_dir, source = Path(args.labels), Path(args.images), Path(args.source)
    with open(labels_path, newline="") as f:
        rows = list(csv.DictReader(f))
    unlabelled = [r["image"] for r in rows if not (r.get("undertone") or "").strip()]
    if unlabelled:
        print(f"{len(unlabelled)} of {len(rows)} rows in {labels_path} have no undertone label yet "
              f"(first: {unlabelled[0]}). Label them, or mark them 'unsure', then run topup again.")
        return 1

    counts = Counter((r["undertone"] or "").strip().lower() for r in rows)
    needed = classes_needing_more(counts, UNDERTONES, args.min_per_class)
    if not needed:
        print(f"Every undertone class already has at least {args.min_per_class} labels. Nothing to add.")
        return 0

    manifest = _read_manifest(image_dir)
    used = {m["original_path"] for m in manifest}
    pool = [p for p in _find_images(source) if _rel(p) not in used]
    print(f"Classes still short: {len(needed)}. Running the pipeline on {len(pool)} unused images to find candidates…")
    predictions = {}
    for path in pool:
        image = cv2.imread(str(path))
        if image is None:
            continue
        try:
            predictions[_rel(path)] = _analyze(image)["undertone"]
        except ValueError:
            continue   # no face: useless for labelling

    rng = random.Random(f"{args.seed}:{len(rows)}")
    picks = select_topup(predictions, needed, rng, oversample=args.oversample, random_extra=args.random_extra)
    if not picks:
        print("No usable images left in the pool.")
        return 1

    next_number = max((int(Path(r["image"]).stem.split("_")[-1]) for r in rows), default=0) + 1
    with open(labels_path, "a", newline="") as f:
        writer = csv.writer(f)
        for offset, (src, selection) in enumerate(picks):
            name = f"img_{next_number + offset:03d}{Path(src).suffix.lower()}"
            shutil.copy(ROOT / src, image_dir / name)
            writer.writerow([name, "", "", "", ""])
            manifest.append({"image": name, "original_path": src, "selection": selection})
    _write_manifest(image_dir, manifest)

    short_for = sorted(c for c in needed if not any(p == c for p in predictions.values()))
    print(f"Added {len(picks)} images (img_{next_number:03d} onwards, shuffled) to {labels_path}.")
    if short_for:
        print(f"Note: the pipeline predicts no remaining image as {', '.join(short_for)}, so those classes "
              "can only be found by chance. Consider adding photos from another source.")
    print("Label them blind, then run `topup` again: it repeats until every class has enough.")
    return 0


def cmd_run(args) -> int:
    labels_path, image_dir = Path(args.labels), Path(args.images)
    with open(labels_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("depth") or r.get("undertone"))]
    if not rows:
        print(f"No labelled rows in {labels_path} yet. Fill in depth/undertone first.")
        return 1

    results = []
    for row in rows:
        image = cv2.imread(str(image_dir / row["image"]))
        if image is None:
            print(f"  skip {row['image']}: not found in {image_dir}")
            continue
        try:
            pred = _analyze(image)
        except ValueError as e:
            pred = {"depth": "no_face", "undertone": "no_face"}
            print(f"  {row['image']}: {e}")
        results.append((row, pred))

    n_haar = sum(1 for _, p in results if p.get("landmark_source") == "haar")
    if n_haar:
        print(f"\nWARNING: {n_haar} of {len(results)} images fell back to the coarse Haar face box "
              "(MediaPipe found no face in them). Their results are less reliable.")

    selection = {m["image"]: m["selection"] for m in _read_manifest(image_dir)}
    n_targeted = sum(1 for r, _ in results if selection.get(r["image"], "initial") not in UNBIASED_SELECTIONS)
    print(f"\nEvaluated {len(results)} labelled images ({labels_path})")
    if n_targeted:
        print(f"  {n_targeted} were added by `topup` because the pipeline predicted a rare class. They count towards")
        print("  precision only; overall accuracy, recall and the confusion matrix use the unbiased sample.")
    print()

    for field, classes, random_rate in (("depth", DEPTHS, 1 / 3), ("undertone", UNDERTONES, 1 / 4)):
        labelled = [(r, p) for r, p in results if (r.get(field) or "").strip().lower() in classes]
        if not labelled:
            continue
        everything = [(r[field].strip().lower(), p[field]) for r, p in labelled]
        unbiased = [(r[field].strip().lower(), p[field]) for r, p in labelled
                    if selection.get(r["image"], "initial") in UNBIASED_SELECTIONS]
        truth, pred = [t for t, _ in unbiased], [p for _, p in unbiased]
        if not truth:
            continue
        correct = sum(t == p for t, p in unbiased)
        lo, hi = wilson_interval(correct, len(unbiased))
        maj_label, maj_acc = majority_baseline(truth)
        recall = per_class_metrics(truth, pred, classes)
        precision = per_class_metrics([t for t, _ in everything], [p for _, p in everything], classes)

        print(f"── {field.upper()} (n={len(unbiased)} unbiased" + (f", {len(everything)} incl. targeted)" if len(everything) != len(unbiased) else ")") + " ──")
        print(f"  overall accuracy  : {correct}/{len(unbiased)} = {correct / len(unbiased):.0%}  (95% CI {lo:.0%}–{hi:.0%})")
        print(f"  balanced accuracy : {balanced_accuracy(truth, pred, classes):.0%}  (mean of per-class recall; any constant guess = {1 / len([c for c in classes if recall[c]['n']]):.0%})")
        print(f"  majority baseline : {maj_acc:.0%}  (always '{maj_label}')")
        print(f"  random baseline   : {random_rate:.0%}  ({len(classes)} classes)")
        print()
        print(f"  {'class':9s} {'labelled':>8s} {'recall (per-class acc.)':>24s} {'95% CI':>11s} {'precision':>10s}")
        for c in classes:
            r, pr = recall[c], precision[c]
            rlo, rhi = wilson_interval(r["correct"], r["n"])
            ci = "" if not r["n"] else f"{rlo:.0%}–{rhi:.0%}"
            flag = "  ← too few to judge" if precision[c]["n"] < 5 else ""
            print(f"  {c:9s} {precision[c]['n']:8d} {r['correct']:>12d}/{r['n']:<3d} {_fmt_pct(r['recall'])} "
                  f"{ci:>11s} {_fmt_pct(pr['precision'])} ({pr['correct']}/{pr['predicted']}){flag}")
        print()
        print(_format_matrix(confusion_matrix(truth, pred, classes), classes))
        failures = Counter(p for p in pred if p not in classes)
        if failures:
            print(f"  not classified    : {dict(failures)}")
        print()
    return 0


def cmd_stability(args) -> int:
    images = _find_images(Path(args.images))
    random.Random(args.seed).shuffle(images)
    images = images[: args.limit]
    if not images:
        print(f"No images under {args.images}")
        return 1

    stats = {name: {"de": [], "ita": [], "depth_flip": 0, "tone_flip": 0, "n": 0} for name in PERTURBATIONS}
    used = 0
    for path in images:
        image = cv2.imread(str(path))
        try:
            base = _analyze(image)
        except (ValueError, TypeError):
            continue
        used += 1
        base_lab = (base["L"], base["a"], base["b"])
        for name, gains in PERTURBATIONS.items():
            try:
                out = _analyze(perturb_lighting(image, gains))
            except ValueError:
                continue
            s = stats[name]
            s["n"] += 1
            s["de"].append(delta_e_2000(base_lab, (out["L"], out["a"], out["b"])))
            s["ita"].append(abs(out["ita_degrees"] - base["ita_degrees"]))
            s["depth_flip"] += out["depth"] != base["depth"]
            s["tone_flip"] += out["undertone"] != base["undertone"]

    print(f"\nLighting stability on {used} images (each compared with its own unmodified photo)\n")
    print(f"{'perturbation':20s} {'median ΔE00':>11s} {'p90 ΔE00':>9s} {'median |ΔITA|':>14s} "
          f"{'depth changed':>14s} {'undertone changed':>18s}")
    for name, s in stats.items():
        if not s["n"]:
            continue
        print(f"{name:20s} {np.median(s['de']):11.1f} {np.percentile(s['de'], 90):9.1f} "
              f"{np.median(s['ita']):13.1f}° {s['depth_flip'] / s['n']:14.0%} {s['tone_flip'] / s['n']:18.0%}")
    print("\nΔE00 ≈ 1 is barely visible; 2–3 is a noticeable shade difference.")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="copy a balanced, anonymised sample for hand labelling")
    p.add_argument("--source", default=str(ROOT / "ai/data/raw"))
    p.add_argument("--out", default=str(ROOT / "ai/data/eval_images"))
    p.add_argument("--labels", default=str(ROOT / "eval/labels.csv"))
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("topup", help="add blind images until each undertone class has enough labels")
    p.add_argument("--source", default=str(ROOT / "ai/data/raw"))
    p.add_argument("--images", default=str(ROOT / "ai/data/eval_images"))
    p.add_argument("--labels", default=str(ROOT / "eval/labels.csv"))
    p.add_argument("--min-per-class", type=int, default=8)
    p.add_argument("--oversample", type=float, default=2.0,
                   help="targeted images per missing label (not every candidate is that class)")
    p.add_argument("--random-extra", type=int, default=None,
                   help="purely random images mixed in for blinding (default: as many as targeted)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--allow-haar", action="store_true",
                   help="run even if MediaPipe is unavailable (Haar fallback; results not comparable)")
    p.set_defaults(func=cmd_topup, needs_detector=True)

    p = sub.add_parser("run", help="score the pipeline against hand labels")
    p.add_argument("--labels", default=str(ROOT / "eval/labels.csv"))
    p.add_argument("--images", default=str(ROOT / "ai/data/eval_images"))
    p.add_argument("--allow-haar", action="store_true",
                   help="run even if MediaPipe is unavailable (Haar fallback; results not comparable)")
    p.set_defaults(func=cmd_run, needs_detector=True)

    p = sub.add_parser("stability", help="lighting perturbation check")
    p.add_argument("--images", default=str(ROOT / "ai/data/eval_images"))
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--allow-haar", action="store_true",
                   help="run even if MediaPipe is unavailable (Haar fallback; results not comparable)")
    p.set_defaults(func=cmd_stability, needs_detector=True)

    args = parser.parse_args(argv)
    if getattr(args, "needs_detector", False) and not require_mediapipe(args.allow_haar):
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
