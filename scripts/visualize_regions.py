#!/usr/bin/env python
"""
Draw the skin-sampling regions on a photo, so you can check they land where their names say.

Usage:
    python scripts/visualize_regions.py path/to/face.jpg [more.jpg ...] [--out region_previews]

Writes <out>/<image-name>_regions.png for each input. Skin regions (used for the overall
skin colour) are filled; observation-only regions (under-eye, perioral) are outlined.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.colorimetry.face_color import SKIN_REGIONS, detect_region_polygons  # noqa: E402

COLORS = {  # BGR
    "forehead": (0, 200, 255),
    "left_cheek": (0, 255, 0),
    "right_cheek": (255, 200, 0),
    "chin": (255, 0, 255),
    "left_under_eye": (0, 128, 255),
    "right_under_eye": (255, 128, 0),
    "left_perioral": (0, 0, 255),
    "right_perioral": (255, 0, 0),
}


def draw_regions(image: np.ndarray) -> tuple[np.ndarray, str]:
    polygons, source = detect_region_polygons(image)
    # Upscale small images so labels stay readable
    scale = max(1, 600 // max(image.shape[:2]))
    canvas = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    overlay = canvas.copy()
    for name, poly in polygons.items():
        poly = (poly * scale).astype(np.int32)
        color = COLORS.get(name, (255, 255, 255))
        if name in SKIN_REGIONS:
            cv2.fillConvexPoly(overlay, poly, color)
        cv2.polylines(canvas, [poly], True, color, 2)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    for name, poly in polygons.items():
        cx, cy = (poly * scale).mean(axis=0).astype(int)
        cv2.putText(canvas, name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(canvas, name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS.get(name), 1)
    cv2.putText(canvas, f"source: {source}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return canvas, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("region_previews"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    failures = 0
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"✗ {path}: could not read image")
            failures += 1
            continue
        try:
            canvas, source = draw_regions(image)
        except ValueError as e:
            print(f"✗ {path}: {e}")
            failures += 1
            continue
        out_path = args.out / f"{path.stem}_regions.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"✓ {path} → {out_path} ({source})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
