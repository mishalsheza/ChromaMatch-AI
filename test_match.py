"""
test_match.py — Test foundation matching with face color extraction
"""

import cv2
from ai.colorimetry.face_color import analyze_face_color
from ai.recommendation.match import match_foundations, load_foundation_db

# Load database
foundations = load_foundation_db()
print(f'📂 Loaded {len(foundations)} foundation shades')

# Analyze face
img_path = '/Users/shezamishal19/Desktop/ShadeSense/ai/data/raw/light_cool/10_0_1_20170110225339066.jpg.chip.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"❌ Could not load image: {img_path}")
    exit()

result = analyze_face_color(img)

# Get skin LAB
skin_lab = (result['L'], result['a'], result['b'])
print(f'\n🎨 Skin LAB: L={skin_lab[0]:.1f}, a={skin_lab[1]:.1f}, b={skin_lab[2]:.1f}')
print(f'   Depth: {result["depth"]}, Undertone: {result["undertone"]}')

# Find matches
matches = match_foundations(skin_lab, foundations, top_k=5)

print('\n🏆 Top Foundation Matches:')
print('=' * 70)
for i, match in enumerate(matches, 1):
    print(f'{i}. {match["brand"]} - {match["product"]}')
    print(f'   Shade: {match["shade"]} ({match["hex"]})')
    print(f'   Distance: {match["distance"]:.2f}')
    print()