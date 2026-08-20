"""
test_full_pipeline.py — Test full ShadeSense pipeline
"""

import cv2
import glob
from ai.colorimetry.face_color import analyze_face_color
from ai.recommendation.match import match_foundations, load_foundation_db
from ai.recommendation.season import get_season_recommendations
from ai.recommendation.writer import get_full_recommendations


def full_pipeline(image_path):
    """Run full analysis pipeline."""
    print(f"📸 Analyzing: {image_path}")
    print("=" * 60)
    
    # 1. Face color extraction
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image")
        return None
    
    face_result = analyze_face_color(img)
    
    print("\n🎨 Skin Analysis:")
    print(f"  Depth: {face_result['depth']}")
    print(f"  Undertone: {face_result['undertone']}")
    print(f"  Clarity: {face_result['clarity']}")
    print(f"  LAB: L={face_result['L']:.1f}, a={face_result['a']:.1f}, b={face_result['b']:.1f}")
    print(f"  ITA: {face_result['ita_degrees']:.1f}°")
    print(f"  Confidence: {face_result['confidence']:.2%}")
    
    # 2. Foundation matching
    foundations = load_foundation_db()
    skin_lab = (face_result['L'], face_result['a'], face_result['b'])
    matches = match_foundations(skin_lab, foundations, top_k=3)
    
    print("\n💄 Top Foundation Matches:")
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m['brand']} - {m['shade']} (distance: {m['distance']:.2f})")
    
    # 3. Season classification
    season = get_season_recommendations(face_result)
    
    print(f"\n🍂 Season: {season['season_label']}")
    print(f"  Jewelry: {', '.join(season['jewelry'])}")
    print(f"  Lipstick: {season['lipstick_family']['name']}")
    print(f"  Blush: {season['blush_family']['name']}")
    
    # 4. Generate recommendations
    recommendation = get_full_recommendations(face_result, matches, season)
    
    print("\n📝 Recommendations:")
    print(recommendation['summary'])
    
    return {
        'face': face_result,
        'foundations': matches,
        'season': season,
        'recommendation': recommendation
    }


if __name__ == "__main__":
    # Test on specific images
    image_paths = glob.glob('/Users/shezamishal19/Desktop/ShadeSense/IMG_7881.jpeg')
    
    if not image_paths:
        print("❌ No images found at /Users/shezamishal19/Desktop/ShadeSense/IMG_7881.jpeg")
        # Try alternative path
        image_paths = glob.glob('/Users/shezamishal19/Desktop/ShadeSense/ai/data/raw/*/*.jpg')[:3]
    
    for img_path in image_paths:
        print("\n" + "=" * 60)
        result = full_pipeline(img_path)
        if result:
            print("\n✅ Analysis complete!")