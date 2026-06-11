"""
AUTO-LABEL ALL  IMAGES IN SECONDS!
Uses your existing undertone_analyzer.py
"""

import cv2
import os
import shutil
import numpy as np
from pathlib import Path

# Import your existing color analyzer
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.undertone_analyzer import analyze_skin_color

def auto_label_all_images():
    """Automatically label all images using color analysis"""
    
    # Where your Kaggle images are
    source_folder = "ai/data/kaggle_original"
    
    # Where to put labeled images
    target_base = "ai/data/raw"
    
    # Create all category folders
    categories = ['light_cool', 'light_warm', 'light_neutral', 'light_olive',
                  'medium_cool', 'medium_warm', 'medium_neutral', 'medium_olive',
                  'deep_cool', 'deep_warm', 'deep_neutral', 'deep_olive']
    
    for cat in categories:
        os.makedirs(os.path.join(target_base, cat), exist_ok=True)
    
    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(source_folder).glob(f'**/{ext}'))
    
    print(f"📸 Found {len(image_files)} images")
    print("🎨 Auto-labeling in progress...\n")
    
    # Statistics
    stats = {cat: 0 for cat in categories}
    
    for idx, img_path in enumerate(image_files):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Estimate skin tone from brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Determine skin tone based on brightness
        if avg_brightness > 150:
            skin_tone = "light"
        elif avg_brightness > 100:
            skin_tone = "medium"
        else:
            skin_tone = "deep"
        
        # Estimate cheek regions (approximate)
        h, w = img.shape[:2]
        left_cheek = img[int(h*0.4):int(h*0.7), int(w*0.1):int(w*0.4)]
        right_cheek = img[int(h*0.4):int(h*0.7), int(w*0.6):int(w*0.9)]
        
        if left_cheek.size == 0 or right_cheek.size == 0:
            # Fallback: use whole face region
            face_region = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
            if face_region.size > 0:
                avg_color = np.mean(face_region, axis=(0, 1))
                left_rgb = right_rgb = avg_color
            else:
                continue
        else:
            left_rgb = np.mean(left_cheek, axis=(0, 1))
            right_rgb = np.mean(right_cheek, axis=(0, 1))
        
        # Use your existing analyzer!
        try:
            analysis = analyze_skin_color(left_rgb, right_rgb)
            undertone = analysis['undertone'].lower()
        except:
            # Fallback: simple color-based detection
            r, g, b = left_rgb
            if r > g and r > b:
                undertone = "warm"
            elif b > r and b > g:
                undertone = "cool"
            else:
                undertone = "neutral"
        
        # Special check for olive (greenish tint)
        r, g, b = left_rgb
        if g > r and g > b and (g - r) < 20:
            undertone = "olive"
        
        # Create category
        category = f"{skin_tone}_{undertone}"
        
        # Copy file
        dest_path = os.path.join(target_base, category, img_path.name)
        shutil.copy2(str(img_path), dest_path)
        
        stats[category] += 1
        
        # Progress update
        if (idx + 1) % 100 == 0:
            print(f"✅ Processed {idx + 1}/{len(image_files)} images")
    
    # Print summary
    print("\n" + "="*50)
    print("📊 AUTO-LABELING COMPLETE!")
    print("="*50)
    for cat in categories:
        if stats[cat] > 0:
            print(f"{cat:20} : {stats[cat]:3} images")
    
    print(f"\n✅ Total: {sum(stats.values())} images labeled")
    print(f"📁 Location: {target_base}")

if __name__ == "__main__":
    auto_label_all_images()