# balance_dataset.py
import cv2
import os
import numpy as np
from pathlib import Path

def augment_low_categories():
    """Create augmented versions for categories with < 50 images"""
    
    target_counts = {
        'light_warm': 50,
        'light_neutral': 30,
        'light_olive': 30,
        'medium_warm': 50,
        'medium_olive': 30,
        'deep_warm': 50,
        'deep_neutral': 30,
    }
    
    data_path = Path('ai/data/raw')
    
    for category, target in target_counts.items():
        category_path = data_path / category
        if not category_path.exists():
            continue
        
        current_images = list(category_path.glob('*.jpg')) + list(category_path.glob('*.png'))
        current_count = len(current_images)
        
        if current_count >= target:
            print(f"✅ {category}: {current_count} images (target met)")
            continue
        
        needed = target - current_count
        print(f"⚠️ {category}: {current_count} images, need {needed} more")
        
        # Create augmented versions
        for i in range(needed):
            # Pick random image to augment
            import random
            img_path = random.choice(current_images)
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
            
            # Apply random augmentation
            h, w = img.shape[:2]
            
            # Random brightness
            if random.random() > 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Random flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            
            # Save augmented image
            new_name = f"aug_{i}_{img_path.name}"
            cv2.imwrite(str(category_path / new_name), img)
        
        print(f"✅ {category}: Now has {len(list(category_path.glob('*')))} images")

if __name__ == "__main__":
    augment_low_categories()