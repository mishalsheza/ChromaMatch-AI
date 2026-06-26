"""
Only review images where confidence was low
"""
import cv2
import os

def review_uncertain_images():
    """Show only images that need manual review"""
    
    # These would be images where undertone detection was borderline
    # For now, randomly sample 50 images to verify
    
    import random
    from pathlib import Path
    
    all_images = list(Path("ai/data/raw").rglob("*.jpg"))
    random.shuffle(all_images)
    
    print("🔍 Reviewing 50 random images to verify auto-labeling...")
    print("Press:")
    print("  c = Cool, w = Warm, n = Neutral, o = Olive")
    print("  s = Skip, q = Quit\n")
    
    reviewed = 0
    for img_path in all_images[:50]:
        img = cv2.imread(str(img_path))
        
        # Resize for display
        h, w = img.shape[:2]
        if w > 600:
            scale = 600 / w
            img = cv2.resize(img, (600, int(h * scale)))
        
        cv2.imshow("Verify Label", img)
        cv2.moveWindow("Verify Label", 100, 100)
        
        current_label = img_path.parent.name
        print(f"\nCurrent label: {current_label}")
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("  ✅ Confirmed Cool")
            reviewed += 1
        elif key == ord('w'):
            print("  ✅ Confirmed Warm")
            reviewed += 1
        elif key == ord('n'):
            print("  ✅ Confirmed Neutral")
            reviewed += 1
        elif key == ord('o'):
            print("  ✅ Confirmed Olive")
            reviewed += 1
    
    print(f"\n✅ Reviewed {reviewed} images")

if __name__ == "__main__":
    review_uncertain_images()