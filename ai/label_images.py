#!/usr/bin/env python3
"""
Interactive Image Labeling Tool for Skin Undertones
"""

import os
import cv2
import shutil
from pathlib import Path

class SkinLabeler:
    def __init__(self):
        self.source_folder = "ai/data/kaggle_original"  # Your Kaggle images
        self.target_base = "ai/data/raw"
        
        # Create all category folders
        self.categories = ['light_cool', 'light_warm', 'light_neutral', 'light_olive',
                          'medium_cool', 'medium_warm', 'medium_neutral', 'medium_olive',
                          'deep_cool', 'deep_warm', 'deep_neutral', 'deep_olive']
        
        for cat in self.categories:
            os.makedirs(os.path.join(self.target_base, cat), exist_ok=True)
        
        self.current_index = 0
        self.image_files = []
        self.load_images()
    
    def load_images(self):
        """Load all images from source folder"""
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(Path(self.source_folder).glob(f'**/{ext}'))
        print(f"📸 Found {len(self.image_files)} images to label")
    
    def show_label_menu(self):
        """Display labeling options"""
        print("\n" + "="*50)
        print("🎨 SKIN UNDERTONE LABELER")
        print("="*50)
        print("\nSelect Undertone:")
        print("  1 = Cool (pink/red hues)")
        print("  2 = Warm (yellow/peach hues)")
        print("  3 = Neutral (balanced)")
        print("  4 = Olive (greenish hues)")
        print("\nSelect Skin Tone:")
        print("  l = Light")
        print("  m = Medium")
        print("  d = Deep")
        print("\n  s = Skip image")
        print("  q = Quit")
    
    def label_image(self, image_path):
        """Display image and get label"""
        img = cv2.imread(str(image_path))
        
        # Resize for display if too large
        height, width = img.shape[:2]
        max_display_size = 800
        if width > max_display_size:
            scale = max_display_size / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        cv2.imshow("Label This Image", img)
        cv2.moveWindow("Label This Image", 100, 100)
        
        while True:
            self.show_label_menu()
            choice = input("\n➡️  Enter choice: ").strip().lower()
            
            undertone_map = {'1': 'cool', '2': 'warm', '3': 'neutral', '4': 'olive'}
            tone_map = {'l': 'light', 'm': 'medium', 'd': 'deep'}
            
            if choice == 'q':
                return 'quit'
            elif choice == 's':
                return 'skip'
            elif choice in undertone_map and choice in tone_map:
                # This won't work - need separate inputs
                pass
            else:
                # Handle two-step input
                pass
        
        cv2.destroyAllWindows()
    
    def run(self):
        """Main labeling loop"""
        print("\n🚀 Starting labeling process...")
        print("Press 'q' to quit, 's' to skip\n")
        
        for idx, img_path in enumerate(self.image_files):
            self.current_index = idx + 1
            print(f"\n📷 Image {self.current_index}/{len(self.image_files)}: {img_path.name}")
            
            img = cv2.imread(str(img_path))
            if img is None:
                print("⚠️ Could not load image, skipping")
                continue
            
            # Display image
            height, width = img.shape[:2]
            display_img = img.copy()
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_img = cv2.resize(display_img, (new_width, new_height))
            
            cv2.imshow("Image to Label", display_img)
            cv2.moveWindow("Image to Label", 100, 100)
            
            # Get skin tone
            print("\n🎨 SKIN TONE:")
            print("  l = Light")
            print("  m = Medium")
            print("  d = Deep")
            
            tone_choice = input("Select skin tone (l/m/d): ").strip().lower()
            if tone_choice == 'q':
                break
            elif tone_choice not in ['l', 'm', 'd']:
                print("❌ Invalid choice, skipping...")
                cv2.destroyAllWindows()
                continue
            
            tone_map = {'l': 'light', 'm': 'medium', 'd': 'deep'}
            skin_tone = tone_map[tone_choice]
            
            # Get undertone
            print("\n🎨 UNDERTOME:")
            print("  1 = Cool (pink/red)")
            print("  2 = Warm (yellow/peach)")
            print("  3 = Neutral (balanced)")
            print("  4 = Olive (greenish)")
            
            under_choice = input("Select undertone (1/2/3/4): ").strip()
            if under_choice == 'q':
                break
            elif under_choice not in ['1', '2', '3', '4']:
                print("❌ Invalid choice, skipping...")
                cv2.destroyAllWindows()
                continue
            
            undertone_map = {'1': 'cool', '2': 'warm', '3': 'neutral', '4': 'olive'}
            undertone = undertone_map[under_choice]
            
            # Move file to correct folder
            category = f"{skin_tone}_{undertone}"
            dest_folder = os.path.join(self.target_base, category)
            dest_path = os.path.join(dest_folder, img_path.name)
            
            shutil.copy2(str(img_path), dest_path)
            print(f"✅ Saved to {category}/")
            
            cv2.destroyAllWindows()
        
        cv2.destroyAllWindows()
        print(f"\n🎉 Labeling complete! Check {self.target_base} for organized images")

if __name__ == "__main__":
    labeler = SkinLabeler()
    labeler.run()