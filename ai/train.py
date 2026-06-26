"""
Train the skin analysis AI model
"""
import tensorflow as tf
import numpy as np
import cv2
import os
from models.cnn_model import SkinAnalyzerModel
from utils.data_preprocessor import DataPreprocessor

class SkinAITrainer:
    def __init__(self):
        self.model_builder = SkinAnalyzerModel()
        self.preprocessor = DataPreprocessor()
    
    def load_data(self, data_path='ai/data/raw'):
        """Load and prepare training data"""
        X, y_skin, y_under = [], [], []
        
        skin_map = {'light': [1,0,0], 'medium': [0,1,0], 'deep': [0,0,1]}
        under_map = {'cool': [1,0,0,0], 'warm': [0,1,0,0], 
                     'neutral': [0,0,1,0], 'olive': [0,0,0,1]}
        
        print("Loading dataset...")
        
        for folder in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder)
            if not os.path.isdir(folder_path):
                continue
            
            # Parse folder name (e.g., "light_cool")
            parts = folder.split('_')
            if len(parts) != 2:
                continue
                
            skin_tone, undertone = parts
            if skin_tone not in skin_map or undertone not in under_map:
                continue
            
            print(f"Loading {folder}...")
            
            for img_file in os.listdir(folder_path):
                if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                    
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Extract cheeks using AI
                    left, right, success = self.preprocessor.extract_cheek_ai(img)
                    
                    if success and left is not None:
                        X.append(left)
                        y_skin.append(skin_map[skin_tone])
                        y_under.append(under_map[undertone])
                        
                        # Add augmented versions
                        for aug in self.preprocessor.augment_image(left):
                            X.append(aug)
                            y_skin.append(skin_map[skin_tone])
                            y_under.append(under_map[undertone])
        
        if len(X) == 0:
            print("\n❌ No training data found!")
            print("Please add images to folders like:")
            print("  ai/data/raw/light_cool/")
            print("  ai/data/raw/medium_warm/")
            print("  ai/data/raw/deep_neutral/")
            return None, None, None
        
        X = np.array(X) / 255.0
        y_skin = np.array(y_skin)
        y_under = np.array(y_under)
        
        print(f"\n✅ Loaded {len(X)} samples")
        return X, y_skin, y_under
    
    def train(self, epochs=30, batch_size=32):
        """Train the model"""
        # Load data
        X, y_skin, y_under = self.load_data()
        if X is None:
            return None
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_skin_train, y_skin_val, y_under_train, y_under_val = train_test_split(
            X, y_skin, y_under, test_size=0.2, random_state=42
        )
        
        # Build and compile model
        model = self.model_builder.build_model()
        model = self.model_builder.compile_model(model)
        
        print("\n📊 Model Summary:")
        model.summary()
        
        # Train
        print("\n🚀 Starting training...")
        history = model.fit(
            X_train,
            {'skin_tone': y_skin_train, 'undertone': y_under_train},
            validation_data=(
                X_val,
                {'skin_tone': y_skin_val, 'undertone': y_under_val}
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save model
        model.save('ai/models/final_skin_analyzer.h5')
        print("\n✅ Model saved to ai/models/final_skin_analyzer.h5")
        
        # Evaluate
        results = model.evaluate(X_val, {'skin_tone': y_skin_val, 'undertone': y_under_val})
        print(f"\n📈 Validation Accuracy:")
        print(f"  Skin Tone: {results[3]:.2%}")
        print(f"  Undertone: {results[4]:.2%}")
        
        return model, history

if __name__ == '__main__':
    trainer = SkinAITrainer()
    trainer.train()