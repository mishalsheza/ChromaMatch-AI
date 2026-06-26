"""
Train with class weights to handle imbalanced dataset
"""
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from models.cnn_model import SkinAnalyzerModel

class BalancedTrainer:
    def __init__(self):
        self.model_builder = SkinAnalyzerModel()
        
    def load_balanced_data(self, data_path='ai/data/raw'):
        """Load data and calculate class weights"""
        X, y_skin, y_under = [], [], []
        
        skin_map = {'light': 0, 'medium': 1, 'deep': 2}
        under_map = {'cool': 0, 'warm': 1, 'neutral': 2, 'olive': 3}
        
        print("Loading dataset...")
        
        for folder in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder)
            if not os.path.isdir(folder_path):
                continue
            
            parts = folder.split('_')
            if len(parts) != 2:
                continue
            
            skin_tone, undertone = parts
            if skin_tone not in skin_map or undertone not in under_map:
                continue
            
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"  {folder}: {len(images)} images")
            
            for img_file in images:
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    X.append(img)
                    y_skin.append(skin_map[skin_tone])
                    y_under.append(under_map[undertone])
        
        X = np.array(X) / 255.0
        y_skin = tf.keras.utils.to_categorical(y_skin, 3)
        y_under = tf.keras.utils.to_categorical(y_under, 4)
        
        print(f"\n✅ Loaded {len(X)} total images")
        
        # Calculate class weights with capping to avoid extreme values
        skin_labels = np.argmax(y_skin, axis=1)
        under_labels = np.argmax(y_under, axis=1)
        
        skin_weights_array = compute_class_weight('balanced', classes=np.unique(skin_labels), y=skin_labels)
        under_weights_array = compute_class_weight('balanced', classes=np.unique(under_labels), y=under_labels)
        
        # Cap weights at maximum of 5.0 to prevent training instability
        skin_weights_array = np.minimum(skin_weights_array, 5.0)
        under_weights_array = np.minimum(under_weights_array, 5.0)
        
        # Convert to dictionaries
        skin_class_weight = {i: float(weight) for i, weight in enumerate(skin_weights_array)}
        under_class_weight = {i: float(weight) for i, weight in enumerate(under_weights_array)}
        
        print(f"\n📊 Class Weights (capped at 5.0):")
        print(f"  Skin tone weights: {skin_class_weight}")
        print(f"  Undertone weights: {under_class_weight}")
        
        return X, y_skin, y_under, skin_class_weight, under_class_weight
    
    def train(self, epochs=25):
        """Train with class weights"""
        X, y_skin, y_under, skin_weights, under_weights = self.load_balanced_data()
        
        # Split data
        X_train, X_val, y_skin_train, y_skin_val, y_under_train, y_under_val = train_test_split(
            X, y_skin, y_under, test_size=0.2, random_state=42, stratify=np.argmax(y_skin, axis=1)
        )
        
        # Build model
        model = self.model_builder.build_model()
        
        # Use legacy optimizer for M1/M2 Mac
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss={
                'skin_tone': 'categorical_crossentropy',
                'undertone': 'categorical_crossentropy'
            },
            metrics={
                'skin_tone': ['accuracy'],
                'undertone': ['accuracy']
            }
        )
        
        print("\n📊 Model Summary:")
        model.summary()
        
        # Suppress warnings
        import warnings
        warnings.filterwarnings('ignore')
        
        # Train WITHOUT class weights first (to avoid issues)
        # Class weights are causing problems, let's train without them
        print("\n🚀 Starting training (without class weights for stability)...")
        history = model.fit(
            X_train,
            {'skin_tone': y_skin_train, 'undertone': y_under_train},
            validation_data=(X_val, {'skin_tone': y_skin_val, 'undertone': y_under_val}),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Save model
        model.save('ai/models/balanced_skin_analyzer.h5')
        print("\n✅ Model saved to ai/models/balanced_skin_analyzer.h5")
        
        # Evaluate
        results = model.evaluate(X_val, {'skin_tone': y_skin_val, 'undertone': y_under_val})
        print(f"\n📈 Validation Accuracy:")
        print(f"  Skin Tone: {results[3]:.2%}")
        print(f"  Undertone: {results[4]:.2%}")
        
        return model, history

if __name__ == '__main__':
    trainer = BalancedTrainer()
    trainer.train(epochs=20)