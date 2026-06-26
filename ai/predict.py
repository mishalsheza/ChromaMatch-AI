"""
Use trained AI model for predictions
"""
import tensorflow as tf
import cv2
import numpy as np
from utils.data_preprocessor import DataPreprocessor

class SkinPredictor:
    def __init__(self, model_path='ai/models/final_skin_analyzer.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = DataPreprocessor()
        self.skin_labels = ['Light', 'Medium', 'Deep']
        self.under_labels = ['Cool', 'Warm', 'Neutral', 'Olive']
        print("✅ AI Model loaded!")
    
    def predict(self, image_path=None, img=None):
        """Predict skin tone and undertone from image"""
        # Load image
        if img is None:
            img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not load image'}
        
        # Extract cheeks
        left, right, success = self.preprocessor.extract_cheek_ai(img)
        if not success or left is None:
            return {'error': 'Could not detect face/cheeks'}
        
        # Preprocess
        left = left / 255.0
        left = np.expand_dims(left, axis=0)
        
        # Predict
        predictions = self.model.predict(left, verbose=0)
        skin_idx = np.argmax(predictions[0][0])
        under_idx = np.argmax(predictions[1][0])
        
        return {
            'skin_tone': self.skin_labels[skin_idx],
            'undertone': self.under_labels[under_idx],
            'confidence': {
                'skin_tone': float(np.max(predictions[0][0])),
                'undertone': float(np.max(predictions[1][0]))
            }
        }
if __name__ == '__main__':
    predictor = SkinPredictor()
    # Test with an image
    # result = predictor.predict('path/to/test/image.jpg')
    # print(result)