from flask import Blueprint, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import os

ai_bp = Blueprint('ai_api', __name__)

# Load your trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai/models/balanced_skin_analyzer.h5')

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ AI Model loaded successfully!")
else:
    model = None
    print(f"⚠️ Model not found at {MODEL_PATH}")

@ai_bp.route('/analyze', methods=['POST'])
def analyze():
    """Simple AI skin analysis - no mediapipe needed"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Read image
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Resize and normalize
        img_resized = cv2.resize(img, (128, 128))
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        skin_pred, under_pred = model.predict(img_input, verbose=0)
        
        skin_labels = ['Light', 'Medium', 'Deep']
        under_labels = ['Cool', 'Warm', 'Neutral', 'Olive']
        
        skin_tone = skin_labels[np.argmax(skin_pred[0])]
        undertone = under_labels[np.argmax(under_pred[0])]
        
        # Get product recommendations
        from recommender import get_recommendations
        recommendations = get_recommendations(skin_tone, undertone)
        
        return jsonify({
            'success': True,
            'analysis': {
                'skin_tone': skin_tone,
                'undertone': undertone,
                'confidence': {
                    'skin_tone': float(np.max(skin_pred[0])),
                    'undertone': float(np.max(under_pred[0]))
                }
            },
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_bp.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})