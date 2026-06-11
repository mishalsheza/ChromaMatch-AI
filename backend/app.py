from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import os
from recommender import get_recommendations

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:5174"])

# Load model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai/models/balanced_skin_analyzer.h5')
model = None
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ AI Model loaded!")
    except Exception as e:
        print(f"Model load error: {e}")

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Check for manual sampling coordinates
    manual_sampling = request.form.get('manual_sampling', 'false').lower() == 'true'
    
    h, w = img.shape[:2]
    
    if manual_sampling:
        import json
        points = json.loads(request.form.get('sampling_points', '[]'))
        colors = []
        for pt in points:
            x, y = int(pt['x']), int(pt['y'])
            # Clamp to image dimensions
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            # Average a small patch for better sampling
            patch = img[max(0, y-5):min(h, y+5), max(0, x-5):min(w, x+5)]
            if patch.size > 0:
                colors.append(np.mean(patch, axis=(0, 1)))
            else:
                colors.append(img[y, x])
        
        if colors:
            avg_color = np.mean(colors, axis=0)
        else:
            face = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
            avg_color = np.mean(face, axis=(0, 1))
    else:
        # Auto mode: Get face region for color sampling
        face = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        avg_color = np.mean(face, axis=(0, 1))
        
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
    
    # AI prediction if available
    if model:
        img_resized = cv2.resize(img, (128, 128)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)
        skin_pred, under_pred = model.predict(img_input, verbose=0)
        
        skin_labels = ['Light', 'Medium', 'Deep']
        under_labels = ['Cool', 'Warm', 'Neutral', 'Olive']
        
        skin_tone = skin_labels[np.argmax(skin_pred[0])]
        undertone = under_labels[np.argmax(under_pred[0])]
    else:
        # Fallback color analysis
        brightness = np.mean(avg_color)
        skin_tone = "Light" if brightness > 150 else "Medium" if brightness > 100 else "Deep"
        r, g, b = avg_color
        undertone = "Warm" if r > g and r > b else "Cool" if b > r and b > g else "Neutral"
    
    # Return properly structured data matching what ResultsPage expects
    return jsonify({
        'success': True,
        'dimensions': {'width': w, 'height': h},
        'cheeks': {
            'left': {
                'center': [int(w*0.35), int(h*0.48)],
                'hex': hex_color
            },
            'right': {
                'center': [int(w*0.65), int(h*0.48)],
                'hex': hex_color
            }
        },
        'analysis': {
            'skin_tone': skin_tone,
            'undertone': undertone,
            'lab': {'L': float(np.mean(avg_color)), 'a': 0, 'b': 0},
            'ita': 30,
            'skin_type': skin_tone,
            'cosmetic_depth': skin_tone
        },
        'recommendations': get_recommendations(skin_tone, undertone)
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/')
def index():
    return jsonify({'message': 'ShadeSense AI Backend Running'})

if __name__ == '__main__':
    print("\n🚀 ShadeSense AI Backend Starting...")
    print(f"🤖 AI Model loaded: {model is not None}")
    print("📍 POST /api/analyze")
    print("📍 GET  /api/health")
    app.run(debug=True, host='0.0.0.0', port=5001)