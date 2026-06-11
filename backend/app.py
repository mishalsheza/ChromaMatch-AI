from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import os
import json

app = Flask(__name__)
CORS(app, origins=["http://localhost:5001", "http://localhost:5500", "http://localhost:5173", "*"])

# Load AI model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai/models/balanced_skin_analyzer.h5')
model = None
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ AI Model loaded!")
    except Exception as e:
        print(f"Model load error: {e}")

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    
    # Read image
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Check for manual sampling points
    manual_sampling = request.form.get('manual_sampling') == 'true'
    sampling_points = None
    
    if manual_sampling and 'sampling_points' in request.form:
        try:
            sampling_points = json.loads(request.form.get('sampling_points'))
            print(f"📍 Manual sampling points received: {len(sampling_points)}")
            for p in sampling_points:
                print(f"   {p['label']}: ({p['x']:.3f}, {p['y']:.3f})")
        except Exception as e:
            print(f"Error parsing sampling points: {e}")
    
    # Extract colors from the sampled points
    h, w = img.shape[:2]
    colors = []
    
    if sampling_points and len(sampling_points) >= 2:
        # Use the manually placed markers
        for point in sampling_points:
            # Convert normalized coordinates to pixel coordinates
            px = int(point['x'] * w)
            py = int(point['y'] * h)
            
            # Get a 10x10 patch around the point for accurate sampling
            patch_size = 15
            x_start = max(0, px - patch_size // 2)
            x_end = min(w, px + patch_size // 2)
            y_start = max(0, py - patch_size // 2)
            y_end = min(h, py + patch_size // 2)
            
            patch = img[y_start:y_end, x_start:x_end]
            if patch.size > 0:
                avg_color = np.mean(patch, axis=(0, 1))
                colors.append(avg_color)
                print(f"   {point['label']}: RGB{avg_color.astype(int)}")
        
        # Average both cheeks
        if len(colors) >= 2:
            avg_rgb = (colors[0] + colors[1]) / 2
        else:
            avg_rgb = colors[0]
    else:
        # Fallback: auto-detect face region
        print("No manual points, using auto-detection")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            left_cheek = img[y + fh//3:y + 2*fh//3, x + fw//4:x + fw//2]
            right_cheek = img[y + fh//3:y + 2*fh//3, x + fw//2:x + 3*fw//4]
            
            if left_cheek.size > 0 and right_cheek.size > 0:
                left_rgb = np.mean(left_cheek, axis=(0, 1))
                right_rgb = np.mean(right_cheek, axis=(0, 1))
                avg_rgb = (left_rgb + right_rgb) / 2
            else:
                avg_rgb = np.mean(img, axis=(0, 1))
        else:
            avg_rgb = np.mean(img, axis=(0, 1))
    
    # Convert RGB to LAB for analysis
    def rgb_to_lab(rgb):
        r, g, b = rgb[2], rgb[1], rgb[0]  # BGR to RGB
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
        
        # RGB to XYZ
        r = r ** 2.2 if r > 0.04045 else r / 12.92
        g = g ** 2.2 if g > 0.04045 else g / 12.92
        b = b ** 2.2 if b > 0.04045 else b / 12.92
        
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        x /= 0.95047
        y /= 1.0
        z /= 1.08883
        
        def f(t):
            return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
        
        L = 116 * f(y) - 16
        a = 500 * (f(x) - f(y))
        b_lab = 200 * (f(y) - f(z))
        
        return [L, a, b_lab]
    
    lab = rgb_to_lab(avg_rgb)
    L, a, b = lab
    
    # Calculate ITA
    import math
    ita = math.degrees(math.atan2(L - 50, b if b != 0 else 0.001))
    
    # Determine skin tone from brightness
    brightness = np.mean(avg_rgb)
    if brightness > 150:
        skin_tone = "Light"
    elif brightness > 100:
        skin_tone = "Medium"
    else:
        skin_tone = "Deep"
    
    # Determine undertone from a/b ratio
    if b > a + 5:
        undertone = "Warm"
    elif a > b + 5:
        undertone = "Cool"
    else:
        undertone = "Neutral"
    
    # Get recommendations
    from recommender import get_recommendations
    recommendations = get_recommendations(skin_tone, undertone)
    
    # Prepare response
    result = {
        'success': True,
        'analysis': {
            'skin_tone': skin_tone,
            'undertone': undertone,
            'ita': ita,
            'skin_type': skin_tone,
            'cosmetic_depth': skin_tone,
            'lab': {'L': L, 'a': a, 'b': b}
        },
        'recommendations': recommendations
    }
    
    print(f"✅ Result: {skin_tone} + {undertone} (ITA: {ita:.1f}°)")
    
    return jsonify(result)

@app.route('/')
def index():
    return jsonify({'message': 'ShadeSense AI Backend Running'})

if __name__ == '__main__':
    print("\n🚀 ShadeSense AI Backend Starting...")
    print(f"🤖 AI Model loaded: {model is not None}")
    print("📍 POST /api/analyze")
    print("📍 GET  /api/health")
    app.run(debug=True, host='0.0.0.0', port=5001)