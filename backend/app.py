from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import os
import json
import uuid
import math

# ──────────────────────────────────────────────────────────────
# CREATE FLASK APP FIRST
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["http://localhost:5001", "http://localhost:5500", "http://localhost:5173", "*"])

# ──────────────────────────────────────────────────────────────
# IMPORT FACE DETECTOR
# ──────────────────────────────────────────────────────────────
from face_detector_v2 import FaceDetector
detector = FaceDetector()

# ──────────────────────────────────────────────────────────────
# IMPORT TRY-ON MODULE
# ──────────────────────────────────────────────────────────────
from foundation_tryon import FoundationTryOn
tryon = FoundationTryOn()

# ──────────────────────────────────────────────────────────────
# LOAD AI MODEL
# ──────────────────────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai/models/balanced_skin_analyzer.h5')
model = None
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ AI Model loaded!")
    except Exception as e:
        print(f"Model load error: {e}")

# ──────────────────────────────────────────────────────────────
# MINICPM-V RECOMMENDATIONS FUNCTION
# ──────────────────────────────────────────────────────────────
def get_minicpm_recommendations(skin_tone, undertone):
    """Get structured color recommendations using MiniCPM-V"""
    from openai import OpenAI
    import json
    
    try:
        client = OpenAI(
            api_key="sk-pQ8L2zF3XmR5kY9wV4jB7hN1tC6vM0xG3aD5sH2bJ9lK4cZ8",
            base_url="https://api.modelbest.cn/v1"
        )
        
        prompt = f"""
You are a professional color analyst for Indian skin tones.

Based ONLY on this skin profile:
- Skin Tone: {skin_tone}
- Undertone: {undertone}

Return EXACTLY this JSON structure with REAL, specific, and WEARABLE recommendations:
{{
  "best_colors": ["5 specific color names that suit this skin tone"],
  "worst_colors": ["3 specific color names to avoid"],
  "jewelry": "One specific metal",
  "hair_colors": ["3 specific hair colors"],
  "makeup": {{
    "blush": "One specific blush shade",
    "lipstick": "One specific lipstick shade"
  }},
  "style_archetype": "A 2-3 word style name",
  "celebrity_references": ["2 Indian celebrities with similar coloring"],
  "description": "1 sentence explaining the best colors"
}}

Use REAL, WEARABLE cosmetic shade names.
"""
        
        response = client.chat.completions.create(
            model="MiniCPM-V-4.6-Instruct",
            messages=[
                {"role": "system", "content": "You are a color analysis expert. Always return valid JSON with specific, real color names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content
        result_text = result_text.strip()
        if result_text.startswith("```"):
            lines = result_text.split('\n')
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            result_text = '\n'.join(lines).strip()
            
        minicpm_data = json.loads(result_text)
        
        return {
            "style_archetype": minicpm_data.get("style_archetype", "Classic Elegance"),
            "seasonal_palette": {"name": minicpm_data.get("style_archetype", f"{undertone} Season")},
            "clothing_palette": {
                "best_colors": minicpm_data.get("best_colors", ["Navy", "Burgundy", "Olive"]),
                "avoid_colors": minicpm_data.get("worst_colors", ["Neon Yellow", "Bright Orange"])
            },
            "jewelry": {"recommended": minicpm_data.get("jewelry", "Gold")},
            "makeup": {
                "blush": minicpm_data.get("makeup", {}).get("blush", "Warm Peach"),
                "lipstick": minicpm_data.get("makeup", {}).get("lipstick", "Terracotta")
            },
            "hair_colors": minicpm_data.get("hair_colors", ["Dark Brown", "Caramel"]),
            "celebrity_references": minicpm_data.get("celebrity_references", []),
            "description": minicpm_data.get("description", "")
        }
        
    except Exception as e:
        print(f"⚠️ MiniCPM API Error: {e}")
        return get_color_recommendations(skin_tone, undertone)


# ──────────────────────────────────────────────────────────────
# COLOR RECOMMENDATIONS FUNCTION (FALLBACK)
# ──────────────────────────────────────────────────────────────
def get_color_recommendations(skin_tone, undertone):
    """Fallback rule-based recommendations"""
    tone_recommendations = {
        "Light": {
            "clothing": ["Pastel Pink", "Lavender", "Mint Green", "Baby Blue"],
            "makeup": {"blush": "Peach Blush", "lipstick": "Rose"},
            "jewelry": "Silver, Rose Gold",
            "hair_colors": ["Ash Blonde", "Platinum", "Honey Blonde"],
            "description": "Light skin looks stunning in soft, pastel shades."
        },
        "Medium": {
            "clothing": ["Coral", "Teal", "Olive Green", "Warm Red"],
            "makeup": {"blush": "Warm Peach Blush", "lipstick": "Terracotta"},
            "jewelry": "Gold, Rose Gold",
            "hair_colors": ["Caramel", "Chestnut Brown", "Warm Blonde"],
            "description": "Medium skin radiates in warm, earthy tones."
        },
        "Deep": {
            "clothing": ["Burgundy", "Emerald Green", "Royal Blue", "Deep Purple"],
            "makeup": {"blush": "Deep Berry Blush", "lipstick": "Plum"},
            "jewelry": "Gold, Brass",
            "hair_colors": ["Dark Brown", "Jet Black", "Warm Caramel"],
            "description": "Deep skin glows in rich, jewel-toned colors."
        }
    }
    
    undertone_map = {
        "Warm": {
            "best_colors": ["Orange", "Peach", "Coral", "Gold", "Olive Green"],
            "avoid_colors": ["Cool Blue", "Silver", "Hot Pink"],
            "jewelry": "Gold",
            "description": "Warm undertones shine in golden, peachy shades."
        },
        "Cool": {
            "best_colors": ["Blue", "Purple", "Pink", "Emerald Green", "Silver"],
            "avoid_colors": ["Orange", "Yellow", "Peach"],
            "jewelry": "Silver",
            "description": "Cool undertones pop in jewel tones and blue-based colors."
        },
        "Neutral": {
            "best_colors": ["Dusty Pink", "Sage Green", "Mauve", "Taupe"],
            "avoid_colors": ["NEON colors"],
            "jewelry": "Both Gold and Silver",
            "description": "Neutral undertones can wear both warm and cool colors."
        },
        "Olive": {
            "best_colors": ["Teal", "Burgundy", "Navy", "Forest Green"],
            "avoid_colors": ["Pastels", "Orange", "Yellow"],
            "jewelry": "Gold, Bronze",
            "description": "Olive undertones glow in earthy, jewel-toned colors."
        }
    }
    
    base = tone_recommendations.get(skin_tone, tone_recommendations["Medium"])
    under = undertone_map.get(undertone, undertone_map["Neutral"])
    
    return {
        "style_archetype": f"{undertone} Elegance",
        "seasonal_palette": {"name": f"{undertone} Season"},
        "clothing_palette": {
            "best_colors": under.get("best_colors", []),
            "avoid_colors": under.get("avoid_colors", [])
        },
        "jewelry": {"recommended": under.get("jewelry", "Gold")},
        "makeup": base.get("makeup", {"blush": "Rose", "lipstick": "Nude"}),
        "hair_colors": base.get("hair_colors", []),
        "celebrity_references": [],
        "description": base.get("description", "") + " " + under.get("description", "")
    }


# ──────────────────────────────────────────────────────────────
# HELPER: GET PATCH
# ──────────────────────────────────────────────────────────────
def get_patch(img, px, py, patch_size=15):
    """Extract a patch around a point"""
    h, w = img.shape[:2]
    x_start = max(0, px - patch_size // 2)
    x_end = min(w, px + patch_size // 2)
    y_start = max(0, py - patch_size // 2)
    y_end = min(h, py + patch_size // 2)
    patch = img[y_start:y_end, x_start:x_end]
    if patch.size > 0:
        return patch
    return None


# ──────────────────────────────────────────────────────────────
# COMBINED SKIN COLOR FUNCTION
# ──────────────────────────────────────────────────────────────
def get_combined_skin_color(img_bgr, sampling_points=None, return_scan=False):
    """Hybrid: Manual pointers + automatic sampling + full face scan"""
    h, w = img_bgr.shape[:2]
    all_colors = []
    weights = []
    
    # 1. Manual pointers (user placed)
    if sampling_points and len(sampling_points) >= 2:
        for point in sampling_points:
            px = int(point['x'] * w)
            py = int(point['y'] * h)
            patch = get_patch(img_bgr, px, py)
            if patch is not None:
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                all_colors.append(np.mean(patch_rgb, axis=(0, 1)))
                weights.append(0.25)
    
    # 2. Full face scan (most accurate)
    full_face_scan, scan_success, scan_error = detector.scan_full_face(img_bgr)
    if scan_success and full_face_scan:
        median_rgb = full_face_scan["median_skin_color"]["rgb"]
        all_colors.append(np.array(median_rgb, dtype=float))
        weights.append(0.45)
        print(f"📊 Full Face Scan: Quality {full_face_scan['quality_score']}/100")
    else:
        print(f"⚠️ Full Face Scan failed: {scan_error}")
        full_face_scan = None
    
    # 3. Region sampling (fallback)
    region_colors, success, error, face_box = detector.detect_face_regions(img_bgr)
    if success and region_colors:
        for region in ["left_cheek", "right_cheek", "jawline_left", "chin"]:
            if region in region_colors:
                lab_arr = np.uint8([[region_colors[region]]])
                rgb_arr = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2RGB)
                all_colors.append(rgb_arr[0][0].astype(float))
                weights.append(0.10 if scan_success else 0.20)
    
    if not all_colors:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        fallback_rgb = np.mean(img_rgb, axis=(0, 1))
        if return_scan:
            return fallback_rgb, None
        return fallback_rgb
    
    weights = np.array(weights) / sum(weights)
    
    final_rgb = np.zeros(3)
    for i, color in enumerate(all_colors):
        final_rgb += color * weights[i]
    
    if return_scan:
        return final_rgb, full_face_scan
    return final_rgb


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return jsonify({'message': 'ShadeSense AI Backend Running'})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/api/tryon', methods=['POST'])
def try_foundation():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    shade_name = request.form.get('shade', '')
    intensity = float(request.form.get('intensity', 0.7))
    
    if not shade_name:
        return jsonify({'error': 'No shade selected'}), 400
    
    file = request.files['image']
    temp_path = f"temp_tryon_{uuid.uuid4().hex}.jpg"
    file.save(temp_path)
    
    try:
        result_base64 = tryon.apply_foundation(temp_path, shade_name, intensity)
        return jsonify({'success': True, 'image': result_base64, 'shade': shade_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Check for manual sampling points
    manual_sampling = request.form.get('manual_sampling') == 'true'
    sampling_points = None
    if manual_sampling and 'sampling_points' in request.form:
        try:
            sampling_points = json.loads(request.form.get('sampling_points'))
        except:
            pass
    
    # ── HYBRID COLOR EXTRACTION ──
    avg_rgb, full_face_scan = get_combined_skin_color(img_bgr, sampling_points, return_scan=True)
    brightness = np.mean(avg_rgb)
    
    # ── CONVERT RGB TO LAB ──
    def rgb_to_lab(rgb_values):
        r, g, b = rgb_values[0], rgb_values[1], rgb_values[2]
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
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
    
    lab_avg = rgb_to_lab(avg_rgb)
    L_val = lab_avg[0]
    a_val = lab_avg[1]
    b_val = lab_avg[2]
    
    # ── SKIN TONE DETECTION ──
    if L_val > 75:
        skin_tone = "Very Fair"
    elif L_val > 65:
        skin_tone = "Fair"
    elif L_val > 55:
        skin_tone = "Light"
    elif L_val > 45:
        skin_tone = "Medium"
    elif L_val > 35:
        skin_tone = "Tan"
    elif L_val > 25:
        skin_tone = "Deep"
    else:
        skin_tone = "Rich Deep"
    
    print(f"📊 Skin Tone: {skin_tone} (L*={L_val:.1f})")
    print(f"📊 RGB: R={avg_rgb[0]:.1f}, G={avg_rgb[1]:.1f}, B={avg_rgb[2]:.1f}")
    
    # ── UNDERTONE DETECTION ──
    angle = math.degrees(math.atan2(b_val, a_val)) if a_val != 0 else 0
    ratio = b_val / (a_val + 0.01) if a_val != 0 else 0
    
    print(f"📊 Undertone: a*={a_val:.1f}, b*={b_val:.1f}, angle={angle:.1f}°")
    
    if angle > 45 and ratio > 1.5:
        undertone = "Warm Golden"
    elif angle > 30 and ratio > 1.2:
        undertone = "Warm"
    elif angle > 15 and ratio > 0.8:
        undertone = "Warm Peach"
    elif angle > -15 and ratio > -0.5:
        undertone = "Neutral"
    elif angle > -30 and ratio < -0.5:
        undertone = "Cool Pink"
    elif angle > -45 and ratio < -0.8:
        undertone = "Cool Rose"
    elif a_val > 0 and b_val < 10 and a_val > b_val:
        undertone = "Olive"
    else:
        undertone = "Neutral"
    
    print(f"🎨 Detected Undertone: {undertone}")
    
    # ── MAP TO SIMPLIFIED CATEGORIES ──
    if "Warm" in undertone:
        simple_undertone = "Warm"
    elif "Cool" in undertone:
        simple_undertone = "Cool"
    elif "Olive" in undertone:
        simple_undertone = "Olive"
    else:
        simple_undertone = "Neutral"
    
    if skin_tone in ["Very Fair", "Fair", "Light"]:
        simple_skin = "Light"
    elif skin_tone in ["Medium", "Tan"]:
        simple_skin = "Medium"
    else:
        simple_skin = "Deep"
    
    print(f"✅ Simple Mapping: {simple_skin} + {simple_undertone}")
    
    # ── GET RECOMMENDATIONS ──
    from recommender import get_recommendations
    foundations = get_recommendations(simple_skin, simple_undertone)
    color_recs = get_minicpm_recommendations(simple_skin, simple_undertone)
    
    # ── RESPONSE ──
    result = {
        'success': True,
        'analysis': {
            'skin_tone': skin_tone,
            'skin_tone_simple': simple_skin,
            'undertone': undertone,
            'undertone_simple': simple_undertone,
            'ita': 30,
            'lab': {'L': float(L_val), 'a': float(a_val), 'b': float(b_val)},
            'full_face_scan': full_face_scan
        },
        'recommendations': {
            'foundations': foundations.get('foundations', []) if isinstance(foundations, dict) else foundations
        },
        'color_recommendations': color_recs,
        'source': 'minicpm'
    }
    
    print(f"✅ Result: {simple_skin} + {simple_undertone}")
    return jsonify(result)


# ──────────────────────────────────────────────────────────────
# ERROR HELPER
# ──────────────────────────────────────────────────────────────
def get_user_friendly_error(error_msg):
    error_lower = error_msg.lower()
    if 'dark' in error_lower:
        return {'title': '🌙 Too Dark', 'message': 'Image too dark. Use better lighting.', 'tip': 'Natural daylight works best!'}
    elif 'overexposed' in error_lower or 'bright' in error_lower:
        return {'title': '☀️ Too Bright', 'message': 'Image overexposed.', 'tip': 'Avoid direct sunlight.'}
    elif 'small' in error_lower:
        return {'title': '📷 Face Too Small', 'message': 'Face too small. Move closer.', 'tip': 'Fill at least 1/4 of frame.'}
    elif 'lighting' in error_lower or 'uneven' in error_lower:
        return {'title': '💡 Uneven Lighting', 'message': 'Lighting is uneven.', 'tip': 'Face the light source directly.'}
    else:
        return {'title': '⚠️ Quality Issue', 'message': error_msg, 'tip': 'Retake with better lighting.'}


# ──────────────────────────────────────────────────────────────
# RUN THE APP
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 ShadeSense AI Backend Starting...")
    print(f"🤖 AI Model loaded: {model is not None}")
    print("📍 POST /api/analyze")
    print("📍 POST /api/tryon")
    print("📍 GET  /api/health")
    app.run(debug=False, host='0.0.0.0', port=5001)