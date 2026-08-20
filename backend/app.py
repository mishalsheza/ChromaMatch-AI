"""
app.py — ShadeSense AI v2 (Colorimetry-based)
- Face color extraction using LAB color space
- Foundation matching with Delta-E 2000
- 12-season color analysis
- Groq AI recommendations
- Skin observations (under-eye, perioral, redness)
- Try-On feature
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import sys
import json
import uuid

# ──────────────────────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ──────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────
from ai.colorimetry.skin_observations import detect_skin_observations

# ──────────────────────────────────────────────────────────────
# CREATE FLASK APP
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5001",
    "http://localhost:5500",
    "http://localhost:5173",
    "*"
])

print("\n" + "=" * 60)
print("🚀 ShadeSense AI v2 Starting...")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# LOAD MODULES
# ──────────────────────────────────────────────────────────────

# Face Color Extraction
try:
    from ai.colorimetry.face_color import analyze_face_color
    print("✅ Face color extraction loaded")
except Exception as e:
    print(f"❌ Face color extraction error: {e}")

# Foundation Database
try:
    from ai.recommendation.match import load_foundation_db, match_foundations
    foundations = load_foundation_db() or []
    print(f"✅ Foundation database loaded: {len(foundations)} shades")
except Exception as e:
    print(f"❌ Foundation DB error: {e}")
    foundations = []

# Season Classification
try:
    from ai.recommendation.season import (
        get_season_recommendations,
        get_ranked_season_recommendations
    )
    print("✅ Season classification loaded")
except Exception as e:
    print(f"❌ Season classification error: {e}")

# Groq AI Recommendations
try:
    from ai.recommendation.groq_writer import (
        get_ai_recommendations,
        get_ai_recommendations_ranked
    )
    import ai.recommendation.groq_writer as gw
    print(f"🗂️ groq_writer.py loaded from: {gw.__file__}")
    print("✅ Groq AI recommendations loaded")
except Exception as e:
    print(f"⚠️ Groq AI recommendations error: {e}")
    get_ai_recommendations = None
    get_ai_recommendations_ranked = None

# Try-On Module
try:
    from foundation_tryon import FoundationTryOn
    tryon = FoundationTryOn()
    print("✅ Try-On module loaded")
except Exception as e:
    print(f"⚠️ Try-On module error: {e}")
    tryon = None

# Legacy CNN (for reference only)
model_path = os.path.join(ROOT_DIR, 'ai/models/balanced_skin_analyzer.h5')
if os.path.exists(model_path):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print("✅ CNN model loaded (for reference only)")
    except Exception as e:
        print(f"⚠️ CNN model not loaded: {e}")

print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return jsonify({
        'message': 'ShadeSense AI v2',
        'version': '2.0.0',
        'method': 'colorimetry',
        'endpoints': {
            'POST /api/analyze': 'Main analysis endpoint',
            'POST /api/tryon': 'Try-On foundation preview',
            'GET /api/health': 'Health check'
        }
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'method': 'colorimetry',
        'foundations_loaded': len(foundations)
    })


@app.route('/api/tryon', methods=['POST'])
def try_foundation():
    """Apply foundation try-on to uploaded image."""
    if tryon is None:
        return jsonify({'error': 'Try-On module not available'}), 503

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
        return jsonify({
            'success': True,
            'image': result_base64,
            'shade': shade_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint - uses colorimetry pipeline + Groq AI.
    """
    # ── Validate Input ──
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    # ── Step 1: Face Color Extraction ──
    try:
        face_result = analyze_face_color(img)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Face analysis failed: {str(e)}'}), 500

    # ── Step 2: Skin Observations ──
    skin_observations = {}
    try:
        per_region = face_result.get('per_region_values', {})
        if per_region:
            # Handle naming mismatch
            if 'jaw_chin' in per_region and 'jaw' not in per_region:
                per_region['jaw'] = per_region['jaw_chin']

            skin_observations = detect_skin_observations(per_region)

            # FORCE TEST: Add mock observation if none detected
            if skin_observations.get('has_observations', False):
                print("🔍 SKIN OBSERVATIONS FOUND:")
                for key in ['under_eye', 'perioral', 'redness']:
                    if skin_observations.get(key):
                        print(f"   - {key}: {skin_observations[key]}")
            else:
                print("🔍 No notable skin observations detected")
            # Debug output
            if skin_observations.get('has_observations', False):
                print("🔍 SKIN OBSERVATIONS FOUND:")
                for key in ['under_eye', 'perioral', 'redness']:
                    if skin_observations.get(key):
                        print(f"   - {key}: {skin_observations[key]}")
            else:
                print("🔍 No notable skin observations detected")

    except Exception as e:
        print(f"⚠️ Skin observation error: {e}")
        import traceback
        traceback.print_exc()

    # ── Step 3: Foundation Matching ──
    matches = []
    if foundations:
        skin_lab = (face_result['L'], face_result['a'], face_result['b'])
        matches = match_foundations(skin_lab, foundations, top_k=5)

    # ── Step 4: Season Classification ──
    try:
        season = get_season_recommendations(face_result)
        print(f"📊 Season classification: {season.get('season_label', 'Unknown')}")
    except Exception as e:
        print(f"⚠️ Season classification error: {e}")
        season = {
            'season_label': 'Neutral',
            'family': 'neutral',
            'jewelry': ['gold', 'silver'],
            'lipstick_family': {'name': 'neutral'},
            'blush_family': {'name': 'neutral'}
        }

    # ── Step 5: Rank Season Candidates ──
    try:
        ranked_seasons = get_ranked_season_recommendations(face_result, top_n=3)
        print("📊 Ranked season candidates:")
        for candidate in ranked_seasons.get("top_seasons", []):
            print(f"   {candidate['score_pct']:.1f}% {candidate['season_label']}")
        print(f"📊 Season margin: {ranked_seasons.get('margin_pct', 0):.1f}% "
              f"| close call: {ranked_seasons.get('is_close_call', False)}")
    except Exception as e:
        print(f"⚠️ Ranked season analysis error: {e}")
        ranked_seasons = {
            "top_seasons": [],
            "margin": 0,
            "margin_pct": 0,
            "is_close_call": False,
            "winner_key": season.get("season_key")
        }

    # ── Step 6: Generate AI Recommendations ──
    ai_used = False
    recommendations_text = "Analysis complete. See your skin profile and foundation matches below."
    structured_recommendations = {}

    season_details = {
        'jewelry': season.get('jewelry', ['gold', 'silver']),
        'lipstick': season.get('lipstick_family', {}).get('name', 'neutral'),
        'blush': season.get('blush_family', {}).get('name', 'neutral'),
        'best_colors': {},
        'worst_colors': []
    }

    if get_ai_recommendations_ranked:
        try:
            season_reference_path = os.path.join(ROOT_DIR, "ai", "data", "season_color_reference.json")
            with open(season_reference_path, "r") as f:
                seasons_full_data = json.load(f).get("seasons", {})

            ai_result = get_ai_recommendations_ranked(
                face_result,
                matches,
                ranked_seasons,
                seasons_full_data,
                skin_observations
            )

            recommendations_text = ai_result.get("summary", "Analysis complete.")
            structured_recommendations = ai_result.get("structured", {})
            structured_recommendations["ranked_seasons"] = ranked_seasons

            if "season" in structured_recommendations:
                season_details = structured_recommendations["season"]

            ai_used = True
            print("✅ Ranked AI recommendations generated successfully")

        except Exception as e:
            print(f"⚠️ AI recommendation error: {e}")
            import traceback
            traceback.print_exc()

            recommendations_text = (
                f"🌟 Based on your skin analysis, you have "
                f"{face_result.get('depth', 'medium')} skin with "
                f"{face_result.get('undertone', 'neutral')} undertones.\n\n"
                f"💄 Your best foundation match is "
                f"{matches[0]['brand'] + ' - ' + matches[0]['shade'] if matches else 'None found'}.\n\n"
                f"🎨 Your color season is {season.get('season_label', 'Neutral')}."
            )
            ai_used = False

    # ── Step 7: Build Response ──
    response = {
        'success': True,
        'method': 'colorimetry_v2',
        'ai_used': ai_used,
        'ai_source': 'groq' if ai_used else 'fallback',
        'analysis': {
            'depth': face_result.get('depth', 'medium'),
            'undertone': face_result.get('undertone', 'neutral'),
            'clarity': face_result.get('clarity', 'medium'),
            'lab': {
                'L': face_result.get('L', 50),
                'a': face_result.get('a', 0),
                'b': face_result.get('b', 0)
            },
            'ita_degrees': face_result.get('ita_degrees', 0),
            'ita_bucket': face_result.get('ita_bucket', 'unknown'),
            'confidence': face_result.get('confidence', 0.5),
            'per_region_values': face_result.get('per_region_values', {})
        },
        'foundations': matches,
        'season': {
            'label': season.get('season_label', 'Neutral'),
            'family': season.get('family', 'neutral'),
            'jewelry': season_details.get('jewelry', season.get('jewelry', ['gold', 'silver'])),
            'lipstick': season_details.get('lipstick', season.get('lipstick_family', {}).get('name', 'neutral')),
            'blush': season_details.get('blush', season.get('blush_family', {}).get('name', 'neutral')),
            'best_colors': season_details.get('best_colors', {}),
            'worst_colors': season_details.get('worst_colors', [])
        },
        'recommendations': recommendations_text,
        'ai_recommendations': {
            'summary': recommendations_text
        },
        'structured_recommendations': structured_recommendations
    }

    return jsonify(response)


@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    """Analyze multiple images for testing."""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    results = []

    for file in files:
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            results.append({'filename': file.filename, 'error': 'Invalid image'})
            continue

        try:
            face_result = analyze_face_color(img)
            skin_lab = (face_result['L'], face_result['a'], face_result['b'])
            matches = match_foundations(skin_lab, foundations, top_k=3) if foundations else []
            season = get_season_recommendations(face_result)

            results.append({
                'filename': file.filename,
                'depth': face_result['depth'],
                'undertone': face_result['undertone'],
                'season': season.get('season_label', 'Unknown'),
                'top_match': matches[0] if matches else None
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify({'success': True, 'results': results})


# ──────────────────────────────────────────────────────────────
# ERROR HANDLING
# ──────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ──────────────────────────────────────────────────────────────
# RUN THE APP
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("📍 Endpoints:")
    print("   POST /api/analyze")
    print("   POST /api/tryon")
    print("   GET  /api/health")
    print("   POST /api/analyze-batch (testing)")
    print("=" * 60)
    print("\n🚀 Server starting on http://0.0.0.0:5001\n")
    app.run(debug=False, host='0.0.0.0', port=5001)