"""
writer.py — LLM recommendation writer (grounded, no hallucination)
Supports local GGUF model or API fallback
"""

import os
import json
from typing import Dict, Any, List

# ── LOAD ENVIRONMENT ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# ── FIND MODEL PATH AUTOMATICALLY ──
def find_model_path():
    """Search for the VLM model in multiple locations."""
    possible_paths = [
        # From environment variable
        os.environ.get("VLM_MODEL_PATH", ""),
        # Absolute path (your actual location)
        "/Users/shezamishal19/Desktop/ShadeSense/backend/models/minicpm-v-2.6-Q4_K_M.gguf",
        # Relative from project root
        "backend/models/minicpm-v-2.6-Q4_K_M.gguf",
        "models/minicpm-v-2.6-Q4_K_M.gguf",
        # Current directory
        "minicpm-v-2.6-Q4_K_M.gguf",
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    
    return None

MODEL_PATH = find_model_path()
USE_METAL = os.environ.get("USE_METAL", "true") == "true"

if MODEL_PATH:
    print(f"📂 VLM model found at: {MODEL_PATH}")
else:
    print(f"⚠️ VLM model not found. Using template fallback.")

# ── PROMPT BUILDER ──
def build_recommendation_prompt(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any]
) -> str:
    """
    Build a structured prompt for the LLM.
    All data comes from deterministic sources (no invention allowed).
    """
    # Extract key info
    depth = face_result.get('depth', 'medium')
    undertone = face_result.get('undertone', 'neutral')
    ita = face_result.get('ita_degrees', 0)
    
    # Season info
    season_label = season_data.get('season_label', 'Neutral')
    jewelry = season_data.get('jewelry', ['gold', 'silver'])
    lipstick = season_data.get('lipstick_family', {}).get('name', 'neutral')
    blush = season_data.get('blush_family', {}).get('name', 'neutral')
    
    # Foundation matches (top 3)
    foundations_text = ""
    for i, f in enumerate(foundation_matches[:3], 1):
        foundations_text += f"{i}. {f['brand']} - {f['shade']} (distance: {f['distance']:.2f})\n"
    
    prompt = f"""You are a friendly, professional color analyst assistant. Based on the skin analysis data below, write a warm, helpful response.

User's Skin Profile:
- Skin Depth: {depth}
- Undertone: {undertone}
- Color Season: {season_label}
- ITA Angle: {ita:.1f}°

Recommended Jewelry: {', '.join(jewelry)}
Recommended Lipstick Family: {lipstick}
Recommended Blush Family: {blush}

Top Foundation Matches (from closest to furthest):
{foundations_text}

Please write a friendly, natural-language response that:
1. Briefly explains their skin profile in simple terms
2. Mentions the top foundation match(es) and why they work
3. Gives style recommendations (jewelry, lipstick, blush) based on their season

IMPORTANT: Only reference the data provided above. Do NOT invent any brand names, shade names, or colors that aren't listed here. Do NOT suggest products that aren't in the foundation matches.
"""
    
    return prompt


def generate_recommendation(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any],
    use_llm: bool = True
) -> str:
    """
    Generate a natural-language recommendation.
    Uses local GGUF model if available, falls back to template.
    """
    if not use_llm:
        return _generate_template_response(face_result, foundation_matches, season_data)
    
    # ── TRY LOCAL GGUF MODEL ──
    if MODEL_PATH:
        try:
            from llama_cpp import Llama
            
            print(f"📂 Loading local VLM model from {MODEL_PATH}...")
            
            # Initialize local LLM
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=35 if USE_METAL else 0,
                verbose=False
            )
            
            # Build prompt
            prompt = build_recommendation_prompt(face_result, foundation_matches, season_data)
            
            print("🧠 Generating recommendation with local VLM...")
            
            # Generate response
            response = llm(
                prompt,
                max_tokens=300,
                temperature=0.5,
                stop=["\n\n", "---"],
                echo=False
            )
            
            result = response["choices"][0]["text"].strip()
            if result:
                return result
            
        except ImportError:
            print("⚠️ llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        except Exception as e:
            print(f"⚠️ Local VLM generation failed: {e}")
    
    # ── FALLBACK: Try API (if enabled) ──
    try:
        from openai import OpenAI
        
        api_key = os.environ.get("MINICPM_API_KEY")
        if api_key:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.modelbest.cn/v1"
            )
            
            prompt = build_recommendation_prompt(face_result, foundation_matches, season_data)
            
            response = client.chat.completions.create(
                model="MiniCPM-V-4.6-Instruct",
                messages=[
                    {"role": "system", "content": "You are a friendly color analysis expert. Always base your response on the provided data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ API fallback failed: {e}")
    
    # ── FINAL FALLBACK: Template ──
    return _generate_template_response(face_result, foundation_matches, season_data)


def _generate_template_response(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any]
) -> str:
    """
    Generate a template-based response when LLM is unavailable.
    """
    depth = face_result.get('depth', 'medium')
    undertone = face_result.get('undertone', 'neutral')
    season_label = season_data.get('season_label', 'Neutral')
    
    # Top match
    top_match = foundation_matches[0] if foundation_matches else None
    
    response = f"🌟 Based on your skin analysis, you have {depth} skin with {undertone} undertones.\n\n"
    
    if top_match:
        response += f"💄 Your best foundation match is {top_match['brand']} - {top_match['shade']}. "
        response += f"This shade closely matches your skin tone (color distance: {top_match['distance']:.2f}).\n\n"
    
    # Season recommendations
    if season_data:
        jewelry = season_data.get('jewelry', ['gold', 'silver'])
        lipstick = season_data.get('lipstick_family', {}).get('name', 'neutral')
        blush = season_data.get('blush_family', {}).get('name', 'neutral')
        
        response += f"🍂 Your color season is {season_label}.\n\n"
        response += f"💍 Recommended Jewelry: {', '.join(jewelry)}\n"
        response += f"💋 Recommended Lipstick: {lipstick}\n"
        response += f"🌸 Recommended Blush: {blush}\n"
    
    return response


def get_full_recommendations(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get complete recommendations including structured data and natural language.
    """
    # Generate natural language
    text = generate_recommendation(face_result, foundation_matches, season_data)
    
    # Also provide structured data for the frontend
    return {
        'summary': text,
        'structured': {
            'skin': {
                'depth': face_result.get('depth'),
                'undertone': face_result.get('undertone'),
                'clarity': face_result.get('clarity'),
                'confidence': face_result.get('confidence'),
            },
            'foundations': foundation_matches[:5],
            'season': {
                'label': season_data.get('season_label'),
                'family': season_data.get('family'),
                'jewelry': season_data.get('jewelry', []),
                'lipstick': season_data.get('lipstick_family', {}).get('name'),
                'blush': season_data.get('blush_family', {}).get('name'),
            }
        }
    }


# ── TEST ──
if __name__ == "__main__":
    # Sample data for testing
    sample_face = {
        'depth': 'medium',
        'undertone': 'cool',
        'clarity': 'muted',
        'ita_degrees': 39.4,
        'confidence': 0.45
    }
    
    sample_foundations = [
        {'brand': 'colourpop', 'shade': 'Dark 180', 'distance': 6.69},
        {'brand': 'colourpop', 'shade': 'Deep Dark 115', 'distance': 7.44},
    ]
    
    sample_season = {
        'season_label': 'True Summer (Cool Summer)',
        'family': 'summer',
        'jewelry': ['silver', 'platinum'],
        'lipstick_family': {'name': 'rose pink / berry mauve'},
        'blush_family': {'name': 'rose / cool pink'}
    }
    
    # Generate recommendation
    result = get_full_recommendations(sample_face, sample_foundations, sample_season)
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(result['summary'])