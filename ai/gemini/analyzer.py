import json
import os
import re
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# Try to import openai, fallback to requests if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    import requests


SYSTEM_PROMPT = """
You are a dermatological color analysis expert specializing in visible skin tone
and undertone classification from face images.

Your task:
- Analyze only visible facial skin areas.
- Ignore makeup, clothing, hair, background, jewelry, lighting artifacts, and shadows.
- Classify skin_tone as exactly one of: "Light", "Medium", "Deep".
- Classify undertone as exactly one of: "Cool", "Warm", "Neutral", "Olive".
- Return only valid JSON matching the schema.

JSON schema:
{
  "skin_tone": "Light | Medium | Deep",
  "undertone": "Cool | Warm | Neutral | Olive",
  "confidence": 0.0
}

Confidence rules:
- Use 0.8-1.0 only when the face is clearly visible with even lighting.
- Use 0.6-0.79 when lighting or image quality is somewhat imperfect.
- Use below 0.6 when the image has strong makeup, shadows, color casts, filters,
  poor lighting, occlusion, or insufficient visible skin.

Do not include explanations, markdown, comments, or extra keys.
"""


# MiniCPM-V API Configuration
MINICPM_API_BASE = "https://api.modelbest.cn/v1"
MINICPM_API_KEY = "sk-pQ8L2zF3XmR5kY9wV4jB7hN1tC6vM0xG3aD5sH2bJ9lK4cZ8"
MINICPM_MODEL = "MiniCPM-V-4.6-Instruct"  # or "MiniCPM-V-4.6-Thinking"


VALID_SKIN_TONES = {"Light", "Medium", "Deep"}
VALID_UNDERTONES = {"Cool", "Warm", "Neutral", "Olive"}


@dataclass
class SkinAnalysisResult:
    skin_tone: str
    undertone: str
    confidence: float
    source: str


def load_all_examples(examples_dir="ai/examples"):
    """Load all images from subfolders as few-shot examples"""
    examples = []
    
    examples_path = Path(examples_dir)
    if not examples_path.exists():
        print(f"⚠️ Examples directory not found: {examples_dir}")
        return []
    
    for category_folder in examples_path.iterdir():
        if not category_folder.is_dir():
            continue
        
        category = category_folder.name
        parts = category.split("_")
        if len(parts) != 2:
            continue
        
        skin_tone, undertone = parts
        skin_tone = skin_tone.capitalize()
        undertone = undertone.capitalize()
        
        if skin_tone not in VALID_SKIN_TONES:
            continue
        if undertone not in VALID_UNDERTONES:
            continue
        
        image_files = list(category_folder.glob("*.jpg")) + list(category_folder.glob("*.png")) + list(category_folder.glob("*.jpeg"))
        image_files = image_files[:5]  # Take up to 5 images per category
        
        for img_path in image_files:
            examples.append({
                "image_path": str(img_path),
                "output": {
                    "skin_tone": skin_tone,
                    "undertone": undertone,
                    "confidence": 0.90,
                },
            })
    
    if len(examples) == 0:
        print(f"⚠️ No examples found in {examples_dir}")
        return []
    
    print(f"✅ Loaded {len(examples)} few-shot examples from {examples_dir}")
    return examples


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for API transmission"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_with_minicpm(
    image_path: str,
    api_key: Optional[str] = None,
    model_name: str = MINICPM_MODEL,
) -> Dict[str, Any]:
    """
    Analyze skin using MiniCPM-V API.
    
    Args:
        image_path: Path to face image
        api_key: API key (optional, uses default if not provided)
        model_name: Model name (default: MiniCPM-V-4.6-Instruct)
    
    Returns:
        Dict with skin_tone, undertone, confidence
    """
    api_key = api_key or MINICPM_API_KEY
    
    # Encode image
    base64_image = encode_image_to_base64(image_path)
    
    # Build the prompt
    prompt = SYSTEM_PROMPT + "\n\n"
    prompt += "Analyze the skin tone and undertone in this face image. Return ONLY valid JSON."
    
    # Try OpenAI SDK first
    if OPENAI_AVAILABLE:
        try:
            client = openai.OpenAI(
                base_url=MINICPM_API_BASE,
                api_key=api_key,
            )
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=256,
            )
            
            result_text = response.choices[0].message.content
            return _parse_gemini_json(result_text)
            
        except Exception as e:
            print(f"⚠️ OpenAI SDK error: {e}")
            # Fall through to direct HTTP request
    
    # Fallback to direct HTTP request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }
    
    response = requests.post(
        f"{MINICPM_API_BASE}/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    result_text = response.json()["choices"][0]["message"]["content"]
    
    return _parse_gemini_json(result_text)


def analyze_skin_undertone(
    face_image_path: str,
    api_key: Optional[str] = None,
    model_name: str = MINICPM_MODEL,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    low_confidence_threshold: float = 0.6,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Analyze skin tone and undertone using MiniCPM-V API.
    
    Args:
        face_image_path: Path to the face image to classify.
        api_key: API key. Defaults to MINICPM_API_KEY.
        model_name: Model name (default: MiniCPM-V-4.6-Instruct).
        few_shot_examples: List of examples (not used for MiniCPM-V, kept for compatibility).
        low_confidence_threshold: Use heuristic fallback if confidence is below this.
        temperature: Temperature for generation.
    
    Returns:
        Dict with skin_tone, undertone, confidence, and source.
    """
    try:
        result = analyze_with_minicpm(face_image_path, api_key, model_name)
        
        if result["confidence"] >= low_confidence_threshold:
            return {
                **result,
                "source": "minicpm",
            }
        
        # Low confidence fallback
        fallback = heuristic_skin_analysis(face_image_path)
        return {
            **fallback,
            "source": "heuristic_fallback",
            "minicpm_result": result,
        }
        
    except Exception as exc:
        print(f"⚠️ MiniCPM-V API error: {exc}")
        fallback = heuristic_skin_analysis(face_image_path)
        return {
            **fallback,
            "source": "heuristic_error_fallback",
            "error": str(exc),
        }


def _parse_gemini_json(text: str) -> Dict[str, Any]:
    """Parse JSON response from the API"""
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError(f"Could not parse JSON from: {text[:100]}...")
    
    return _validate_and_normalize_result(data)


def _validate_and_normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    skin_tone = data.get("skin_tone")
    undertone = data.get("undertone")
    confidence = float(data.get("confidence", 0.0))
    
    if skin_tone not in VALID_SKIN_TONES:
        # Try to match case-insensitively
        for valid in VALID_SKIN_TONES:
            if skin_tone.lower() == valid.lower():
                skin_tone = valid
                break
        else:
            raise ValueError(f"Invalid skin_tone: {skin_tone}")
    
    if undertone not in VALID_UNDERTONES:
        for valid in VALID_UNDERTONES:
            if undertone.lower() == valid.lower():
                undertone = valid
                break
        else:
            raise ValueError(f"Invalid undertone: {undertone}")
    
    confidence = max(0.0, min(1.0, confidence))
    
    return {
        "skin_tone": skin_tone,
        "undertone": undertone,
        "confidence": confidence,
    }


def heuristic_skin_analysis(image_path: str) -> Dict[str, Any]:
    """Simple fallback based on approximate skin-colored pixels"""
    image = Image.open(image_path).convert("RGB")
    image = _center_crop(image, crop_ratio=0.72)
    image.thumbnail((512, 512))
    
    rgb = np.asarray(image).astype(np.float32)
    pixels = rgb.reshape(-1, 3)
    
    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]
    
    max_rgb = np.max(pixels, axis=1)
    min_rgb = np.min(pixels, axis=1)
    
    skin_mask = (
        (r > 45)
        & (g > 35)
        & (b > 25)
        & ((max_rgb - min_rgb) > 10)
        & (r > b)
        & (r >= g * 0.85)
        & (r <= g * 1.8)
    )
    
    skin_pixels = pixels[skin_mask]
    if len(skin_pixels) < 100:
        skin_pixels = pixels
    
    median_rgb = np.median(skin_pixels, axis=0)
    r_med, g_med, b_med = median_rgb
    
    luminance = 0.2126 * r_med + 0.7152 * g_med + 0.0722 * b_med
    
    if luminance >= 170:
        skin_tone = "Light"
    elif luminance >= 95:
        skin_tone = "Medium"
    else:
        skin_tone = "Deep"
    
    red_green = r_med - g_med
    yellow_blue = ((r_med + g_med) / 2.0) - b_med
    green_cast = g_med - ((r_med + b_med) / 2.0)
    
    if green_cast > 6 and yellow_blue > 8:
        undertone = "Olive"
    elif yellow_blue > 18 and red_green > -2:
        undertone = "Warm"
    elif b_med > g_med - 2 or red_green > 18 and yellow_blue < 14:
        undertone = "Cool"
    else:
        undertone = "Neutral"
    
    confidence = 0.55 if len(skin_pixels) >= 100 else 0.4
    
    return {
        "skin_tone": skin_tone,
        "undertone": undertone,
        "confidence": confidence,
    }


def _center_crop(image: Image.Image, crop_ratio: float = 0.72) -> Image.Image:
    width, height = image.size
    crop_w = int(width * crop_ratio)
    crop_h = int(height * crop_ratio)
    
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    
    return image.crop((left, top, right, bottom))


if __name__ == "__main__":
    # Test the analyzer
    examples = load_all_examples("ai/examples")
    
    result = analyze_skin_undertone(
        face_image_path="test_face.jpg",
        few_shot_examples=examples,
    )
    
    print(json.dumps(result, indent=2))