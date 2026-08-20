"""
groq_writer.py — AI-powered recommendation writer using Groq LLM
Enhanced to use rich season data for personalized recommendations.
"""

import os
from typing import Dict, Any, List
from groq import Groq

# ── LOAD ENVIRONMENT ──
try:
    from dotenv import load_dotenv
    env_path = "/Users/shezamishal19/Desktop/ShadeSense/backend/.env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"📂 Loaded .env from: {env_path}")
    else:
        print(f"⚠️ .env not found at: {env_path}")
except Exception as e:
    print(f"⚠️ Could not load .env: {e}")

# ── GET API KEY ──
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY:
    print(f"✅ GROQ_API_KEY found: {GROQ_API_KEY[:10]}...")
else:
    print("⚠️ GROQ_API_KEY not found in environment")


def get_groq_recommendations(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any],
    season_key: str = None
) -> str:
    """
    Use Groq LLM to generate personalized recommendations using rich season data.
    """
    if not GROQ_API_KEY:
        print("⚠️ GROQ_API_KEY not set. Using template fallback.")
        return _generate_template_response(face_result, foundation_matches, season_data)

    try:
        client = Groq(api_key=GROQ_API_KEY)

        rich_season_data = season_data.get('raw_data', {}) or {}

        if not rich_season_data:
            print(f"⚠️ season_data has no 'raw_data' (season_key='{season_data.get('season_key')}').")

        if rich_season_data:
            print(f"✅ Using rich season data for: {rich_season_data.get('season', 'Unknown')}")
            prompt = build_enhanced_ai_prompt(
                face_result,
                foundation_matches,
                season_data,
                rich_season_data
            )
        else:
            print("⚠️ No rich season data found, using basic prompt")
            prompt = build_basic_ai_prompt(face_result, foundation_matches, season_data)

        print("🤖 Calling Groq API...")

        response = client.chat.completions.create(
            model='openai/gpt-oss-120b',
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional color analyst and beauty expert. 

CRITICAL RULES:
1. Use EXACTLY ONE emoji per section header maximum. Do NOT put emojis in the middle of sentences or on every bullet point. Use them sparingly as section markers only.
2. Respond in clean Markdown with section headers (###) and bullet lists.
3. Every color, shade, or product name must be **bolded**.
4. For lipstick, blush, eye makeup, hair color, and clothing - use ONLY the exact shade/color names given in the data. Do NOT invent brand names.
5. Real brand names are ONLY allowed in the Foundation Match section.
6. Be professional and warm, but concise. No fluff, no excessive enthusiasm.
7. Keep it structured - this is a reference card, not a chatty letter."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200,
            stop=["<think>", "Thinking:", "Analysis:"]
        )

        result = response.choices[0].message.content
        print("✅ Groq API response received")
        return result

    except Exception as e:
        print(f"⚠️ Groq API error: {e}")
        return _generate_template_response(face_result, foundation_matches, season_data)


def build_enhanced_ai_prompt(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any],
    rich_season_data: Dict[str, Any]
) -> str:
    """Build a rich, detailed prompt using the enhanced season data."""

    depth = face_result.get('depth', 'medium')
    undertone = face_result.get('undertone', 'neutral')
    clarity = face_result.get('clarity', 'medium')
    confidence = face_result.get('confidence', 0.8)
    ita_degrees = face_result.get('ita_degrees', 0)

    season_label = season_data.get('season_label', 'Neutral')
    season_name = rich_season_data.get('season', season_label)
    season_family = rich_season_data.get('family', '')

    skin_chars = rich_season_data.get('skin_characteristics', {})
    best_colors = rich_season_data.get('best_clothing_colors', {})
    worst_colors = rich_season_data.get('worst_clothing_colors', [])
    jewelry = rich_season_data.get('jewelry', {})
    lipstick = rich_season_data.get('lipstick', {})
    blush = rich_season_data.get('blush', {})
    dress_occasions = rich_season_data.get('dress_occasions', {})
    hair_colors = rich_season_data.get('hair_colors', {})
    eye_makeup = rich_season_data.get('eye_makeup', {})
    special_notes = rich_season_data.get('special_notes', {})

    top_foundations = foundation_matches[:3] if foundation_matches else []
    foundation_text = "\n".join([
        f"  • {match['brand']} - {match['shade']} (ΔE: {match.get('distance', 0):.2f})"
        for match in top_foundations
    ]) if top_foundations else "  • No matches found"

    lipstick_best = lipstick.get('best', [])
    blush_best = blush.get('best', [])

    prompt = f"""
You are a color analyst and beauty expert. Create a structured recommendation card for a client with:

## Skin Profile
- Depth: {depth}
- Undertone: {undertone}
- Clarity: {clarity}
- ITA Value: {ita_degrees:.1f}°
- Confidence: {confidence:.0%}

## Color Season
- Season: {season_name}
- Family: {season_family}

## Best Foundation Matches (Delta-E 2000)
{foundation_text}

## Detailed Season Recommendations

### Skin Characteristics
- Pigmentation Notes: {skin_chars.get('pigmentation_notes', 'Not specified')}

### Best Clothing Colors to Wear
- Primary Colors: {', '.join(best_colors.get('primary', []))}
- Secondary Colors: {', '.join(best_colors.get('secondary', []))}
- Colors to Avoid: {', '.join(worst_colors) if worst_colors else 'Not specified'}

### Jewelry Recommendations
- Best Jewelry Metals: {', '.join(jewelry.get('best', []))}
- Jewelry to Avoid: {', '.join(jewelry.get('avoid', []))}

### Makeup Recommendations
#### Lipstick — ALL of these shades (do not drop any):
{chr(10).join([f"  • {item['name']} ({item.get('hex','')}) - {item['description']}" for item in lipstick_best])}
Lipsticks to Avoid: {', '.join(lipstick.get('avoid', []))}
Note for Pigmented Lips: {lipstick.get('pigmented_lips_note', '')}

#### Blush — ALL of these shades (do not drop any):
{chr(10).join([f"  • {item['name']} ({item.get('hex','')}) - {item['description']}" for item in blush_best])}
Blush to Avoid: {', '.join(blush.get('avoid', []))}

#### Eye Makeup
Best Eye Colors: {', '.join(eye_makeup.get('best', []))}
Eye Colors to Avoid: {', '.join(eye_makeup.get('avoid', []))}

### Hair Color Guidance
Best Hair Colors: {', '.join(hair_colors.get('best', []))}
Hair Colors to Avoid: {', '.join(hair_colors.get('avoid', []))}
Highlight Notes: {hair_colors.get('highlight_notes', '')}

### Outfit Ideas for Different Occasions
- Casual: {', '.join(dress_occasions.get('casual', []))}
- Office Wear: {', '.join(dress_occasions.get('office', []))}
- Evening Looks: {', '.join(dress_occasions.get('evening', []))}
- Wedding Guest: {', '.join(dress_occasions.get('wedding_guest', []))}
- Summer Dresses: {', '.join(dress_occasions.get('summer_dresses', []))}

### Special Skin Considerations
- Pigmented Lips: {special_notes.get('pigmented_lips', '')}
- Hyperpigmentation: {special_notes.get('hyperpigmentation', '')}
- Olive Skin Tone: {special_notes.get('olive_skin', '')}

## OUTPUT FORMAT — FOLLOW EXACTLY
Respond in Markdown with these EXACT sections. Use ONE emoji per section header maximum (e.g., "### 🌟 Your Skin & Season"). Do NOT put emojis on bullet points or in the middle of sentences.

### 🌟 Your Skin & Season
2-3 professional sentences summarizing their skin profile and season.

### 💄 Foundation Match
Bullet list of the top foundation matches with a brief reason.

### 👄 Lipstick Shades
One bullet per shade from the data above. Bold the name, include the hex in backticks.

### 🌸 Blush Shades
One bullet per shade from the data above. Bold the name, include the hex in backticks.

### 👁️ Eye Makeup
Bullet list of recommended eye colors, bolded. Then list what to avoid.

### 💍 Jewelry
Bullet list: best metals bolded, what to avoid.

### 👗 Clothing Colors
- **Primary colors:** bullet list
- **Secondary colors:** bullet list
- **Colors to avoid:** bullet list

### 👚 Outfit Ideas by Occasion
Sub-bullet lists for Casual, Office, Evening, Wedding Guest, Summer.

### 💇 Hair Color
Bullet list of best hair colors (bolded) and what to avoid.

### ✨ Special Notes
Only include if the data above has content for pigmented lips, hyperpigmentation, or olive skin. Skip entirely if all are blank.

Keep it professional, structured, and concise. No fluff. No excessive enthusiasm. This is a reference card, not a chat message.
"""

    return prompt


def build_basic_ai_prompt(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any]
) -> str:
    """Fallback prompt when rich season data isn't available."""
    depth = face_result.get('depth', 'medium')
    undertone = face_result.get('undertone', 'neutral')
    season_label = season_data.get('season_label', 'Neutral')
    jewelry = season_data.get('jewelry', ['gold', 'silver'])
    lipstick = season_data.get('lipstick_family', {}).get('name', 'neutral')
    blush = season_data.get('blush_family', {}).get('name', 'neutral')

    top_foundation = foundation_matches[0] if foundation_matches else None
    foundation_text = f"{top_foundation['brand']} - {top_foundation['shade']}" if top_foundation else "None found"

    prompt = f"""
Based on this skin data:

Skin: {depth} with {undertone} undertones
Season: {season_label}
Best Foundation: {foundation_text}
Jewelry: {', '.join(jewelry)}
Lipstick family: {lipstick}
Blush family: {blush}

NOTE: Detailed per-shade data isn't available for this season match, so recommend general shades within the given family.

Respond in Markdown with these bolded sections: **Your Skin Profile**, **Best Foundation Match**, **Recommended Colors** (jewelry/makeup), **Why These Choices Work**.

Use ONE emoji per section maximum. Be professional and structured. No excessive emojis.
"""

    return prompt


def _generate_template_response(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any]
) -> str:
    """Template fallback when Groq is unavailable."""
    depth = face_result.get('depth', 'medium')
    undertone = face_result.get('undertone', 'neutral')
    season_label = season_data.get('season_label', 'Neutral')

    top_match = foundation_matches[0] if foundation_matches else None

    response = f"## Your Skin Profile\n\n**Skin:** {depth} with {undertone} undertones\n\n"

    if top_match:
        response += f"**Best Foundation Match:** {top_match['brand']} - {top_match['shade']}\n\n"

    if season_data:
        jewelry = season_data.get('jewelry', ['gold', 'silver'])
        lipstick = season_data.get('lipstick_family', {}).get('name', 'neutral')
        blush = season_data.get('blush_family', {}).get('name', 'neutral')

        response += f"**Season:** {season_label}\n\n"
        response += f"**Jewelry:** {', '.join(jewelry)}\n"
        response += f"**Lipstick:** {lipstick}\n"
        response += f"**Blush:** {blush}\n"

    return response


def get_ai_recommendations(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main function to get AI recommendations.
    """
    season_key = season_data.get('season_key')
    text = get_groq_recommendations(face_result, foundation_matches, season_data, season_key)
    rich_season_data = season_data.get('raw_data', {}) or {}

    return {
        'summary': text,
        'structured': {
            'skin': {
                'depth': face_result.get('depth'),
                'undertone': face_result.get('undertone'),
                'clarity': face_result.get('clarity'),
                'confidence': face_result.get('confidence'),
                'ita_degrees': face_result.get('ita_degrees'),
            },
            'foundations': foundation_matches[:5],
            'season': {
                'label': season_data.get('season_label'),
                'family': season_data.get('family'),
                'jewelry': rich_season_data.get('jewelry', {}).get('best', season_data.get('jewelry', [])),
                'lipstick': rich_season_data.get('lipstick', {}).get('best', []),
                'blush': rich_season_data.get('blush', {}).get('best', []),
                'best_colors': rich_season_data.get('best_clothing_colors', {}),
                'worst_colors': rich_season_data.get('worst_clothing_colors', []),
            },
            'recommendations': {
                'dress_occasions': rich_season_data.get('dress_occasions', {}),
                'hair_colors': rich_season_data.get('hair_colors', {}),
                'eye_makeup': rich_season_data.get('eye_makeup', {}),
                'special_notes': rich_season_data.get('special_notes', {})
            }
        }
    }