"""
groq_writer.py — AI-powered recommendation writer using Groq LLM
Enhanced to use rich season data for personalized recommendations.

CHANGE: The AI text now writes ONLY a short narrative (skin/season summary +
foundation match reasoning). It no longer re-lists lipstick/blush/jewelry/
clothing/hair shades as bullet text, because that data is already rendered
as colored swatch chips by renderColorPalette() on the frontend — listing it
twice made the AI card look like a redundant wall of plain-text duplicates
of the chips shown right above it.
"""

import os
from typing import Dict, Any, List
from urllib import response
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


# ── SHARED SYSTEM PROMPT ──
NARRATIVE_SYSTEM_PROMPT = """You are a professional color analyst writing a short,
grounded summary for a client — think a knowledgeable friend explaining their
results clearly, not a marketing blurb.

CRITICAL RULES:
1. Write ONLY two sections: a skin/season summary and foundation match
   reasoning. Do NOT list lipstick, blush, jewelry, clothing, or hair
   shades — those are already shown as colored swatch chips elsewhere.
2. Do NOT quote raw numbers (percentages, ITA degrees, ΔE values, margin
   points) in your prose. Use them internally to decide your tone and
   wording, but write the conclusion in plain language instead of citing
   the figure. Example: instead of "confidence is 38%, so this is not
   absolute," write "this reading is a solid starting point, though your
   coloring has some flexibility."
3. AVOID overused AI phrases: "radiates a [X] vibe", "sun-kissed", "lands
   perfectly in", "outpacing by a wide margin", "confidently embrace".
4. Vary sentence openers across responses.
5. One emoji per section header maximum.
6. Respond in Markdown with ### headers and flowing prose.
7. Real brand names only in the Foundation Match section.
8. Be warm, concise, and specific — two short paragraphs total, no filler."""


def get_groq_recommendations(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any],
    season_key: str = None
) -> str:
    """
    Use Groq LLM to generate a short personalized narrative using rich season data.
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
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=700,
        )

        result = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        usage = getattr(response, "usage", None)
        print(f"✅ Groq API response received (ranked) — finish_reason={finish_reason}, len={len(result or '')}, usage={usage}")
        return {
            "summary": result,
            "structured": structured,
            "ranked_seasons": ranked_seasons,
        }
    except Exception as e:
        print(f"⚠️ Groq API error: {e}")
        return _generate_template_response(face_result, foundation_matches, season_data)


def build_enhanced_ai_prompt(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any],
    rich_season_data: Dict[str, Any]
) -> str:
    """Build a prompt for a short narrative using the enhanced season data.

    NOTE: lipstick/blush/jewelry/clothing/hair/outfit data is still passed in
    below so Groq can reference the *mood* of the palette in its summary
    (e.g. "soft, dusty jewel tones") — but the OUTPUT FORMAT explicitly
    forbids listing individual shade names, since those are already shown
    as colored chips on the frontend.
    """

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
    lipstick = rich_season_data.get('lipstick', {})
    blush = rich_season_data.get('blush', {})
    special_notes = rich_season_data.get('special_notes', {})

    top_foundations = foundation_matches[:3] if foundation_matches else []
    foundation_text = "\n".join([
        f"  • {match['brand']} - {match['shade']} (ΔE: {match.get('distance', 0):.2f})"
        for match in top_foundations
    ]) if top_foundations else "  • No matches found"

    lipstick_best = lipstick.get('best', [])
    blush_best = blush.get('best', [])

    has_special_notes = any([
        special_notes.get('pigmented_lips'),
        special_notes.get('hyperpigmentation'),
        special_notes.get('olive_skin'),
    ])

    prompt = f"""
You are writing a short narrative summary for a client with this skin/season profile.
Reference material is provided below so you understand the *mood* of their palette —
but do NOT list individual shade names from it; those are already shown to the
client as colored chips elsewhere on the page.

## Skin Profile
- Depth: {depth}
- Undertone: {undertone}
- Clarity: {clarity}
- ITA Value: {ita_degrees:.1f}°
- Confidence: {confidence:.0%}

## Color Season
- Season: {season_name}
- Family: {season_family}
- Pigmentation Notes: {skin_chars.get('pigmentation_notes', 'Not specified')}

## Best Foundation Matches (Delta-E 2000) — the only place you may name brands/products
{foundation_text}

## Reference only (do NOT list these by name — use them to understand the palette's mood)
- Primary clothing colors: {', '.join(best_colors.get('primary', []))}
- Secondary clothing colors: {', '.join(best_colors.get('secondary', []))}
- Lipstick mood: {', '.join(item['name'] for item in lipstick_best)}
- Blush mood: {', '.join(item['name'] for item in blush_best)}

## Special notes (only use if writing the third section below)
- Pigmented Lips: {special_notes.get('pigmented_lips', '')}
- Hyperpigmentation: {special_notes.get('hyperpigmentation', '')}
- Olive Skin Tone: {special_notes.get('olive_skin', '')}

## OUTPUT FORMAT

Respond in Markdown with ONLY these two sections, with one emoji per header maximum.

### 🌟 Your Skin & Season

Write 3–4 natural, personalized sentences.

Your job is to INTERPRET the supplied analysis, not simply repeat it.

Start with the most useful characteristic of this person's coloring:
- undertone
- depth
- clarity
- contrast
- or whether the season classification was decisive or close.

Name the final season once.

Explain what this means practically for the person's coloring.

For example, depending on the supplied analysis, you may explain that they are likely to suit:
- warm vs cool colors
- clear vs muted colors
- rich vs delicate colors
- high vs low contrast combinations
- brighter vs softer shades

Use natural beauty language such as:
"warm shades", "clear colors", "rich tones", "soft muted colors",
"golden hues", "cool jewel tones", etc. when supported by the season data.

Do NOT use generic seasonal marketing language such as:
"your skin glows", "lively and energetic", "radiant beauty", "perfect harmony",
unless it is genuinely relevant.

If the ranked season results are close, explicitly acknowledge the uncertainty
and explain what the two seasons have in common instead of pretending the
classification is absolute.

If the image-analysis confidence is moderate or low, do NOT describe the
classification as certain or definitive.

Do NOT mention the numerical season score or ranking margin unless it is
necessary to explain ambiguity.

Do NOT invent characteristics that are not present in the supplied analysis.

### 💄 Foundation Match

Lead with the single best foundation match using its actual brand and shade.

Use the ΔE value to honestly describe the match quality:

- ΔE < 1:
  "near-perfect" or "extremely close"
- ΔE 1–2:
  "very close"
- ΔE 2–3.5:
  "good match"
- ΔE > 3.5:
  "workable match" and explicitly mention that there is a
  noticeable difference and that testing it in person is recommended.
Do not mention numeric scores, percentages, ITA degrees, or ΔE values directly
in your prose — translate them into plain descriptive language instead.

Then briefly explain WHY the match is useful based on the available data,
such as depth, undertone, or colorimetric similarity.

If a second foundation is worth mentioning, mention it only if it provides
a meaningful alternative, such as:
- a similar color with a different undertone
- a slightly lighter/deeper option
- a closer undertone match
- a similar match with a meaningfully different ΔE

Do NOT recommend a foundation simply because its product name sounds suitable.

Do NOT invent product properties such as coverage, finish, oxidation,
formulation, or wear unless those properties are explicitly provided.

Lipstick, blush, jewelry, clothing, and hair recommendations are already
displayed elsewhere in the UI. Do not repeat their exact shade names here.

Keep the entire response concise, warm, practical, and personalized.
The goal is to sound like an AI beauty consultant interpreting the analysis,
not like a technical diagnostic report.""".strip()

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

    top_foundation = foundation_matches[0] if foundation_matches else None
    foundation_text = f"{top_foundation['brand']} - {top_foundation['shade']}" if top_foundation else "None found"

    prompt = f"""
Based on this skin data:

Skin: {depth} with {undertone} undertones
Season: {season_label}
Best Foundation: {foundation_text}

Write a short, warm narrative in Markdown with exactly two sections:

### 🌟 Your Skin & Season
2-3 sentences describing their skin and season in plain language. Do not
list individual color/shade names — describe the general mood of the
palette instead (e.g. "warm, golden tones").

### 💄 Foundation Match
1-2 sentences on why the foundation above is a good match.

One emoji per header maximum. Be professional and concise, not chatty.
"""

    return prompt


def _generate_template_response(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any]
) -> str:
    depth = face_result.get('depth', 'medium')
    undertone = face_result.get('undertone', 'neutral')
    season_label = season_data.get('season_label', 'Neutral')
    confidence = face_result.get('confidence', 0)

    response = (
        f"### 🌟 Your Skin & Season\n\n"
        f"Your skin reads as **{depth}** depth with a **{undertone}** undertone "
        f"({confidence:.0%} confidence), placing you in the **{season_label}** palette.\n\n"
    )

    top_match = foundation_matches[0] if foundation_matches else None
    if top_match:
        delta_e = top_match.get('distance', 0)
        closeness = "an excellent, nearly imperceptible match" if delta_e < 1 else "a solid match"
        response += (
            f"### 💄 Foundation Match\n\n"
            f"Your closest match is **{top_match['brand']} - {top_match['shade']}** "
            f"(ΔE {delta_e:.2f}) — {closeness}.\n"
        )

    return response

def get_ai_recommendations(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main function to get AI recommendations. The 'structured' payload still
    carries the full shade/jewelry/hair/etc. data for the frontend's colored
    palette chips — only the narrative 'summary' text was trimmed.
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

# ══════════════════════════════════════════════════════════════
# RANKED-AWARE PROMPT — additive, does not replace
# build_enhanced_ai_prompt() or get_groq_recommendations().
#
# Same trim applied here: only a narrative summary + foundation
# reasoning are requested. Shade lists per candidate are still
# passed IN to the prompt (so Groq can reason about closeness/
# overlap in its prose), but are no longer requested as bulleted
# OUTPUT — the frontend's colored chips already show them.
# ══════════════════════════════════════════════════════════════

def build_ranked_ai_prompt(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    ranked_seasons: Dict[str, Any],
    seasons_full_data: Dict[str, Dict[str, Any]],
) -> str:
    """
    Build a prompt that gives Groq the top 2-3 season candidates with
    their scores, and asks it to write a short narrative that's honest
    about closeness — without re-listing every shade as bullets.
    """
    depth = face_result.get('depth', 'medium')
    undertone = face_result.get('undertone', 'neutral')
    clarity = face_result.get('clarity', 'medium')
    confidence = face_result.get('confidence', 0.8)
    ita_degrees = face_result.get('ita_degrees', 0)

    top = ranked_seasons["top_seasons"]
    is_close = ranked_seasons["is_close_call"]
    margin_pct = ranked_seasons["margin_pct"]

    top_foundations = foundation_matches[:3] if foundation_matches else []
    foundation_text = "\n".join([
        f"  • {match['brand']} - {match['shade']} (ΔE: {match.get('distance', 0):.2f})"
        for match in top_foundations
    ]) if top_foundations else "  • No matches found"

    # Compact reference data per candidate — for Groq's understanding only,
    # not to be listed verbatim in the output (see OUTPUT FORMAT below).
    candidate_blocks = []
    for rank, candidate in enumerate(top, start=1):
        key = candidate["season_key"]
        full = seasons_full_data.get(key, {})
        lipstick_names = ", ".join(item["name"] for item in full.get("lipstick", {}).get("best", []))
        blush_names = ", ".join(item["name"] for item in full.get("blush", {}).get("best", []))

        candidate_blocks.append(f"""
Candidate #{rank}: {candidate['season_label']} (match score: {candidate['score_pct']}%)
  Lipstick mood: {lipstick_names}
  Blush mood: {blush_names}""")

    candidates_text = "\n".join(candidate_blocks)

    closeness_instruction = (
    f"""
This is a close call between {top[0]['season_label']} and {top[1]['season_label']}.
Acknowledge the profile shows traits of both, in plain language — don't cite
percentages or margins, just explain what they have in common and lean toward
{top[0]['season_label']} as the primary read.
"""
    if is_close else
    f"""
{top[0]['season_label']} is the clear, confident classification. Present it
as the primary result without hedging or citing scores.
"""
)

    prompt = f"""
You are writing a short narrative summary for a client. Reference material
for each season candidate is provided below so you understand the mood of
each palette — do NOT list individual shade names from it in your output.

## Skin Profile
- Depth: {depth}
- Undertone: {undertone}
- Clarity: {clarity}
- ITA Value: {ita_degrees:.1f}°
- Confidence: {confidence:.0%}

## Best Foundation Matches (Delta-E 2000) — the only place you may name brands/products
{foundation_text}

## Season Candidates (reference only — do not list by name in your output)
{candidates_text}

## Your task
{closeness_instruction}

## OUTPUT FORMAT
Respond in Markdown with ONLY these two sections, one emoji per header max:

### 🌟 Your Skin & Season
Write 4-6 sentences. Don't just state the season — explain the reasoning
chain: which characteristic (undertone/depth/clarity/contrast) drove the
classification, what that characteristic means for color choice in
practice, and what this person should look for or avoid as a *principle*
(not a shade list) when shopping. If the confidence is moderate, say so
and explain what that uncertainty means for how strictly they should
follow the season rules.

### 💄 Foundation Match
Write 4-5 sentences. Lead with the shade and ΔE-based verdict, then explain
concretely what that ΔE means in real terms (visible under what lighting,
how it'll wear through the day, whether undertone or depth is the bigger
factor in the match). If a second option is mentioned, explain the specific
scenario where someone would reach for it instead of the top pick.
"""
    return prompt


def get_ai_recommendations_ranked(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    ranked_seasons: Dict[str, Any],
    seasons_full_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ranked-aware entry point using season scoring instead of a single
    pre-decided winner. Falls back to the template response if Groq is
    unavailable.

    Returns the same 'structured' shape as get_ai_recommendations() so
    app.py and the frontend's renderColorPalette() get full season data
    (best/worst colors, eye makeup, hair, dress occasions, special notes)
    regardless of which entry point was used.
    """
    print("🔥🔥🔥 THIS IS THE NEW CODE 🔥🔥🔥")

    winner = ranked_seasons["top_seasons"][0]
    winner_key = winner["season_key"]
    rich_season_data = seasons_full_data.get(winner_key, {}) or {}

    structured = {
        'skin': {
            'depth': face_result.get('depth'),
            'undertone': face_result.get('undertone'),
            'clarity': face_result.get('clarity'),
            'confidence': face_result.get('confidence'),
            'ita_degrees': face_result.get('ita_degrees'),
        },
        'foundations': foundation_matches[:5],
        'season': {
            'label': winner.get('season_label'),
            'family': rich_season_data.get('family'),
            'jewelry': rich_season_data.get('jewelry', {}).get('best', []),
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
        },
        'ranked_seasons': ranked_seasons,
    }

    if not GROQ_API_KEY:
        print("⚠️ GROQ_API_KEY not set. Using template fallback.")
        fake_season_data = {"season_label": winner["season_label"]}
        return {
            "summary": _generate_template_response(face_result, foundation_matches, fake_season_data),
            "structured": structured,
            "ranked_seasons": ranked_seasons,
        }

    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = build_ranked_ai_prompt(face_result, foundation_matches, ranked_seasons, seasons_full_data)

        print(f"🤖 Calling Groq API (ranked, close_call={ranked_seasons['is_close_call']})...")

        response = client.chat.completions.create(
            model='openai/gpt-oss-120b',
            messages=[
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=700,
            
        )

        result = response.choices[0].message.content
        print("✅ Groq API response received (ranked)")
        return {
            "summary": result,
            "structured": structured,
            "ranked_seasons": ranked_seasons,
        }

    except Exception as e:
        print(f"⚠️ Groq API error (ranked): {e}")
        fake_season_data = {"season_label": winner["season_label"]}
        return {
            "summary": _generate_template_response(face_result, foundation_matches, fake_season_data),
            "structured": structured,
            "ranked_seasons": ranked_seasons,
        }