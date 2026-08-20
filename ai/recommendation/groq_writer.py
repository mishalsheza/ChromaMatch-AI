"""
groq_writer.py — AI-powered recommendation writer using Groq LLM
Enhanced to use rich season data for personalized recommendations.
"""

import os
from typing import Dict, Any, List, Optional
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
1. Write ONLY the sections requested in the output format. Do NOT list lipstick,
   blush, jewelry, clothing, or hair shades — those are already shown as
   colored swatch chips elsewhere.
2. Do NOT quote raw numbers (percentages, ITA degrees, ΔE values, margin
   points) in your prose. Use them internally to decide your tone and
   wording, but write the conclusion in plain language instead of citing
   the figure.
3. AVOID overused AI phrases: "radiates a [X] vibe", "sun-kissed", "lands
   perfectly in", "outpacing by a wide margin", "confidently embrace".
4. Vary sentence openers across responses.
5. One emoji per section header maximum.
6. Respond in Markdown with ### headers and flowing prose.
7. Real brand names only in the Foundation Match section.
8. Be warm, concise, and specific — no filler."""


# ═══════════════════════════════════════════════════════════════
# SKIN OBSERVATIONS — DETERMINISTIC, NOT DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════

def build_observations_text(observations: Dict[str, Any]) -> str:
    """
    Turns detected color-space deltas into (a) a plain-language observation
    summary and (b) ONLY the relevant advice block(s) for what was found.
    Minimal-severity findings are omitted entirely — not worth flagging.
    """
    if not observations or not observations.get('has_observations', False):
        return "No notable observations beyond the primary skin analysis."

    lines = []
    guidance_blocks = []

    # ── Under-eye ──
    under_eye = observations.get('under_eye')
    if under_eye and under_eye.get('severity') in ('mild', 'pronounced','slight'):
        cast = under_eye.get('cast', 'neutral')
        lines.append(
            f"- Under-eye area: {under_eye['severity']} darkness relative to "
            f"cheek baseline, {cast} cast"
        )
        if cast == 'blue-purple':
            guidance_blocks.append(
                "For the under-eye area's blue-purple cast: warm peach/coral "
                "concealers neutralize it; cool-toned or deep shadow colors "
                "near the eyes will emphasize it."
            )
        elif cast == 'brown':
            guidance_blocks.append(
                "For the under-eye area's brown cast: yellow-based correctors "
                "help; ash-toned concealers can make it read grayer."
            )

    # ── Perioral ──
    perioral = observations.get('perioral')
    if perioral and perioral.get('severity') in ('mild', 'pronounced'):
        lines.append(
            f"- Perioral area: {perioral['severity']} discoloration relative "
            f"to jaw baseline"
        )
        guidance_blocks.append(
            "For perioral discoloration: warm/yellow-based concealer helps; "
            "cool pinks or ash tones emphasize it."
        )

    # ── Redness ──
    redness = observations.get('redness')
    if redness and redness.get('severity') in ('mild', 'pronounced'):
        lines.append(f"- Cheek area: {redness['severity']} redness relative to baseline")
        guidance_blocks.append(
            "For redness: green-correcting primer helps; overly warm foundations "
            "can emphasize it."
        )

    # ── Tone evenness (informational, no guidance) ──
    if observations.get('tone_evenness') == 'uneven':
        lines.append("- Skin tone: some variation across regions (common in natural skin)")

    if not lines:
        return "No notable observations beyond the primary skin analysis."

    observation_summary = "\n".join(lines)

    if guidance_blocks:
        guidance_summary = "\n".join(f"- {g}" for g in guidance_blocks)
        return f"{observation_summary}\n\nRelevant guidance:\n{guidance_summary}"

    return observation_summary


# ═══════════════════════════════════════════════════════════════
# RANKED-AWARE PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_ranked_ai_prompt(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    ranked_seasons: Dict[str, Any],
    seasons_full_data: Dict[str, Dict[str, Any]],
    observations: Optional[Dict[str, Any]] = None,
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

    top_foundations = foundation_matches[:3] if foundation_matches else []
    foundation_text = "\n".join([
        f"  • {match['brand']} - {match['shade']} (ΔE: {match.get('distance', 0):.2f})"
        for match in top_foundations
    ]) if top_foundations else "  • No matches found"

    # Compact reference data per candidate — for Groq's understanding only
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

    # ── Build observations text (only if present) ──
    observations_text = build_observations_text(observations or {})
    has_observations = observations and observations.get('has_observations', False)

    # ── Assemble the full prompt ──
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
"""

    # ── ONLY add observations section if there's something to say ──
    if has_observations:
        prompt += f"""

## Skin Observations (color-space derived, not a diagnosis)
{observations_text}

## GUIDANCE FOR SKIN OBSERVATIONS
- Acknowledge any observations in cosmetic terms only (e.g., "the under-eye
  area reads slightly darker than your cheek tone").
- NEVER use medical/clinical terms: "hyperpigmentation," "melasma,"
  "periorbital hyperpigmentation," "rosacea," "dark circles" as a diagnosis.
- Frame everything constructively — as a styling/concealer tip, never as
  a flaw or problem.
- Use the specific advice from "Relevant guidance" above.
"""
    else:
        prompt += """

## No notable skin observations were detected beyond the primary analysis.
Do not invent or mention any skin concerns.
"""

    # ── Task and output format ──
    prompt += f"""

## Your task
{closeness_instruction}

## OUTPUT FORMAT
Respond in Markdown with the following sections. Only include the "A Few Extra Notes" section if skin observations were provided above.

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

    # ── Only add the third section if there are observations ──
    if has_observations:
        prompt += """

### 💡 A Few Extra Notes
Write 2-3 sentences with the concealer/color advice from the observations.
Be warm and constructive — e.g., "A peach-toned concealer can help balance
the natural shadow-like discoloration under the eyes..." rather than
"This is a problem." Keep it brief and practical.
"""

    prompt += """

One emoji per section header maximum. Keep the tone warm, grounded, and specific.
"""

    return prompt.strip()


# ═══════════════════════════════════════════════════════════════
# LEGACY/BASIC PROMPT (fallback when no ranked data)
# ═══════════════════════════════════════════════════════════════

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


def build_enhanced_ai_prompt(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any],
    rich_season_data: Dict[str, Any]
) -> str:
    """Legacy prompt for non-ranked flow."""
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

    top_foundations = foundation_matches[:3] if foundation_matches else []
    foundation_text = "\n".join([
        f"  • {match['brand']} - {match['shade']} (ΔE: {match.get('distance', 0):.2f})"
        for match in top_foundations
    ]) if top_foundations else "  • No matches found"

    lipstick_best = lipstick.get('best', [])
    blush_best = blush.get('best', [])

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

## OUTPUT FORMAT
Respond in Markdown with ONLY these two sections:

### 🌟 Your Skin & Season
Write 3-4 natural, personalized sentences. Name the final season once.
Explain what this means practically for the person's coloring.

### 💄 Foundation Match
Lead with the single best foundation match using its actual brand and shade.
Use the ΔE value to honestly describe the match quality. Do not mention
numeric scores directly — translate them into plain language.
"""

    return prompt


# ═══════════════════════════════════════════════════════════════
# TEMPLATE FALLBACK (when Groq is unavailable)
# ═══════════════════════════════════════════════════════════════

def _generate_template_response(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any]
) -> str:
    """Fallback template when Groq API is unavailable."""
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
        if delta_e < 1:
            closeness = "an excellent, nearly imperceptible match"
        elif delta_e < 2:
            closeness = "a very close match with minimal visual difference"
        elif delta_e < 3.5:
            closeness = "a good match, noticeable only up close"
        else:
            closeness = "a workable match — we recommend testing in person"
        response += (
            f"### 💄 Foundation Match\n\n"
            f"Your closest match is **{top_match['brand']} - {top_match['shade']}** "
            f"— {closeness}.\n"
        )

    return response


# ═══════════════════════════════════════════════════════════════
# LEGACY NON-RANKED ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def get_groq_recommendations(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any],
    season_key: str = None
) -> Dict[str, Any]:
    """
    Legacy non-ranked Groq call. Returns a dict with 'summary' and 'structured'.
    """
    if not GROQ_API_KEY:
        print("⚠️ GROQ_API_KEY not set. Using template fallback.")
        return {
            "summary": _generate_template_response(face_result, foundation_matches, season_data),
            "structured": {}
        }

    try:
        client = Groq(api_key=GROQ_API_KEY)
        rich_season_data = season_data.get('raw_data', {}) or {}

        if rich_season_data:
            prompt = build_enhanced_ai_prompt(
                face_result,
                foundation_matches,
                season_data,
                rich_season_data
            )
        else:
            prompt = build_basic_ai_prompt(face_result, foundation_matches, season_data)

        print("🤖 Calling Groq API (legacy)...")

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
        print("✅ Groq API response received (legacy)")

        return {
            "summary": result,
            "structured": {}
        }

    except Exception as e:
        print(f"⚠️ Groq API error (legacy): {e}")
        return {
            "summary": _generate_template_response(face_result, foundation_matches, season_data),
            "structured": {}
        }


# ═══════════════════════════════════════════════════════════════
# MAIN RANKED ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def get_ai_recommendations_ranked(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    ranked_seasons: Dict[str, Any],
    seasons_full_data: Dict[str, Dict[str, Any]],
    observations: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ranked-aware entry point using season scoring. Falls back to template
    if Groq is unavailable.
    """
    print("🔥🔥🔥 Groq Writer (Ranked) — Observations Included 🔥🔥🔥")

    winner = ranked_seasons["top_seasons"][0]
    winner_key = winner["season_key"]
    rich_season_data = seasons_full_data.get(winner_key, {}) or {}

    # ── Build structured data for frontend ──
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

    # ── If no API key, use template ──
    if not GROQ_API_KEY:
        print("⚠️ GROQ_API_KEY not set. Using template fallback.")
        fake_season_data = {"season_label": winner["season_label"]}
        return {
            "summary": _generate_template_response(face_result, foundation_matches, fake_season_data),
            "structured": structured,
            "ranked_seasons": ranked_seasons,
        }

    # ── Call Groq API ──
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = build_ranked_ai_prompt(
            face_result,
            foundation_matches,
            ranked_seasons,
            seasons_full_data,
            observations
        )

        has_obs = observations and observations.get('has_observations', False)
        print(f"🤖 Calling Groq API (ranked, close_call={ranked_seasons['is_close_call']}, has_obs={has_obs})...")

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


# ═══════════════════════════════════════════════════════════════
# LEGACY WRAPPER (for backward compatibility)
# ═══════════════════════════════════════════════════════════════

def get_ai_recommendations(
    face_result: Dict[str, Any],
    foundation_matches: List[Dict[str, Any]],
    season_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Legacy wrapper that mimics the old function signature.
    """
    season_key = season_data.get('season_key')
    result = get_groq_recommendations(face_result, foundation_matches, season_data, season_key)
    rich_season_data = season_data.get('raw_data', {}) or {}

    return {
        'summary': result.get('summary', ''),
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