"""
skin_observations.py — Deterministic skin issue detection
Detects pigmentation, dark circles, redness from regional LAB values.
All outputs are color-space observations, NOT diagnoses.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union

# ── THRESHOLDS ──
UNDER_EYE_THRESHOLDS = {
    'pronounced': 6.0,
    'mild': 3.0,
    'slight': 1.5,
}

PERIORAL_THRESHOLDS = {
    'pronounced': 4.0,
    'mild': 2.0,
}

REDNESS_THRESHOLD = 4.0
REDNESS_MILD_THRESHOLD = 2.0


def _get_lab_values(region_data: Any) -> Optional[List[float]]:
    """
    Extract [L, a, b] from various possible formats.
    Handles: list, dict with L/a/b keys, dict with 0/1/2 keys.
    """
    if region_data is None:
        return None
    
    # If it's already a list/tuple of 3 values
    if isinstance(region_data, (list, tuple)) and len(region_data) >= 3:
        return [float(region_data[0]), float(region_data[1]), float(region_data[2])]
    
    # If it's a dict with L, a, b keys
    if isinstance(region_data, dict):
        if 'L' in region_data and 'a' in region_data and 'b' in region_data:
            return [float(region_data['L']), float(region_data['a']), float(region_data['b'])]
        # If it's a dict with 0, 1, 2 keys
        if 0 in region_data and 1 in region_data and 2 in region_data:
            return [float(region_data[0]), float(region_data[1]), float(region_data[2])]
        # If it's a dict with 'lab' key
        if 'lab' in region_data and isinstance(region_data['lab'], (list, tuple)):
            return [float(region_data['lab'][0]), float(region_data['lab'][1]), float(region_data['lab'][2])]
    
    # If it's a numpy array or similar
    try:
        if len(region_data) >= 3:
            return [float(region_data[0]), float(region_data[1]), float(region_data[2])]
    except:
        pass
    
    return None


def detect_skin_observations(regions_lab: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect skin issues from regional LAB values.
    """
    observations = {
        'under_eye': None,
        'perioral': None,
        'redness': None,
        'tone_evenness': None,
        'has_observations': False
    }
    
    print(f"🔍 detect_skin_observations received {len(regions_lab)} regions: {list(regions_lab.keys())}")
    
    # ── Debug: Print first few regions ──
    for name in list(regions_lab.keys())[:3]:
        print(f"   {name}: {type(regions_lab[name])} = {regions_lab[name]}")
    
    # ── Extract LAB values for each region ──
    lab_values = {}
    for name, data in regions_lab.items():
        lab = _get_lab_values(data)
        if lab:
            lab_values[name] = lab
            print(f"✅ Extracted {name}: L={lab[0]:.1f}, a={lab[1]:.1f}, b={lab[2]:.1f}")
        else:
            print(f"⚠️ Could not extract LAB from {name}: {type(data)} = {data}")
    
    # ── Ensure we have required regions ──
    required = ['left_cheek', 'right_cheek', 'forehead', 'jaw']
    for req in required:
        if req not in lab_values:
            # Try alternative names
            if req == 'jaw' and 'jaw_chin' in lab_values:
                lab_values['jaw'] = lab_values['jaw_chin']
            else:
                print(f"⚠️ Missing required region: {req}")
                return observations
    
    # ── Calculate baselines ──
    baseline_L = np.mean([lab_values['left_cheek'][0], lab_values['right_cheek'][0]])
    baseline_a = np.mean([lab_values['left_cheek'][1], lab_values['right_cheek'][1]])
    baseline_b = np.mean([lab_values['left_cheek'][2], lab_values['right_cheek'][2]])
    
    print(f"📊 Baseline LAB: L={baseline_L:.1f}, a={baseline_a:.1f}, b={baseline_b:.1f}")
    
    # ── 1. UNDER-EYE DARKNESS ──
    if 'left_under_eye' in lab_values and 'right_under_eye' in lab_values:
        under_eye_L = np.mean([lab_values['left_under_eye'][0], lab_values['right_under_eye'][0]])
        under_eye_a = np.mean([lab_values['left_under_eye'][1], lab_values['right_under_eye'][1]])
        under_eye_b = np.mean([lab_values['left_under_eye'][2], lab_values['right_under_eye'][2]])
        
        L_delta = baseline_L - under_eye_L
        
        print(f"👁️ Under-eye L_delta: {L_delta:.1f}")
        
        if L_delta >= UNDER_EYE_THRESHOLDS['pronounced']:
            severity = 'pronounced'
        elif L_delta >= UNDER_EYE_THRESHOLDS['mild']:
            severity = 'mild'
        elif L_delta >= UNDER_EYE_THRESHOLDS['slight']:
            severity = 'slight'
        else:
            severity = None
        
        if severity:
            a_delta = under_eye_a - baseline_a
            b_delta = under_eye_b - baseline_b
            
            if a_delta > 2 and b_delta < -1:
                cast = 'blue-purple'
            elif a_delta > 3 and b_delta > 3:
                cast = 'brown'
            else:
                cast = 'neutral'
            
            observations['under_eye'] = {'severity': severity, 'cast': cast}
            observations['has_observations'] = True
            print(f"✅ Under-eye: {severity}, {cast} cast")
    
    # ── 2. PERIORAL PIGMENTATION ──
    if 'left_perioral' in lab_values and 'right_perioral' in lab_values and 'jaw' in lab_values:
        perioral_L = np.mean([lab_values['left_perioral'][0], lab_values['right_perioral'][0]])
        jaw_L = lab_values['jaw'][0]
        
        perioral_delta = jaw_L - perioral_L
        print(f"👄 Perioral delta: {perioral_delta:.1f}")
        
        if perioral_delta >= PERIORAL_THRESHOLDS['pronounced']:
            severity = 'pronounced'
        elif perioral_delta >= PERIORAL_THRESHOLDS['mild']:
            severity = 'mild'
        else:
            severity = None
        
        if severity:
            observations['perioral'] = {'severity': severity}
            observations['has_observations'] = True
            print(f"✅ Perioral: {severity}")
    
    # ── 3. CHEEK REDNESS ──
    cheek_a = np.mean([lab_values['left_cheek'][1], lab_values['right_cheek'][1]])
    forehead_a = lab_values['forehead'][1]
    redness = cheek_a - forehead_a
    
    print(f"🔴 Redness delta: {redness:.1f}")
    
    if redness > REDNESS_THRESHOLD:
        severity = 'pronounced'
    elif redness > REDNESS_MILD_THRESHOLD:
        severity = 'mild'
    else:
        severity = None
    
    if severity:
        observations['redness'] = {'severity': severity}
        observations['has_observations'] = True
        print(f"✅ Redness: {severity}")
    
    # ── 4. TONE EVENNESS ──
    L_values = [v[0] for v in lab_values.values() if len(v) >= 1]
    if len(L_values) >= 4:
        L_std = np.std(L_values)
        print(f"📊 L* std: {L_std:.1f}")
        if L_std > 5.0:
            observations['tone_evenness'] = 'uneven'
            observations['has_observations'] = True
        else:
            observations['tone_evenness'] = 'even'
    
    print(f"🔍 Final observations: {observations}")
    return observations


def build_observations_text(observations: Dict[str, Any]) -> str:
    """Convert observations to prompt-friendly text."""
    if not observations or not observations.get('has_observations', False):
        return "No notable observations beyond the primary skin analysis."

    lines = []
    guidance_blocks = []

    # ── Under-eye ──
    under_eye = observations.get('under_eye')
    # Change this line to include 'slight':
    if under_eye and under_eye.get('severity') in ('slight', 'mild', 'pronounced'):  # ← Added 'slight'
        cast = under_eye.get('cast', 'neutral')
        lines.append(f"- Under-eye area: {under_eye['severity']} darkness relative to cheek baseline, {cast} cast")
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
    # ... rest of function

    # ── Perioral ──
    perioral = observations.get('perioral')
    if perioral and perioral.get('severity') in ('mild', 'pronounced'):
        lines.append(f"- Perioral area: {perioral['severity']} discoloration relative to jaw baseline")
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

    # ── Tone evenness ──
    if observations.get('tone_evenness') == 'uneven':
        lines.append("- Skin tone: some variation across regions (common in natural skin)")

    if not lines:
        return "No notable observations beyond the primary skin analysis."

    observation_summary = "\n".join(lines)

    if guidance_blocks:
        guidance_summary = "\n".join(f"- {g}" for g in guidance_blocks)
        return f"{observation_summary}\n\nRelevant guidance:\n{guidance_summary}"

    return observation_summary