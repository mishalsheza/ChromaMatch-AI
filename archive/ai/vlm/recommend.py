import json
import os
from typing import List, Dict, Any, Optional

# Load color palettes
def load_palettes():
    """Load the color palette dataset"""
    palette_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'color_palettes.json'
    )
    
    if os.path.exists(palette_path):
        with open(palette_path, 'r') as f:
            return json.load(f)
    else:
        # Return default palettes if file doesn't exist
        return [
            {
                "undertone": "warm",
                "complexion": "light",
                "colors": ["#F5C5A3", "#E8A87C", "#D4956B"],
                "category": "Spring",
                "color_names": ["Peach", "Coral", "Terracotta"],
                "description": "Warm, light complexion colors"
            },
            {
                "undertone": "cool",
                "complexion": "light",
                "colors": ["#D4C5D9", "#C4B5C9", "#B4A5B9"],
                "category": "Summer",
                "color_names": ["Lavender", "Mauve", "Dusty Rose"],
                "description": "Cool, light complexion colors"
            },
            {
                "undertone": "warm",
                "complexion": "medium",
                "colors": ["#E8B89D", "#D4A084", "#C08874"],
                "category": "Autumn",
                "color_names": ["Apricot", "Copper", "Bronze"],
                "description": "Warm, medium complexion colors"
            },
            {
                "undertone": "cool",
                "complexion": "medium",
                "colors": ["#C4B5C9", "#B4A5B9", "#A495A9"],
                "category": "Winter",
                "color_names": ["Plum", "Burgundy", "Mulberry"],
                "description": "Cool, medium complexion colors"
            }
        ]

_PALETTES = None

def get_palettes():
    """Get the loaded palettes (cached)"""
    global _PALETTES
    if _PALETTES is None:
        _PALETTES = load_palettes()
    return _PALETTES

def get_recommendations(undertone: str, complexion: str) -> List[Dict[str, Any]]:
    """
    Get color recommendations based on undertone and complexion
    
    Args:
        undertone: "warm", "cool", "neutral", or "olive"
        complexion: "very_light", "light", "medium", "tan", or "deep"
    
    Returns:
        List of recommended palettes
    """
    palettes = get_palettes()
    
    # Filter by undertone and complexion
    recommendations = []
    for palette in palettes:
        palette_undertone = palette.get('undertone', '').lower()
        palette_complexion = palette.get('complexion', '').lower()
        
        if palette_undertone == undertone.lower():
            if palette_complexion == complexion.lower():
                recommendations.append(palette)
    
    # If no exact match, try just undertone
    if not recommendations:
        for palette in palettes:
            palette_undertone = palette.get('undertone', '').lower()
            if palette_undertone == undertone.lower():
                recommendations.append(palette)
    
    # If still no match, return all
    if not recommendations:
        recommendations = palettes[:4]
    
    return recommendations

def generate_recommendation_text(undertone: str, complexion: str) -> str:
    """
    Generate a text description of recommendations
    
    Args:
        undertone: Skin undertone
        complexion: Skin complexion
    
    Returns:
        Formatted text with recommendations
    """
    recommendations = get_recommendations(undertone, complexion)
    
    if not recommendations:
        return f"No specific recommendations found for {undertone} skin with {complexion} complexion."
    
    text = f"🎨 Recommendations for {undertone} skin with {complexion} complexion:\n\n"
    
    for i, rec in enumerate(recommendations[:5], 1):
        category = rec.get('category', 'General')
        colors = rec.get('colors', [])
        color_names = rec.get('color_names', [])
        description = rec.get('description', '')
        
        # Handle colors properly - they should be strings
        if colors and isinstance(colors[0], dict):
            # Extract hex codes from dicts
            color_hexes = [c.get('hex', c.get('color', str(c))) for c in colors if isinstance(c, dict)]
        else:
            # Colors are already strings
            color_hexes = [str(c) for c in colors if c]
        
        text += f"{i}. {category}:\n"
        if color_hexes:
            text += f"   Colors: {', '.join(color_hexes)}\n"
        if color_names:
            text += f"   Color Names: {', '.join(color_names)}\n"
        if description:
            text += f"   Description: {description}\n"
        text += "\n"
    
    # Add a summary of color recommendations
    text += "💡 Tips for using these colors:\n"
    text += "   • Choose lighter shades for a natural look\n"
    text += "   • Use darker shades for a bold statement\n"
    text += "   • Mix and match within the same color family\n"
    text += "   • Consider the occasion when selecting intensity\n"
    
    return text

# Keep backward compatibility
def generate_recommendation_text_old(undertone: str, complexion: str) -> str:
    """Legacy function - calls the new one"""
    return generate_recommendation_text(undertone, complexion)
