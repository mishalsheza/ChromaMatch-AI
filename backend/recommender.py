"""
recommender.py — Cosmetic shade + palette recommendations.
CNN predicts only: depth + undertone.
Everything else is derived from these two values.
"""


def get_recommendations(
    depth: str,
    undertone: str,
    confidence: float = 0.85,
    eye_color: str = "Dark Brown",
    hair_color: str = "Black"
) -> dict:
    """
    Returns personalized recommendations based on CNN predictions.
    
    CNN Predicts:
        depth: Fair | Light | Medium | Medium Tan | Tan | Deep
        undertone: Warm | Neutral | Cool | Olive | Warm Olive
    
    Args:
        depth: CNN-predicted skin depth
        undertone: CNN-predicted undertone
        confidence: CNN confidence score (0-1)
        eye_color: User-provided (optional)
        hair_color: User-provided (optional)
    """

    # ──────────────────────────────────────────────────────────────────
    #  1. DEPTH MAPPING (CNN output → Display)
    # ──────────────────────────────────────────────────────────────────
    depth_display = {
        "Fair": "Fair",
        "Light": "Light", 
        "Medium": "Medium",
        "Medium Tan": "Medium Tan",
        "Tan": "Tan",
        "Deep": "Deep"
    }
    
    # ──────────────────────────────────────────────────────────────────
    #  2. STYLE ARCHETYPE
    # ──────────────────────────────────────────────────────────────────
    style_archetypes = {
        "Warm": "Golden Radiance",
        "Neutral": "Balanced Classic",
        "Cool": "Rose Elegance",
        "Olive": "Earthy Jewel Tone",
        "Warm Olive": "Golden Olive Luxe"
    }
    
    # ──────────────────────────────────────────────────────────────────
    #  3. FOUNDATION DATABASE (30 combinations only)
    # ──────────────────────────────────────────────────────────────────
    foundation_db = {
        # ── FAIR ──────────────────────────────────────────────────────
        ("Fair", "Warm"): {
            "loreal": "Light",
            "maybelline_eraser": "Fair",
            "maybelline_superstay": "115",
            "mac": "NC15",
            "lancome": "095W Ivoire",
        },
        ("Fair", "Neutral"): {
            "loreal": "Light",
            "maybelline_eraser": "Ivory",
            "maybelline_superstay": "123",
            "mac": "NC12",
            "lancome": "100N Ivoire Naturel",
        },
        ("Fair", "Cool"): {
            "loreal": "Rosy Light",
            "maybelline_eraser": "4.5",
            "maybelline_superstay": "120",
            "mac": "NW10",
            "lancome": "090N Ivoire",
        },
        ("Fair", "Olive"): {
            "loreal": "Light",
            "maybelline_eraser": "4.5",
            "maybelline_superstay": "120",
            "mac": "NC13",
            "lancome": "095W Ivoire",
        },
        ("Fair", "Warm Olive"): {
            "loreal": "Light",
            "maybelline_eraser": "Fair",
            "maybelline_superstay": "118",
            "mac": "NC15",
            "lancome": "095W Ivoire",
        },

        # ── LIGHT ─────────────────────────────────────────────────────
        ("Light", "Warm"): {
            "loreal": "Light Medium",
            "maybelline_eraser": "Light",
            "maybelline_superstay": "125",
            "mac": "NC20",
            "lancome": "120W Ivoire",
        },
        ("Light", "Neutral"): {
            "loreal": "Light Medium",
            "maybelline_eraser": "122 Sand",
            "maybelline_superstay": "128",
            "mac": "N4",
            "lancome": "130N Ivoire Naturel",
        },
        ("Light", "Cool"): {
            "loreal": "Light",
            "maybelline_eraser": "122 Sand",
            "maybelline_superstay": "128",
            "mac": "NW15",
            "lancome": "110C Ivoire",
        },
        ("Light", "Olive"): {
            "loreal": "Light Medium",
            "maybelline_eraser": "Light",
            "maybelline_superstay": "128",
            "mac": "NC20",
            "lancome": "120W Ivoire",
        },
        ("Light", "Warm Olive"): {
            "loreal": "Light Medium",
            "maybelline_eraser": "Light",
            "maybelline_superstay": "125",
            "mac": "NC22",
            "lancome": "120W Ivoire",
        },

        # ── MEDIUM ────────────────────────────────────────────────────
        ("Medium", "Warm"): {
            "loreal": "Medium",
            "maybelline_eraser": "Honey",
            "maybelline_superstay": "220",
            "mac": "NC30",
            "lancome": "230W Buff",
        },
        ("Medium", "Neutral"): {
            "loreal": "Medium",
            "maybelline_eraser": "122 Sand",
            "maybelline_superstay": "230",
            "mac": "N6",
            "lancome": "235N",
        },
        ("Medium", "Cool"): {
            "loreal": "Medium",
            "maybelline_eraser": "122 Sand",
            "maybelline_superstay": "228",
            "mac": "NW25",
            "lancome": "220C",
        },
        ("Medium", "Olive"): {
            "loreal": "Medium",
            "maybelline_eraser": "Honey",
            "maybelline_superstay": "230",
            "mac": "NC30",
            "lancome": "230W Buff",
        },
        ("Medium", "Warm Olive"): {
            "loreal": "Medium",
            "maybelline_eraser": "Honey",
            "maybelline_superstay": "230",
            "mac": "NC33",
            "lancome": "230W Buff",
        },

        # ── MEDIUM TAN ────────────────────────────────────────────────
        ("Medium Tan", "Warm"): {
            "loreal": "Medium Tan",
            "maybelline_eraser": "Honey",
            "maybelline_superstay": "310",
            "mac": "NC40",
            "lancome": "320W Bisque",
        },
        ("Medium Tan", "Neutral"): {
            "loreal": "Medium Tan",
            "maybelline_eraser": "Caramel",
            "maybelline_superstay": "310",
            "mac": "N7",
            "lancome": "305N",
        },
        ("Medium Tan", "Cool"): {
            "loreal": "Medium Tan",
            "maybelline_eraser": "Caramel",
            "maybelline_superstay": "311",
            "mac": "NW35",
            "lancome": "315C",
        },
        ("Medium Tan", "Olive"): {
            "loreal": "Medium Tan",
            "maybelline_eraser": "Honey",
            "maybelline_superstay": "310",
            "mac": "NC40",
            "lancome": "320W Bisque",
        },
        ("Medium Tan", "Warm Olive"): {
            "loreal": "Medium Tan",
            "maybelline_eraser": "Honey",
            "maybelline_superstay": "310",
            "mac": "NC42",
            "lancome": "320W Bisque",
        },

        # ── TAN ───────────────────────────────────────────────────────
        ("Tan", "Warm"): {
            "loreal": "Tan",
            "maybelline_eraser": "Caramel",
            "maybelline_superstay": "326",
            "mac": "NC43",
            "lancome": "320W Bisque",
        },
        ("Tan", "Neutral"): {
            "loreal": "Tan",
            "maybelline_eraser": "Caramel",
            "maybelline_superstay": "326",
            "mac": "N8",
            "lancome": "305N",
        },
        ("Tan", "Cool"): {
            "loreal": "Tan",
            "maybelline_eraser": "Caramel",
            "maybelline_superstay": "330",
            "mac": "NW40",
            "lancome": "315C",
        },
        ("Tan", "Olive"): {
            "loreal": "Tan",
            "maybelline_eraser": "Caramel",
            "maybelline_superstay": "326",
            "mac": "NC43",
            "lancome": "320W Bisque",
        },
        ("Tan", "Warm Olive"): {
            "loreal": "Tan",
            "maybelline_eraser": "Caramel",
            "maybelline_superstay": "326",
            "mac": "NC44",
            "lancome": "320W Bisque",
        },

        # ── DEEP ──────────────────────────────────────────────────────
        ("Deep", "Warm"): {
            "loreal": "Tan",
            "maybelline_eraser": "Butterscotch",
            "maybelline_superstay": "330",
            "mac": "NC45",
            "lancome": "400W",
        },
        ("Deep", "Neutral"): {
            "loreal": "Tan",
            "maybelline_eraser": "150 Neutralizer",
            "maybelline_superstay": "340",
            "mac": "N10",
            "lancome": "345N",
        },
        ("Deep", "Cool"): {
            "loreal": "Tan",
            "maybelline_eraser": "150 Neutralizer",
            "maybelline_superstay": "340",
            "mac": "NW43",
            "lancome": "430C",
        },
        ("Deep", "Olive"): {
            "loreal": "Tan",
            "maybelline_eraser": "Butterscotch",
            "maybelline_superstay": "340",
            "mac": "NC45",
            "lancome": "400W",
        },
        ("Deep", "Warm Olive"): {
            "loreal": "Tan",
            "maybelline_eraser": "Butterscotch",
            "maybelline_superstay": "340",
            "mac": "NC47",
            "lancome": "400W",
        },
    }

    # ──────────────────────────────────────────────────────────────────
    #  4. COLOR PALETTES (by undertone)
    # ──────────────────────────────────────────────────────────────────
    color_palettes = {
        "Warm": {
            "best": [
                {"name": "Terracotta", "hex": "#BC6C25"},
                {"name": "Mustard", "hex": "#E0A030"},
                {"name": "Olive Green", "hex": "#606C38"},
                {"name": "Teal", "hex": "#2A9D8F"},
                {"name": "Burnt Orange", "hex": "#CA6702"},
                {"name": "Cream", "hex": "#FEF6DC"},
            ],
            "avoid": ["Cool Gray", "Silver", "Fuchsia", "Ice Blue"],
            "season": "Warm Autumn / Warm Spring",
            "description": "Your skin carries rich golden and yellow undertones. Earthy oranges, warm spices, and organic earth tones amplify your natural glow."
        },
        "Neutral": {
            "best": [
                {"name": "Dusty Rose", "hex": "#C49090"},
                {"name": "Sage Green", "hex": "#8A9A86"},
                {"name": "Mauve", "hex": "#A06880"},
                {"name": "Taupe", "hex": "#B8A898"},
                {"name": "Rose Gold", "hex": "#D4A070"},
                {"name": "Cream", "hex": "#FEF6DC"},
            ],
            "avoid": ["NEON", "Extremely bright colors"],
            "season": "Universal / Neutral Spectrum",
            "description": "You have a highly versatile skin profile balancing warm and cool notes. Muted, desaturated tones work beautifully."
        },
        "Cool": {
            "best": [
                {"name": "Emerald Green", "hex": "#0E5245"},
                {"name": "Royal Sapphire", "hex": "#0F4C81"},
                {"name": "Magenta", "hex": "#A03048"},
                {"name": "Charcoal", "hex": "#2A3040"},
                {"name": "Plum", "hex": "#5C1A4A"},
                {"name": "Silver", "hex": "#C0C0C0"},
            ],
            "avoid": ["Orange", "Yellow", "Peach", "Gold"],
            "season": "Cool Winter / Cool Summer",
            "description": "Your skin features pink, rose, or blue under-layers. Jewel tones, deep berries, and icy silvers accentuate your features flawlessly."
        },
        "Olive": {
            "best": [
                {"name": "Burgundy", "hex": "#800020"},
                {"name": "Forest Green", "hex": "#1B4332"},
                {"name": "Navy", "hex": "#001F54"},
                {"name": "Deep Plum", "hex": "#5A189A"},
                {"name": "Teal", "hex": "#008080"},
                {"name": "Bronze", "hex": "#CD7F32"},
            ],
            "avoid": ["Pastel Yellow", "Neon Pink", "Orange", "Bright Pink"],
            "season": "Olive Spectrum",
            "description": "Your olive undertone carries a subtle greenish cast. Deep, rich jewel tones and earthy colors enhance your natural warmth."
        },
        "Warm Olive": {
            "best": [
                {"name": "Burgundy", "hex": "#800020"},
                {"name": "Deep Teal", "hex": "#004D4D"},
                {"name": "Navy", "hex": "#001F54"},
                {"name": "Deep Purple", "hex": "#4A0080"},
                {"name": "Gold", "hex": "#D4AF37"},
                {"name": "Olive Green", "hex": "#606C38"},
            ],
            "avoid": ["Pastels", "Orange", "Yellow", "Hot Pink"],
            "season": "Golden Olive Spectrum",
            "description": "Your olive undertone is enriched with golden warmth. You glow in earthy jewel tones, warm greens, and gold-accented colors."
        }
    }

    # ──────────────────────────────────────────────────────────────────
    #  5. HAIR RECOMMENDATIONS
    # ──────────────────────────────────────────────────────────────────
    hair_recommendations = {
        "Warm": [
            "Caramel Highlights",
            "Honey Brown",
            "Chocolate Brown",
            "Golden Brown",
            "Cinnamon"
        ],
        "Neutral": [
            "Mushroom Brown",
            "Mocha Brown",
            "Soft Ash Brown",
            "Chocolate",
            "Chestnut"
        ],
        "Cool": [
            "Ash Brown",
            "Burgundy",
            "Plum Brown",
            "Cool Chocolate",
            "Deep Mahogany"
        ],
        "Olive": [
            "Espresso",
            "Dark Chocolate",
            "Auburn",
            "Dark Cherry",
            "Deep Mahogany"
        ],
        "Warm Olive": [
            "Espresso",
            "Cinnamon",
            "Dark Cherry",
            "Warm Chocolate",
            "Mahogany"
        ]
    }

    # ──────────────────────────────────────────────────────────────────
    #  6. JEWELRY RECOMMENDATIONS
    # ──────────────────────────────────────────────────────────────────
    jewelry_recommendations = {
        "Warm": "Gold",
        "Neutral": "Gold & Silver",
        "Cool": "Silver",
        "Olive": "Gold & Bronze",
        "Warm Olive": "Gold"
    }

    # ──────────────────────────────────────────────────────────────────
    #  7. MAKEUP RECOMMENDATIONS
    # ──────────────────────────────────────────────────────────────────
    makeup_recommendations = {
        "Warm": {
            "blush": "Apricot / Peach",
            "lipstick": "Warm Brick / Terracotta",
            "eyeliner": "Brown",
            "foundation_finish": "Luminous / Dewy"
        },
        "Neutral": {
            "blush": "Rose / Dusty Pink",
            "lipstick": "Neutral Rose / Mauve",
            "eyeliner": "Brown or Charcoal",
            "foundation_finish": "Natural / Satin"
        },
        "Cool": {
            "blush": "Cool Pink / Soft Mauve",
            "lipstick": "Rose / Berry / Plum",
            "eyeliner": "Charcoal",
            "foundation_finish": "Matte / Natural"
        },
        "Olive": {
            "blush": "Warm Berry / Muted Rose",
            "lipstick": "Burgundy / Plum / Deep Berry",
            "eyeliner": "Deep Brown",
            "foundation_finish": "Natural / Satin"
        },
        "Warm Olive": {
            "blush": "Warm Berry / Apricot",
            "lipstick": "Warm Plum / Brick Red",
            "eyeliner": "Deep Brown",
            "foundation_finish": "Luminous / Natural"
        }
    }

    # ──────────────────────────────────────────────────────────────────
    #  8. DERIVED ATTRIBUTES
    # ──────────────────────────────────────────────────────────────────
    
    # Style Archetype
    archetype = style_archetypes.get(undertone, "Balanced Classic")
    
    # Color Palette
    palette = color_palettes.get(undertone, color_palettes["Neutral"])
    
    # Hair Colors
    hair_colors = hair_recommendations.get(undertone, hair_recommendations["Neutral"])
    
    # Jewelry
    jewelry = jewelry_recommendations.get(undertone, "Gold")
    
    # Makeup
    makeup = makeup_recommendations.get(undertone, makeup_recommendations["Neutral"])
    
    # Foundation Lookup
    foundation_key = (depth, undertone)
    
    if foundation_key in foundation_db:
        products = foundation_db[foundation_key]
    else:
        # Fallback to Neutral of same depth
        fallback_key = (depth, "Neutral")
        if fallback_key in foundation_db:
            products = foundation_db[fallback_key]
        else:
            products = foundation_db[("Medium", "Neutral")]

    

    # ──────────────────────────────────────────────────────────────────
    #  10. OUTFIT COMBINATIONS
    # ──────────────────────────────────────────────────────────────────
    # Best colors as outfit combos
    best_colors = palette["best"]
    outfit_combinations = []
    if len(best_colors) >= 4:
        outfit_combinations = [
            {"top": best_colors[0]["name"], "bottom": best_colors[1]["name"], "accessory": jewelry},
            {"top": best_colors[2]["name"], "bottom": best_colors[3]["name"], "accessory": jewelry},
            {"top": best_colors[0]["name"], "bottom": "Neutral", "accessory": jewelry},
        ]

    # ──────────────────────────────────────────────────────────────────
    #  RETURN PAYLOAD
    # ──────────────────────────────────────────────────────────────────
    return {
        "skin_profile": {
            "depth": depth_display.get(depth, depth),
            "undertone": undertone,
            "confidence": confidence,
            "style_archetype": archetype,
            "seasonal_profile": {
                "name": palette["season"],
                "description": palette["description"]
            }
        },
        "foundations": [
            {"brand": "L'Oréal Paris", "product": "Infallible 24H Tinted Serum Foundation", "shade": products["loreal"]},
            {"brand": "Maybelline", "product": "Age Rewind Eraser Concealer", "shade": products["maybelline_eraser"]},
            {"brand": "Maybelline", "product": "Super Stay Foundation", "shade": products["maybelline_superstay"]},
            {"brand": "MAC Cosmetics", "product": "Studio Fix Fluid SPF 15", "shade": products["mac"]},
            {"brand": "Lancôme", "product": "Teint Idole Ultra Wear Foundation", "shade": products["lancome"]},
        ],
        "clothing_palette": {
            "best_colors": palette["best"],
            "avoid_colors": palette["avoid"],
            "season": palette["season"],
            "description": palette["description"]
        },
        "hair_colors": hair_colors,
        "jewelry": jewelry,
        "makeup": makeup,
        "outfit_combinations": outfit_combinations,
        
    }