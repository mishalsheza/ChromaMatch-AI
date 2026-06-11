"""
recommender.py — Cosmetic shade + palette recommendations.
"""


def get_recommendations(cosmetic_depth: str, undertone: str) -> dict:
    """
    Returns foundation/concealer shades based on skin depth + undertone.
    
    Depth: Fair | Light | Medium | Tan | Deep
    Undertone: Warm | Cool | Neutral
    """

    # ──────────────────────────────────────────────────────────────────
    #  FOUNDATION DATABASE - REAL SHADE NAMES FROM YOUR SCREENSHOTS
    # ──────────────────────────────────────────────────────────────────
    #
    #  Products:
    #   • L'Oréal Paris    → Infallible 24H Tinted Serum (7 shades)
    #   • Maybelline       → Age Rewind Eraser Concealer
    #   • Maybelline       → Super Stay Foundation
    #   • MAC Cosmetics    → Studio Fix Fluid SPF 15
    #   • Lancôme          → Teint Idole Ultra Wear Foundation
    #

    product_db = {
        # ── FAIR ──────────────────────────────────────────────────────
        "Fair_Warm": {
            "loreal":           "Very Light",
            "maybelline_eraser": "Fair",
            "maybelline_superstay": "115",
            "mac":              "NC15",
            "lancome":          "095W Ivoire",
        },
        "Fair_Cool": {
            "loreal":           "Very Light",
            "maybelline_eraser": "4.5",
            "maybelline_superstay": "120",
            "mac":              "NW10",
            "lancome":          "090N Ivoire",
        },
        "Fair_Neutral": {
            "loreal":           "Very Light",
            "maybelline_eraser": "Ivory",
            "maybelline_superstay": "123",
            "mac":              "NC12",
            "lancome":          "100N Ivoire Naturel",
        },

        # ── LIGHT ─────────────────────────────────────────────────────
        "Light_Warm": {
            "loreal":           "Light",
            "maybelline_eraser": "Light",
            "maybelline_superstay": "125",
            "mac":              "NC20",
            "lancome":          "120W Ivoire",
        },
        "Light_Cool": {
            "loreal":           "Rosy Light",
            "maybelline_eraser": "122 Sand",
            "maybelline_superstay": "128",
            "mac":              "NW15",
            "lancome":          "110C Ivoire",
        },
        "Light_Neutral": {
            "loreal":           "Light",
            "maybelline_eraser": "Light",
            "maybelline_superstay": "140",
            "mac":              "NC17",
            "lancome":          "130N Ivoire Naturel",
        },

        # ── MEDIUM ────────────────────────────────────────────────────
        "Medium_Warm": {
            "loreal":           "Light Medium",
            "maybelline_eraser": "Medium",
            "maybelline_superstay": "220",
            "mac":              "NC30",
            "lancome":          "230W Buff",
        },
        "Medium_Cool": {
            "loreal":           "Medium",
            "maybelline_eraser": "122 Sand",
            "maybelline_superstay": "228",
            "mac":              "NW25",
            "lancome":          "220C",
        },
        "Medium_Neutral": {
            "loreal":           "Medium",
            "maybelline_eraser": "Medium",
            "maybelline_superstay": "230",
            "mac":              "NC27",
            "lancome":          "235N",
        },

        # ── TAN ───────────────────────────────────────────────────────
        "Tan_Warm": {
            "loreal":           "Medium Tan",
            "maybelline_eraser": "Honey",
            "maybelline_superstay": "310",
            "mac":              "NC40",
            "lancome":          "320W Bisque",
        },
        "Tan_Cool": {
            "loreal":           "Tan",
            "maybelline_eraser": "Caramel",
            "maybelline_superstay": "311",
            "mac":              "NW35",
            "lancome":          "315C",
        },
        "Tan_Neutral": {
            "loreal":           "Medium Tan",
            "maybelline_eraser": "Caramel",
            "maybelline_superstay": "326",
            "mac":              "NC41",
            "lancome":          "305N",
        },

        # ── DEEP ──────────────────────────────────────────────────────
        "Deep_Warm": {
            "loreal":           "Tan",
            "maybelline_eraser": "Butterscotch",
            "maybelline_superstay": "330",
            "mac":              "NC45",
            "lancome":          "400W",
        },
        "Deep_Cool": {
            "loreal":           "Tan",
            "maybelline_eraser": "Butterscotch",
            "maybelline_superstay": "340",
            "mac":              "NW43",
            "lancome":          "430C",
        },
        "Deep_Neutral": {
            "loreal":           "Tan",
            "maybelline_eraser": "150 Neutralizer",
            "maybelline_superstay": "250",
            "mac":              "NC46",
            "lancome":          "345N",
        },
    }

    # ──────────────────────────────────────────────────────────────────
    #  LOOKUP LOGIC
    # ──────────────────────────────────────────────────────────────────
    key = f"{cosmetic_depth}_{undertone}"
    
    # Fallback if exact match not found
    if key not in product_db:
        key = f"{cosmetic_depth}_Neutral"
    if key not in product_db:
        key = "Medium_Neutral"
    
    products = product_db[key]

    # ──────────────────────────────────────────────────────────────────
    #  MAKEUP + CLOTHING PALETTES
    # ──────────────────────────────────────────────────────────────────

    if undertone == "Warm":
        makeup_palette = {
            "lipstick": [
                {"name": "Peach Nude", "hex": "#E8A898"},
                {"name": "Warm Terracotta", "hex": "#C05030"},
                {"name": "Spiced Brick", "hex": "#9E4733"},
            ],
            "blush": [
                {"name": "Warm Apricot", "hex": "#F4A261"},
                {"name": "Peachy Coral", "hex": "#E07050"},
            ],
            "eyeshadow": [
                {"name": "Champagne Gold", "hex": "#E9D8A6"},
                {"name": "Burnished Bronze", "hex": "#CA6702"},
                {"name": "Copper Glow", "hex": "#A85030"},
            ],
        }
        clothing_palette = [
            {"name": "Mustard Gold", "hex": "#E0A030", "description": "Illuminates golden undertones beautifully."},
            {"name": "Olive Green", "hex": "#606C38", "description": "Grounding and highly complementary."},
            {"name": "Burnt Terracotta", "hex": "#BC6C25", "description": "Draws out natural warmth."},
            {"name": "Teal Blue", "hex": "#2A9D8F", "description": "The ideal contrasting jewel tone."},
        ]
        seasonal_profile = {
            "name": "Warm Autumn / Warm Spring",
            "description": "Your skin carries rich golden and yellow undertones. Earthy oranges, warm spices, and organic earth tones amplify your natural glow.",
        }

    elif undertone == "Cool":
        makeup_palette = {
            "lipstick": [
                {"name": "Cool Rosewood", "hex": "#B76E79"},
                {"name": "Berry Pink", "hex": "#C83060"},
                {"name": "Deep Plum", "hex": "#5C1A4A"},
            ],
            "blush": [
                {"name": "Soft Mauve", "hex": "#C49090"},
                {"name": "Cool Rose", "hex": "#D06070"},
            ],
            "eyeshadow": [
                {"name": "Silver Shimmer", "hex": "#D8D8E0"},
                {"name": "Taupe Gray", "hex": "#8090A0"},
                {"name": "Amethyst", "hex": "#705880"},
            ],
        }
        clothing_palette = [
            {"name": "Emerald Green", "hex": "#0E5245", "description": "Stunning contrast against pink undertones."},
            {"name": "Royal Sapphire", "hex": "#0F4C81", "description": "Highlights cool blue coordinates."},
            {"name": "Magenta Crush", "hex": "#A03048", "description": "Dramatically pops against rosy undertones."},
            {"name": "Charcoal", "hex": "#2A3040", "description": "Elegant, crisp, and contrast-rich."},
        ]
        seasonal_profile = {
            "name": "Cool Winter / Cool Summer",
            "description": "Your skin features pink, rose, or blue under-layers. Jewel tones, deep berries, and icy silvers accentuate your features flawlessly.",
        }

    else:  # Neutral
        makeup_palette = {
            "lipstick": [
                {"name": "Dusty Rose Nude", "hex": "#C89090"},
                {"name": "Soft Caramel", "hex": "#C48050"},
                {"name": "Mauve Berry", "hex": "#A06880"},
            ],
            "blush": [
                {"name": "Neutral Rose", "hex": "#D09078"},
                {"name": "Soft Coral-Pink", "hex": "#E8A090"},
            ],
            "eyeshadow": [
                {"name": "Rose Gold", "hex": "#D4A070"},
                {"name": "Soft Taupe", "hex": "#A8A090"},
                {"name": "Warm Mushroom", "hex": "#908878"},
            ],
        }
        clothing_palette = [
            {"name": "Dusty Sage", "hex": "#8A9A86", "description": "Understated elegance highlighting neutral tones."},
            {"name": "Taupe Sand", "hex": "#D0C8C0", "description": "Seamlessly blends with a versatile palette."},
            {"name": "Dusty Rose Gold", "hex": "#B87080", "description": "Flatters the balanced warmth/coolness."},
            {"name": "Slate Blue", "hex": "#5A6878", "description": "Soft contrast without overwhelming."},
        ]
        seasonal_profile = {
            "name": "Universal / Neutral Spectrum",
            "description": "You have a highly versatile skin profile balancing warm and cool notes. Muted, desaturated tones work beautifully.",
        }

    # ──────────────────────────────────────────────────────────────────
    #  RETURN PAYLOAD
    # ──────────────────────────────────────────────────────────────────
    return {
        "foundations": [
            {
                "brand":   "L'Oréal Paris",
                "product": "Infallible 24H Tinted Serum Foundation",
                "shade":   products["loreal"],
            },
            {
                "brand":   "Maybelline",
                "product": "Age Rewind Eraser Concealer",
                "shade":   products["maybelline_eraser"],
            },
            {
                "brand":   "Maybelline",
                "product": "Super Stay Foundation",
                "shade":   products["maybelline_superstay"],
            },
            {
                "brand":   "MAC Cosmetics",
                "product": "Studio Fix Fluid SPF 15",
                "shade":   products["mac"],
            },
            {
                "brand":   "Lancôme",
                "product": "Teint Idole Ultra Wear Foundation",
                "shade":   products["lancome"],
            },
        ],
        "makeup_swatches": makeup_palette,
        "clothing_palette": clothing_palette,
        "seasonal_profile": seasonal_profile,
    }