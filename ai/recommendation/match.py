"""
match.py — Foundation shade matching using Delta-E 2000
"""

import math
import json
import os
from typing import List, Dict, Any


# ── CIEDE2000 COLOR DIFFERENCE ──

def delta_e_2000(
    lab1: tuple,
    lab2: tuple
) -> float:
    """
    Calculate CIEDE2000 color difference.
    """

    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Constants
    kL, kC, kH = 1.0, 1.0, 1.0

    # Step 1: Calculate C1, C2
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)

    C_bar = (C1 + C2) / 2.0

    # Step 2: Calculate G
    G = 0.5 * (
        1 - math.sqrt(
            C_bar**7 / (C_bar**7 + 25**7)
        )
    )

    # Step 3: Calculate a1', a2'
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    # Step 4: Calculate C1', C2', h1', h2'
    C1_prime = math.sqrt(
        a1_prime**2 + b1**2
    )

    C2_prime = math.sqrt(
        a2_prime**2 + b2**2
    )

    def calculate_h_prime(
        a_prime,
        b_val
    ):
        h = math.atan2(
            b_val,
            a_prime
        )

        if h < 0:
            h += 2 * math.pi

        return h

    h1_prime = calculate_h_prime(
        a1_prime,
        b1
    )

    h2_prime = calculate_h_prime(
        a2_prime,
        b2
    )

    # Step 5: Calculate ΔL', ΔC', ΔH'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    delta_h_prime = (
        h2_prime - h1_prime
    )

    if delta_h_prime > math.pi:
        delta_h_prime -= 2 * math.pi

    elif delta_h_prime < -math.pi:
        delta_h_prime += 2 * math.pi

    delta_H_prime = (
        2
        * math.sqrt(C1_prime * C2_prime)
        * math.sin(delta_h_prime / 2)
    )

    # Step 6: Calculate L', C', h' averages
    L_bar_prime = (
        L1 + L2
    ) / 2

    C_bar_prime = (
        C1_prime + C2_prime
    ) / 2

    h_bar_prime = (
        h1_prime + h2_prime
    )

    if abs(h1_prime - h2_prime) > math.pi:
        h_bar_prime += 2 * math.pi

    h_bar_prime /= 2

    # Step 7: Calculate T
    T = (
        1
        - 0.17 * math.cos(
            h_bar_prime - math.radians(30)
        )
        + 0.24 * math.cos(
            2 * h_bar_prime
        )
        + 0.20 * math.cos(
            3 * h_bar_prime
            + math.radians(6)
        )
        - 0.20 * math.cos(
            4 * h_bar_prime
            - math.radians(63)
        )
    )

    # Step 8: Calculate SL, SC, SH
    SL = (
        1
        + (
            0.015
            * (L_bar_prime - 50)**2
        )
        / math.sqrt(
            20 + (L_bar_prime - 50)**2
        )
    )

    SC = (
        1
        + 0.045 * C_bar_prime
    )

    SH = (
        1
        + 0.015
        * C_bar_prime
        * T
    )

    # Step 9: Calculate RT
    RT = (
        -2
        * math.sqrt(
            C_bar_prime**7
            / (C_bar_prime**7 + 25**7)
        )
        * math.sin(
            math.radians(60)
            * math.exp(
                -(
                    (
                        h_bar_prime
                        - math.radians(275)
                    )
                    / math.radians(25)
                )**2
            )
        )
    )

    # Step 10: Calculate Delta-E
    delta_E = math.sqrt(
        (delta_L_prime / (kL * SL))**2
        + (delta_C_prime / (kC * SC))**2
        + (delta_H_prime / (kH * SH))**2
        + RT
        * (
            delta_C_prime
            / (kC * SC)
        )
        * (
            delta_H_prime
            / (kH * SH)
        )
    )

    return delta_E


# ── HEX TO LAB ──

def hex_to_lab(
    hex_color: str
) -> tuple:
    """
    Convert hex color to LAB.
    """

    if hex_color.startswith("#"):
        hex_color = hex_color[1:]

    if len(hex_color) == 3:
        hex_color = "".join(
            [c * 2 for c in hex_color]
        )

    try:
        r = int(
            hex_color[0:2],
            16
        ) / 255.0

        g = int(
            hex_color[2:4],
            16
        ) / 255.0

        b = int(
            hex_color[4:6],
            16
        ) / 255.0

    except ValueError:
        return (
            50.0,
            0.0,
            0.0
        )

    # RGB to LAB

    def f(t):
        if t > 0.008856:
            return t ** (1 / 3)

        return (
            7.787 * t
            + 16 / 116
        )

    r = (
        r ** 2.4
        if r > 0.04045
        else r / 12.92
    )

    g = (
        g ** 2.4
        if g > 0.04045
        else g / 12.92
    )

    b = (
        b ** 2.4
        if b > 0.04045
        else b / 12.92
    )

    x = (
        r * 0.4124564
        + g * 0.3575761
        + b * 0.1804375
    )

    y = (
        r * 0.2126729
        + g * 0.7151522
        + b * 0.0721750
    )

    z = (
        r * 0.0193339
        + g * 0.1191920
        + b * 0.9503041
    )

    x_n = 0.95047
    y_n = 1.0
    z_n = 1.08883

    L = (
        116 * f(y / y_n)
        - 16
    )

    a = 500 * (
        f(x / x_n)
        - f(y / y_n)
    )

    b_lab = 200 * (
        f(y / y_n)
        - f(z / z_n)
    )

    return (
        L,
        a,
        b_lab
    )


# ── LOAD FOUNDATION DATABASE ──

def load_foundation_db(
    data_path: str = None
) -> List[Dict[str, Any]]:
    """
    Load foundation database with LAB values.
    """

    # If no path is provided,
    # try multiple locations.
    if data_path is None:

        possible_paths = [
            "ai/data/foundations.json",
            "../ai/data/foundations.json",

            os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(__file__)
                    )
                ),
                "ai/data/foundations.json"
            ),

            os.path.join(
                os.getcwd(),
                "ai/data/foundations.json"
            ),

            "/Users/shezamishal19/Desktop/"
            "ShadeSense/ai/data/foundations.json"
        ]

    else:
        possible_paths = [
            data_path
        ]

    for path in possible_paths:

        if os.path.exists(path):

            print(
                f"📂 Loading foundations from: {path}"
            )

            with open(path, "r") as f:
                data = json.load(f)

            # Add LAB values for each foundation
            for item in data:

                if "hex" in item and item["hex"]:

                    try:
                        item["lab"] = hex_to_lab(
                            item["hex"]
                        )

                    except Exception:
                        item["lab"] = (
                            50.0,
                            0.0,
                            0.0
                        )

            return data

    print(
        f"❌ Foundation database not found at: "
        f"{data_path}"
    )

    return []


# ── UNDERTONE FROM LAB ──

def get_undertone_from_lab(
    lab: tuple
) -> str:
    """
    Classify undertone from LAB values.
    """

    L, a, b = lab

    if b > a + 5:
        return "warm"

    elif a > b + 5:
        return "cool"

    else:
        return "neutral"


# ── FOUNDATION MATCHING ──

def match_foundations(
    skin_lab: tuple,
    foundations: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Find best foundation matches
    using Delta-E 2000.
    """

    if not foundations:
        return []

    matches = []

    for foundation in foundations:

        if "lab" not in foundation:
            continue

        try:
            distance = delta_e_2000(
                skin_lab,
                foundation["lab"]
            )

        except Exception:
            continue

        matches.append({
            **foundation,
            "distance": distance,
            "undertone": get_undertone_from_lab(
                foundation["lab"]
            )
        })

    matches.sort(
        key=lambda x: x["distance"]
    )

    return matches[:top_k]