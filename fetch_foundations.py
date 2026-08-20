"""fetch_foundations.py — Fetch foundation data from Makeup API"""

import requests
import json
import os


def fetch_foundations():
    """Fetch foundation products from the Makeup API"""
    print("📡 Fetching foundation data from Makeup API...")

    url = "https://makeup-api.herokuapp.com/api/v1/products.json"
    params = {
        "product_type": "foundation",
        "limit": 100
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        print(f"✅ Fetched {len(data)} foundation products")
        return data

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


def convert_to_shade_db(api_data):
    """Convert API data to our foundation database format"""
    foundations = []

    for product in api_data:
        brand = product.get("brand", "Unknown")
        product_name = product.get("name", "Unknown")

        # Check if product has color swatches
        if "product_colors" in product and product["product_colors"]:
            for color in product["product_colors"]:
                hex_val = color.get("hex_value", "")
                shade_name = color.get("colour_name", "Unknown")

                if hex_val and len(hex_val) >= 6:
                    foundations.append({
                        "brand": brand,
                        "product": product_name,
                        "shade": shade_name,
                        "hex": hex_val,
                        "price": product.get("price", ""),
                        "product_link": product.get("product_link", ""),
                        "image_link": product.get("api_featured_image", "")
                    })

    print(f"✅ Converted {len(foundations)} foundation shades")
    return foundations


def save_foundation_db(
    foundations,
    path="ai/data/foundations.json"
):
    """Save foundation database to JSON file"""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(foundations, f, indent=2)

    print(f"✅ Saved {len(foundations)} shades to {path}")
    return path


def load_foundation_db(
    path="ai/data/foundations.json"
):
    """Load foundation database from JSON file"""

    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    return None


if __name__ == "__main__":

    print("=" * 50)
    print("🛠️ Foundation Database Builder")
    print("=" * 50)

    # Check if we already have a database
    existing = load_foundation_db()

    if existing:
        print(
            f"📂 Found existing database with "
            f"{len(existing)} shades"
        )

        choice = input(
            "Do you want to rebuild? (y/n): "
        ).strip().lower()

        if choice != "y":
            print("✅ Using existing database")
            exit()

    # Fetch from API
    api_data = fetch_foundations()

    if not api_data:
        print(
            "❌ Could not fetch data. "
            "Please check your internet connection."
        )
        exit()

    # Convert to our format
    foundations = convert_to_shade_db(api_data)

    # Save
    save_foundation_db(foundations)

    print("\n📊 Sample shades:")

    for i, f in enumerate(foundations[:5], 1):
        print(
            f"  {i}. {f['brand']} - "
            f"{f['product']} - "
            f"{f['shade']} ({f['hex']})"
        )