# test_gemini.py

from ai.gemini.analyzer import analyze_skin_undertone, load_all_examples
import json

# Load all examples from folders
examples = load_all_examples("ai/examples")

# Test on a new image
result = analyze_skin_undertone(
    face_image_path="/Users/shezamishal19/Desktop/shadeSense/ai/examples/light_cool/Screenshot 2026-06-16 at 13.36.25.png",  # Replace with actual image path
    few_shot_examples=examples,
)

print(json.dumps(result, indent=2))