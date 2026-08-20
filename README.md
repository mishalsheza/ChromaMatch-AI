# ShadeSense

ShadeSense is a full-stack, AI-powered skin color analysis and foundation matching application. It extracts multi-region facial color data from user images, deterministically computes 12-season color profiles using CIELAB colorimetry, finds optimal foundation matches via CIEDE2000 calculations, and uses a Groq-powered LLM to deliver a highly personalized narrative interpretation of the results.

## ✨ Features

- **Facial Skin Color Extraction**: Accurate skin region isolation.
- **MediaPipe Face Landmarker**: Used for robust landmark detection, with a deterministic Haar Cascade fallback.
- **Multi-Region Skin Sampling**: Samples left cheek, right cheek, forehead, and jaw/chin while filtering out non-skin pixels and lighting artifacts.
- **CIELAB Color Analysis**: Operates in the perceptually uniform LAB color space.
- **Skin Attributes**: Calculates depth (via Individual Typology Angle), undertone (warm/cool/neutral/olive), and contrast.
- **12-Season Color Analysis**: Deterministically assigns one of 12 seasonal palettes using hard-coded rules and rich JSON reference data.
- **Ranked Season Candidates**: Provides top candidates and detects ambiguous/close-call margins.
- **Foundation Shade Matching**: Matches against a database of 1,303 shades using CIEDE2000 ($\Delta E_{00}$) for precision.
- **Groq AI Narrative**: Enhances the deterministic output with a natural language summary, interpreting the skin profile and foundation matches.
- **Deterministic Fallback**: Provides a hard-coded template response if the Groq API is unavailable.
- **Try-On Preview**: Experimental endpoint to visualize foundation shades.

## 🧠 How It Works

The system operates through a structured pipeline that strictly separates deterministic computer vision/color science from the generative AI layer:

1. **User Image Upload** $\rightarrow$ `backend/app.py`
2. **Face Detection & Landmarking** $\rightarrow$ `ai/colorimetry/face_color.py` (MediaPipe Tasks API / Haar Cascade)
3. **Facial Skin-Region Extraction** $\rightarrow$ `ai/colorimetry/face_color.py`
4. **Pixel Filtering & Averaging** $\rightarrow$ Outlier rejection and RGB to LAB conversion.
5. **Skin Attribute Derivation** $\rightarrow$ Calculates ITA, contrast, and undertone.
6. **12-Season Classification** $\rightarrow$ `ai/recommendation/season.py` (Rule-based matching against `season_color_reference.json`)
7. **Ranked Candidate Scoring** $\rightarrow$ Evaluates season margins and close-call flags.
8. **Foundation Matching** $\rightarrow$ `ai/recommendation/match.py` (CIEDE2000 against 1,303 database records)
9. **Groq Interpretation** $\rightarrow$ `ai/recommendation/groq_writer.py` (Generates short personalized narrative)
10. **Structured JSON Response** $\rightarrow$ Sent back via Flask REST API.
11. **Frontend Rendering** $\rightarrow$ `ShadeSense.html` renders color palettes, UI, and text.

## 🎨 Skin Analysis

### Face Detection
The application natively targets the MediaPipe Tasks API (`face_landmarker.task`) for robust 3D facial landmark detection. If MediaPipe fails or is unavailable, the pipeline gracefully falls back to a deterministic Haar Cascade classifier (`haarcascade_frontalface_default.xml` via `face_detector_v2.py`).

### Regional Skin Sampling
Skin is sampled from specific landmark polygons: left cheek, right cheek, forehead, and jaw/chin. These regions are weighted to prioritize stable areas (cheeks and jaw) while mitigating lighting variance. 

### Color Extraction
The pixels in extracted regions are filtered heavily. The pipeline converts RGB arrays to CIELAB, applying hard minimum/maximum lightness ($L^*$) thresholds and percentile trimming (10th to 90th) to remove deep shadows, blown highlights, and color casts before determining the median color.

### CIELAB
All color math is performed in CIELAB space because it is perceptually uniform. Distances in LAB closely map to human color perception, making it vastly superior to RGB for skin and foundation comparisons.

### ITA (Individual Typology Angle)
ITA is calculated mathematically using $L^*$ (lightness) and $b^*$ (yellow/blue axis):
$\text{ITA} = \text{atan}((L^* - 50) / b^*) \times (180 / \pi)$.
The angle is used to place the user's skin depth into standard buckets (`very_light`, `light`, `intermediate`, `tan`, `brown`, `dark`).

### Undertone
Undertone is mapped using the difference between the $b^*$ (yellow-blue) and $a^*$ (red-green) channels. A significant positive delta indicates warm; negative indicates cool. Notably, the code explicitly detects **olive** undertones if $a^*$ is low (subdued redness) while $b^*$ is strongly present.

### Contrast
Contrast is evaluated by measuring the maximum spread (difference) in Lightness ($L^*$) across the different facial regions. High variance implies high contrast.

## 🍂 12-Season Color Analysis

The application maps the user to one of 12 standard seasons (e.g., True Winter, Soft Summer). 

- **Deterministic Classification:** The classification is strictly rule-based. The code iterates through `classification_rules` in `season_color_reference.json`, evaluating conditions against the user's computed depth, undertone, clarity, and contrast. 
- **Ranked Candidates:** The system provides a ranked list (top 3) of alternative seasons and calculates a margin to flag "close call" categorizations where the user straddles two palettes.
- **LLM Integration:** Groq does *not* decide the season. It simply reads the deterministic output (the winning season and the margin) to tailor its prose.

## 💄 Foundation Matching

- **Database:** Includes 1,303 foundation shades.
- **Conversion:** Each foundation's stored hex color is converted to CIELAB.
- **CIEDE2000:** The match quality is measured using the CIEDE2000 ($\Delta E_{00}$) formula—the industry standard for color difference.
- **Results:** A lower $\Delta E_{00}$ value means the color is visually closer. The API sorts all 1,303 shades by $\Delta E_{00}$ and returns the top 5 closest matches.

## 🤖 Groq AI Layer

The Groq API is strictly used as an **interpretation and presentation layer**. The LLM never touches raw pixels and never makes color science decisions.

- **Inputs:** The deterministic pipeline feeds the LLM the computed skin attributes, ranked season candidates, margin of victory, and the CIEDE2000 foundation matches.
- **Function:** Groq writes a concise, two-paragraph narrative summarizing the user's skin profile and justifying the foundation recommendations.
- **Constraints:** The system prompt explicitly forbids the LLM from listing raw numbers, repeating shade names (which are handled by frontend UI chips), or using generic marketing buzzwords.
- **Model:** Currently configured to point to `openai/gpt-oss-120b` (or similar proxy endpoint) via the standard Groq python client, with `temperature=0.3` for consistency.
- **Fallback:** If the API key is missing or the request fails, the backend seamlessly falls back to a deterministic, templated string response.

## 🏗️ Architecture

```text
 [ ShadeSense.html ]
        │
   (HTTP POST)
        ▼
 [ backend/app.py ]
        │
        ├─► [ ai/colorimetry/face_color.py ] (MediaPipe / Haar)
        │       └─► Pixel filtering & CIELAB & ITA extraction
        │
        ├─► [ ai/recommendation/season.py ]
        │       └─► Rule-based 12-season JSON mapping & Ranking
        │
        ├─► [ ai/recommendation/match.py ]
        │       └─► CIEDE2000 vs 1303 Foundations
        │
        └─► [ ai/recommendation/groq_writer.py ]
                └─► LLM personalized narrative summary
        │
   (JSON Response)
        ▼
 [ ShadeSense.html ]
```

## 📁 Project Structure

```text
ShadeSense/
├── ShadeSense.html                 # Main vanilla JS/HTML frontend
├── backend/
│   ├── app.py                      # Main Flask API server
│   ├── face_detector_v2.py         # Haar cascade fallback logic
│   ├── foundation_tryon.py         # Try-On image processing
│   ├── requirements.txt            # Python dependencies
│   └── .env                        # API keys
├── ai/
│   ├── colorimetry/
│   │   └── face_color.py           # Core skin sampling and CIELAB logic
│   ├── data/
│   │   ├── foundations.json        # Database of 1303 foundation shades
│   │   ├── foundations_enriched.json
│   │   ├── color_palettes.json
│   │   └── season_color_reference.json # Rules for classification
│   ├── recommendation/
│   │   ├── match.py                # CIEDE2000 math
│   │   ├── season.py               # 12-season deterministic logic
│   │   └── groq_writer.py          # LLM prompt construction
├── tests/
│   ├── test_match.py               # Unit tests for matching
│   ├── test_season.py              # Unit tests for seasons
│   └── test_ranked_writer.py       # Unit tests for LLM prompts
└── archive/                        # (Unused/deprecated code)
```

## 🔌 API

### `GET /api/health`
- **Purpose:** Checks if the backend and foundation database are loaded.
- **Response:** `{"status": "healthy", "version": "2.0.0", "foundations_loaded": 1303}`

### `POST /api/analyze`
- **Purpose:** Main pipeline endpoint. Processes an image and returns full analysis.
- **Request Format:** `multipart/form-data` with an `image` file.
- **Response:** JSON containing `face_result` (LAB, ITA, undertone), `season_result` (winning season, rules), `ranked_seasons` (top candidates and margins), `foundation_matches` (top 5), and `ai_recommendation` (Groq summary).

### `POST /api/tryon`
- **Purpose:** Experimental try-on preview of a foundation shade.
- **Request Format:** `multipart/form-data` with `image`, `shade` (name), and `intensity` (float).
- **Response:** Base64 encoded JPEG overlay. *(Note: Returns 503 if Try-On module dependencies are unavailable).*

### `POST /api/analyze-batch`
- **Purpose:** Experimental bulk processing.

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | HTML5, Vanilla JavaScript, CSS |
| **Backend API** | Python, Flask, Flask-CORS |
| **Computer Vision** | OpenCV, MediaPipe Tasks API |
| **Color Science** | NumPy, Custom CIEDE2000 |
| **AI / LLM** | Groq Python SDK |
| **Data** | JSON (Foundations, Season Rules) |

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ShadeSense.git
   cd ShadeSense
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
   *(Note: The MediaPipe `face_landmarker.task` file will download automatically on first run).*

4. **Configure environment variables:**
   Create a `.env` file in the `backend/` directory:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

5. **Start the backend server:**
   ```bash
   python app.py
   ```

6. **Open the frontend:**
   Open `ShadeSense.html` directly in your browser or serve it using a local server (e.g., Live Server or `python -m http.server`).

## 🔐 Environment Variables

- `GROQ_API_KEY`: Required in `backend/.env` to enable the generative AI narrative. If omitted, the app will safely fall back to a templated text response without breaking the colorimetry pipeline.

## 🧪 Testing

The repository includes unit tests to verify the color science and classification logic without needing the Flask server.

- `test_match.py`: Validates the CIEDE2000 mathematical implementation.
- `test_season.py`: Ensures the rule-based seasonal classification outputs the correct json configurations based on mock skin data.
- `test_ranked_writer.py`: Tests the formatting and margin detection for close-call seasonal candidates.

Run tests using:
```bash
python -m unittest discover tests/
```

## 📊 Technical Design Decisions

- **Why CIELAB?** RGB color space is heavily skewed by lighting intensity and is not perceptually uniform. CIELAB separates luminosity ($L^*$) from color ($a^*$, $b^*$), making it dramatically more reliable for analyzing human skin tones across different lighting conditions.
- **Why CIEDE2000?** Standard Euclidean distance in RGB or basic LAB is inaccurate for color matching. CIEDE2000 is the gold-standard mathematical formula used in the cosmetics and paint industries for calculating perceptible color difference.
- **Why deterministic season classification?** LLMs are highly inconsistent at spatial math and color theory. By keeping the classification strictly rule-based in python, the app guarantees repeatable, scientifically grounded results.
- **Why use an LLM only for interpretation?** The LLM serves to translate dense data (ITA degrees, $\Delta E$ values) into warm, human-readable prose.
- **Why regional skin sampling?** Sampling multiple distinct facial polygons (cheeks, forehead, chin) and discarding outliers prevents a single shadow or shiny spot from ruining the overall extraction.

## ⚠️ Limitations

- **Lighting Dependency:** While CIELAB mitigates some lighting variance, extreme color casts (e.g., neon lights) or very dark shadows will still impact extraction. The app does not currently perform full explicit illumination normalization.
- **Cosmetic Formulation:** CIEDE2000 matches pure color values; it cannot account for foundation oxidation on the skin, coverage level (sheer vs full), or finish (matte vs dewy).
- **Physical vs Visual:** Color analysis measures the surface visual color, not the physiological melanin index.
- **Try-On Module:** The try-on endpoint is experimental and relies on simple alpha blending rather than advanced texture-aware recoloring.

## 🔮 Future Improvements

*(FUTURE WORK)*
- **Illumination Normalization:** Implement a White Balance or Color Constancy algorithm prior to sampling.
- **Texture-Aware Try-On:** Migrate from basic blending to a GAN or advanced Poisson blending for realistic cosmetic application.
- **Database Expansion:** Clean up foundation tags and integrate real-time product availability links.
- **Automated Benchmarking:** Create a standardized dataset of faces with known undertones to measure system accuracy automatically over time.

---

## Resume Highlights

- **Engineered an end-to-end computer vision pipeline** utilizing MediaPipe and OpenCV to isolate facial skin regions, applying percentile trimming to reject lighting artifacts and outliers.
- **Implemented robust color science algorithms** translating RGB pixels into CIELAB color space to accurately derive skin undertone, contrast, and depth via Individual Typology Angle (ITA).
- **Developed a deterministic 12-season classification engine** with ranked candidate scoring and ambiguity detection, ensuring scientifically repeatable color analysis.
- **Integrated CIEDE2000 color matching** against a database of 1,303 foundation shades, programmatically identifying the lowest perceptible color difference ($\Delta E_{00}$) for precise cosmetic recommendations.
- **Designed a hybrid AI architecture** utilizing a Groq LLM layer strictly for narrative interpretation, backed by a fully deterministic fallback pipeline for guaranteed uptime.
