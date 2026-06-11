# ShadeSense AI ✨
### Objective, Calibrated Skin Tone & Undertone Matcher

ShadeSense AI is a state-of-the-art web application designed to analyze human facial selfies, balance ambient lighting parameters, extract precise cheek regions, and run objective color-science mathematical formulas to identify the user's cosmetic depth, Individual Typology Angle (ITA), and skin undertone (Warm, Cool, Neutral). 

It features a high-fidelity **Flask API** leveraging MediaPipe and OpenCV, and an ultra-premium, luxury-branded **React (Vite) frontend** using glassmorphic styling, live camera framing cues, interactive canvases, and customizable color palette controls.

---

## 🏗️ Architectural Topology

```
shadesense-ai/
├── backend/
│   ├── app.py                 ← Flask app orchestrator
│   ├── routes.py              ← API endpoints (/health, /analyze)
│   ├── face_detector.py       ← MediaPipe Face Mesh landmark extraction
│   ├── undertone_analyzer.py  ← LAB, ITA, and Warm/Cool/Neutral rules
│   ├── recommender.py         ← Personalized foundation & clothing DB
│   ├── lighting.py            ← Gray World White Balance & CLAHE exposure
│   ├── requirements.txt       ← Python environment dependencies
│   └── .env                   ← Port and debugging configs
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx            ← State Router
│   │   ├── main.jsx           ← Entry index
│   │   ├── index.css          ← Luxury Dark Custom styling & variables
│   │   ├── services/
│   │   │   └── api.js         ← Native fetch wrappers
│   │   ├── pages/
│   │   │   ├── UploadPage.jsx ← Splash, Drag-Drop, Live Webcam Capture
│   │   │   └── ResultsPage.jsx← Dashboard canvas targets & color grids
│   │   └── components/
│   │       ├── DropZone.jsx   ← Animated drag-and-drop
│   │       ├── PaletteCard.jsx← Color swatches with click-to-copy
│   │       └── ShadeBar.jsx   ← Pinpoint skin spectrum visualizer
│   ├── index.html             ← Fonts & SEO tags
│   ├── vite.config.js
│   ├── package.json
│   └── .env                   ← VITE_API_URL pointer
│
└── README.md                  ← This documentation
```

---

## 🧪 Scientific Foundations & Algorithms

### 1. Ambient Lighting Correction (`lighting.py`)
To prevent ambient colored lights (e.g., warm indoor lighting or blue daylight) from skewing color results, ShadeSense AI performs dual-stage normalization:
* **Gray World White Balance:** Adjusts the image channels ($R$, $G$, $B$) under the mathematical assumption that the mean scene color is neutral gray.
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Executed in the CIELAB $L^*$ (Lightness) channel to level localized shadows or highlights without altering color values.

### 2. Cheek Localization Mesh (`face_detector.py`)
Rather than crude center-cropping, the backend deploys **MediaPipe Face Mesh** to extract exact coordinate regions:
* Locates anatomical **Left Cheek** (Landmark 346) and **Right Cheek** (Landmark 117).
* Extracts a $4\% \times 4\%$ regional bounding box around each landmark, calculating the spatial pixel averages to eliminate noise.
* Gracefully falls back to optimized mathematical default coordinates if the user's face landmarks cannot be mapped.

### 3. CIELAB & ITA Mathematics (`undertone_analyzer.py`)
Averages cheek RGB values and translates them into the **CIE $L^*a^*b^*$** color space (highly uniform and close to human perception):
* **Individual Typology Angle (ITA):**
  $$ITA = \arctan2(L^* - 50, b^*) \times \frac{180}{\pi}$$
  Classifies skin scientifically into *Very Light (>55)*, *Light (41-55)*, *Intermediate (28-41)*, *Tan (10-28)*, *Brown (-30 to 10)*, or *Dark (<-30)*.
* **Undertone Profiling:**
  Evaluates the yellow-to-red ratio ($b^* / a^*$) and Hue Angle.
  * $b^* / a^* > 1.6 \rightarrow$ **Warm**
  * $b^* / a^* < 1.15 \rightarrow$ **Cool**
  * Otherwise $\rightarrow$ **Neutral** (adjusted for near-gray saturation thresholds)

---

## 🚀 Running the Project Locally

### 1. Launch the Backend API
1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Activate your virtual environment and install dependencies:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the development Flask server:
   ```bash
   python app.py
   ```
   *The server launches at `http://localhost:5001`.*

### 2. Launch the React Frontend
1. Navigate to the `frontend/` directory:
   ```bash
   cd ../frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Start the Vite development client:
   ```bash
   npm run dev
   ```
   *The web client launches at `http://localhost:5173`.*

---

## 🎨 Luxury Brand Aesthetics
* **Theme**: Deep Charcoal (`#08080a`), rich graphite panels, glowing Champagne Gold (`#d4af37`), and soft gold/rose gold radial accents.
* **Typography**: Luxury serif *Playfair Display* for titles combined with modern clean *Plus Jakarta Sans* for controls.
* **Animations**: Shimmering skeleton loaders, interactive canvas targets drawing precise facial overlays, and micro-interactive color swatches that copy values directly to your clipboard.
