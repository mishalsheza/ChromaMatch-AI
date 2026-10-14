# ShadeSense

ShadeSense measures skin colour from a face photo and suggests foundation shades and a
seasonal colour palette. The measurements are deterministic colour science: MediaPipe
landmarks, CIELAB, ITA and CIEDE2000. An LLM (Groq) is used only to explain the computed
results in plain language. It never sees the photo.

> The GitHub repository is named **ChromaMatch-AI**; ShadeSense is the app's name.

**Status:** portfolio project. On 50 randomly sampled, hand-labelled photos, **skin depth
agrees with people 80% of the time, but undertone only 25%, which is about chance**. The app therefore
shows undertone (and the season, which depends on it) as a low-confidence best guess. See
[Evaluation & limitations](#evaluation--limitations). Foundation matching uses the measured
colour directly, not the undertone label.

---

## Architecture: colour science computes, the LLM only explains

```text
 Browser (ShadeSense.html)
        │  photo (multipart upload, decoded in memory, never written to disk)
        ▼
 backend/app.py ── thin Flask routes: validation, limits, CORS, rate limiting
        │
        ▼
 backend/analysis_service.py
        │
        ├─ ai/colorimetry/face_color.py      landmarks → skin regions → CIELAB → depth, undertone, clarity, contrast
        ├─ ai/colorimetry/skin_observations.py  under-eye / perioral / redness colour differences (cosmetic only)
        ├─ ai/recommendation/match.py        CIEDE2000 against 1,303 foundation shades
        ├─ ai/recommendation/season.py       score and rank all 12 seasons
        └─ ai/recommendation/groq_writer.py  numbers → short narrative (LLM), or a fixed template
                    ▲
                    └─ receives TEXT only: the computed labels, scores and shade names
```

Every decision (skin colour, depth, undertone, shade ranking, season) is made by code you
can read and test. The LLM gets a text prompt built from those results and writes two or
three short paragraphs. If `GROQ_API_KEY` isn't set or the call fails, a deterministic
template is used instead, and the API response says so: `"ai_source": "template"` vs
`"groq"`. The photo is never sent to any external service.

---

## How the pipeline works

### 1. Face landmarks
MediaPipe Face Landmarker (`backend/face_landmarker.task`; if missing it's downloaded and
SHA-256-verified) returns 478
landmarks: the 468-point face mesh plus 10 iris points. The skin regions use mesh points
only. If it finds no face, a Haar-cascade face box is used as a coarse fallback; those
results get a 50% confidence penalty and are labelled `"landmark_source": "haar"`.

### 2. Skin regions
Four **skin regions** (forehead, left cheek, right cheek, chin) determine the skin colour.
Four **observation regions** (both under-eye bands, both sides of the mouth) are measured
separately and only feed the cosmetic observations, because they're expected to differ.

The landmark indices were derived from MediaPipe's **official canonical face mesh**
(`ai/data/mediapipe/canonical_face_model.obj`). Each region is the set of mesh vertices
inside an anatomical box, kept clear of MediaPipe's official eye, eyebrow, lip and nose
landmark sets. `tests/test_face_regions.py` checks this, and
`scripts/visualize_regions.py photo.jpg` draws the regions on any photo. Each polygon is
shrunk to 85% around its centre to stay off hair, brows and edges.

### 3. Colour per region
- Pixels with HSV value < 35 (deep shadow) or > 245 (specular highlight) are dropped.
- k-means (k=3) on the rest; the largest cluster is kept, which rejects stray hair, stubble
  and make-up edges.
- Its **median** is converted to CIELAB (sRGB, D65), using the same conversion function as
  the foundation hex codes (`ai/colorimetry/color_space.py`).

### 4. Overall skin colour
The median of the skin regions. A region more than ΔE 20 (CIE76) from the others' median
is discarded first (typically bangs over the forehead).

**No illumination correction is applied.** A gray-world white balance was tried and is
switched off: on face close-ups the "scene average" is mostly skin, so it pushed b\*
towards 0 and lightness towards mid-gray, turning light faces "deep" and deep faces
"light". It's still available behind `SHADESENSE_APPLY_WHITE_BALANCE=1` for experiments.

### 5. Labels
| Output | Rule |
|---|---|
| **Depth** | ITA = atan((L\* − 50) / b\*). Buckets (Chardon et al.): >55 very light, 41–55 light, 28–41 intermediate, 10–28 tan, −30–10 brown, < −30 dark. Mapped to depth as: very light + light (ITA ≥ 41°) → `light`; intermediate + tan (10° ≤ ITA < 41°) → `medium`; brown + dark (ITA < 10°) → `deep` |
| **Undertone** | `olive` if a\* ≤ 6, b\* ≥ 14 and b\* ≥ 2.2·a\*; otherwise by the ratio b\*/a\*: ≥ 1.8 `warm`, ≤ 0.6 `cool`, else `neutral`. **Unreliable (≈ chance against human labels)**: the API returns it with `undertone_reliability: "low"` and a note, and it's presented as a best guess. |
| **Clarity** | chroma √(a\*² + b\*²) ≥ 22 → `clear`, else `muted` |
| **Contrast** | spread of L\* across the skin regions: ≥ 8 `high`, ≤ 3 `low`. This is a proxy: seasonal "contrast" usually means hair/eyes vs skin, which isn't measured. It's forced to `medium` when region confidence is low. |
| **Measurement quality** | heuristic, 0–1: usable pixels per region × dominant-cluster share, reduced by disagreement between regions. Not a calibrated probability. |

All thresholds are named constants at the top of `face_color.py`. They are hand-picked,
not fitted to labelled data.

### 6. Seasons (`season.py`)
Each of the 12 seasons in `ai/data/season_color_reference.json` is scored against the
profile: undertone compatibility 0–3, depth 0–2 (adjacent depth 0.5), clarity 0–2,
contrast 0–2, for a maximum of 9. A zero on undertone multiplies the score by 0.25; other
zeros by 0.75 (one) or 0.55 (two or more). The top season wins. If the gap to second place
is under 10 percentage points, it's flagged as a **close call** and the narrative hedges.
Every season can win for some input (tested). 21 of the 135 input combinations tie at the
top: the tie is broken by file order and always reported as a close call.

### 7. Foundation matching (`match.py`)
1,303 shades from 28 brands (`ai/data/foundations.json`, collected from the public
[Makeup API](https://makeup-api.herokuapp.com/) by `scripts/fetch_foundations.py`). Each
hex code is converted to CIELAB and ranked by **CIEDE2000** distance to the skin colour;
the top 5 are returned. The implementation reproduces all 34 reference pairs from Sharma,
Wu & Dalal (2005) to 4 decimal places (`tests/test_color_space.py`).

Caveat: the hex codes are product-swatch colours from a web API, not measured values of
the foundation on skin. Oxidation, finish and coverage aren't modelled.

### 8. Narrative (`groq_writer.py`)
Model `openai/gpt-oss-120b` via Groq (configurable), temperature 0.3, 20 s timeout. The
prompt contains the labels, ITA, top season candidates and their scores, the top 3 shades
with ΔE, and any skin observations. The system prompt forbids clinical terms and asks the
model not to quote raw numbers.

### 9. Try-on (experimental)
`POST /api/tryon` takes a brand and shade, looks up its hex in the same shade database,
and shifts the skin pixels' colour towards it with simple alpha blending. It's a rough
preview, not texture-aware recolouring.

---

## Why not a trained classifier?

The first version trained a CNN to classify complexion from photos (no longer in the
repository). The data couldn't support it:

- **Severe class imbalance:** about 300 "light" images against about 30 "olive", so a model
  can score well by mostly ignoring the rare classes.
- The volume was small.
- The labels weren't human judgements: they came from an earlier colour heuristic.

A model trained on that would mostly learn the dataset's biases. A deterministic pipeline
can be inspected, explained line by line, tested against published reference values, and
applies the same rules to every face, so its errors are at least systematic and
discoverable rather than hidden in weights.

---

## Evaluation & limitations

### What has been measured

**Colour maths:** sRGB→CIELAB matches reference values (white → L\*=100, sRGB 128 grey →
L\*=53.59, the sRGB primaries). CIEDE2000 matches all 34 Sharma reference pairs.

**Lighting stability** (`python scripts/eval.py stability`, 40 images, each compared with
its own unmodified photo):

| Perturbation (applied in linear RGB) | median ΔE00 | depth label changed | undertone label changed |
|---|---|---|---|
| exposure −0.5 EV | 7.4 | 20% | 2% |
| exposure +0.5 EV | 5.7 | 9% | 16% |
| warmer light (R×1.10, B×0.88) | 3.6 | 15% | 21% |
| cooler light (R×0.90, B×1.12) | 3.7 | 10% | 18% |
| much warmer (R×1.20, B×0.78) | 6.3 | 27% | 27% |
| much cooler (R×0.80, B×1.25) | 7.7 | 25% | 40% |

ΔE00 ≈ 1 is barely visible and 2–3 is a noticeable shade difference. In other words,
**the result depends heavily on the lighting**. Without a colour reference in the photo,
exposure and illuminant colour can't be separated from skin colour, and this pipeline
doesn't try.

### Accuracy against human labels
I hand-labelled 60 UTKFace images: 50 sampled at random and 10 extra rare-class
candidates. **The accuracy figures below are on the 50 random images** (undertone on 48,
because I marked two as unsure); the 10 extras count towards precision only. Reproduce
with `python scripts/eval.py run`.

| | Agreement with my labels | 95% CI | Baseline (always majority) |
|---|---|---|---|
| **Depth** | 80% (balanced 79%) | 67–89% | 44% |
| **Undertone** | 25% (balanced 30%) | 15–39% | 33% |

Depth works; 7 of its 10 errors are between neighbouring classes (the other 3 are deep faces
read as light). Undertone is no better than chance: the classifier predicts "neutral" for
75% of faces and never detects olive (0/13). When it does predict "warm", it was right 6/6
times, so the thresholds are too conservative. Foundation matching uses the measured
colour, not the undertone label, so it isn't affected. The UI now presents undertone and
season as low-confidence guesses. I didn't re-tune thresholds on this set, because that
would overfit it.

**Limitations:** single labeller; small sample; undertone is ambiguous from photos
(rechecking my first 40 labels changed 16 of them); I'd seen the old heuristic labels for
the first 40 images.

### What changed because of this evaluation
- **UI:** undertone is shown as *"Undertone (best guess)"* in italics, with a note that it
  matched people about as often as chance. The suggested palette is described as
  depending heavily on that guess. The "Confidence %" badge is now called *measurement
  quality*, because it measures how clean the skin sample was, not how likely the labels
  are to be right.
- **LLM prompt:** the system prompt tells the model never to state the undertone as a
  fact or use it as the main reason for a conclusion, to hedge it if mentioned, and to
  present the season as a starting point. The template fallback says the same.
- **API:** `analysis.undertone_reliability = "low"` plus `analysis.undertone_note`, so any
  client can show the caveat.
- **Unchanged:** thresholds, season weights and foundation matching. Matching never used
  the undertone label; it ranks shades by ΔE00 from the measured colour.

Why the season needs the caveat too: the season scorer gives undertone the largest weight
(3 of 9 points, ×0.25 on a mismatch). In every one of the 27 depth/clarity/contrast
combinations, changing only the undertone changes the winning season. So the season
inherits the undertone's unreliability. Re-weighting it towards depth is a sensible next
step, but it needs a season-labelled evaluation set first.

Earlier versions of this repo reported "97% undertone accuracy". That number was invalid:
it scored the pipeline against labels the pipeline had generated itself (details in
[`archive/circular_eval/README.md`](archive/circular_eval/README.md)).

### Known limitations
- **Lighting**, as above. Results from coloured or mixed lighting shouldn't be trusted.
- **Undertone is effectively unreliable** (25%, about chance); see above. Treat it, and the
  season, as a suggestion to check in person.
- **Contrast** is a lighting-sensitive proxy, and **season ties** exist (see §6).
- **Skin observations** (under-eye, perioral, redness) use hand-picked thresholds that
  haven't been validated. They're phrased as cosmetic notes, never diagnoses.
- **Small or low-resolution faces** give small regions and low confidence.
- **Foundation colours** are web swatches, not measurements (see §7).

---

## Setup (fresh clone)

Requires **Python 3.11**.

```bash
git clone https://github.com/mishalsheza/ChromaMatch-AI.git
cd ChromaMatch-AI
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # runtime + test/lint tools (runtime only: backend/requirements.txt)

cp backend/.env.example backend/.env # optional: add GROQ_API_KEY for LLM-written text
python backend/app.py
```

Then open **http://localhost:5001**. The backend serves the page itself. Without a Groq key
everything works, and the explanation text comes from a template.

```bash
pytest            # one end-to-end test is skipped if the dev dataset isn't present
ruff check .      # lint
```

### Configuration
All settings are environment variables, documented in `backend/.env.example`: port, CORS
origins, upload size (10 MB), maximum image pixels, batch size (5), rate limits (10/min
and 200/day per IP for analysis), Groq model and timeout, and the white-balance flag.

For deployment, use a WSGI server instead of Flask's dev server, for example
`pip install gunicorn` then `gunicorn -w 2 -b 0.0.0.0:5001 backend.app:app`. With several workers, point
`SHADESENSE_RATE_LIMIT_STORAGE` at Redis so the rate limits are shared.

### Tools
```bash
python scripts/visualize_regions.py face.jpg      # draw the sampling regions → region_previews/
python scripts/eval.py sample|run|stability       # evaluation (see eval/README.md)
python scripts/bias_audit.py ai/data/raw          # per-group output distribution on a folder dataset
```

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` (`image`) | Full analysis: `analysis`, `foundations`, `season`, `skin_observations`, `recommendations`, `ai_source` |
| `POST /api/tryon` (`image`, `brand`, `shade`, `intensity` 0–1) | Try-on preview as base64 JPEG |
| `POST /api/analyze-batch` (`images`, ≤ 5) | Compact results, no LLM call |
| `GET /api/health` | `status` (`ok` / `degraded` / `unavailable`) and per-component readiness |

Errors are JSON (`{"success": false, "error": "..."}`) with 400 (bad input / no face),
413 (too large), 429 (rate limited) or 500 (logged server-side; no internals returned).

---

## Project structure

```text
ShadeSense.html                  single-page frontend (vanilla JS)
backend/
  app.py                         Flask routes, config, limits
  analysis_service.py            runs the pipeline for one image
  foundation_tryon.py            experimental try-on preview
  face_landmarker.task           MediaPipe model (Apache 2.0)
ai/
  colorimetry/
    color_space.py               sRGB → CIELAB (single implementation)
    face_color.py                landmarks, regions, colour, labels
    skin_observations.py         cosmetic colour-difference notes
  recommendation/
    match.py                     CIEDE2000 + foundation database
    season.py                    12-season scoring
    groq_writer.py               prompt + LLM call + template fallback
  data/
    foundations.json             1,303 shades
    season_color_reference.json  season definitions and palettes
    mediapipe/                   official canonical face mesh (for tests)
scripts/                         region visualiser, evaluation, audit, data provenance
eval/                            hand-labelling template and instructions
tests/                           pytest suite (no network calls)
archive/circular_eval/README.md  why the earlier "97% accuracy" figure was invalid
```

---

## Data, licences and privacy

- **Photos** are decoded in memory and never stored. Try-on output is re-encoded, so no
  EXIF (e.g. GPS) metadata is returned.
- **No photo is sent to any external service.** Only derived text goes to Groq, and only
  when a key is configured.
- MediaPipe model and canonical face mesh: © Google, Apache License 2.0.
- Foundation data: Makeup API (makeup-api.herokuapp.com).
- Face images: development and evaluation used [UTKFace](https://susanqq.github.io/UTKFace/),
  which is licensed for **non-commercial research purposes only**. The images are not
  included in the repository and aren't redistributable; `eval/README.md` explains how to
  build a labelled set.
