"""
Step 1 of the ShadeSense hybrid matching upgrade.

Reads ai/data/foundations.json, tags each shade with finish / coverage /
skin_type_fit using keyword matching first and a Groq LLM fallback for
anything keywords can't resolve, and writes ai/data/foundations_enriched.json.

Does NOT modify foundations.json or touch match.py — this is an isolated,
re-runnable enrichment step.

Usage:
    python ai/data/enrich_foundations.py

Requires:
    GROQ_API_KEY environment variable set (only used for entries keyword
    matching can't tag; if unset, those entries are simply left "unknown"
    and the script still runs).
"""

import json
import os
import re
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    _script_dir = Path(__file__).resolve().parent
    _project_root = _script_dir.parent.parent  # ai/data -> ai -> project root
    _candidates = [
        _project_root / "backend" / ".env",  # matches groq_writer.py's location
        _project_root / ".env",
        Path(".env"),
    ]
    _loaded = False
    for _env_path in _candidates:
        if _env_path.exists():
            load_dotenv(_env_path)
            print(f"📂 Loaded .env from: {_env_path}")
            _loaded = True
            break
    if not _loaded:
        print("⚠️  No .env file found in backend/, project root, or cwd.")
except ImportError:
    pass  # fine if python-dotenv isn't installed — falls back to shell env vars

# ---- Adjust these two paths if your project layout differs ----
INPUT_PATH = Path("ai/data/foundations.json")
OUTPUT_PATH = Path("ai/data/foundations_enriched.json")
# ------------------------------------------------------------------

FINISH_KEYWORDS = {
    "dewy": ["dewy", "luminous", "radiant", "glow", "hydrating finish", "glossy"],
    "matte": ["matte", "shine free", "oil free", "mattifying", "shine control"],
    "satin": ["satin", "semi matte", "soft matte", "velvet"],
    "natural": ["natural finish", "skin like", "second skin", "natural looking", "no filter"],
}

COVERAGE_KEYWORDS = {
    "light": ["light coverage", "sheer", "light medium", "tinted", "bb cream", "cc cream", "skin tint"],
    "medium": ["medium coverage", "buildable", "medium full"],
    "full": ["full coverage", "high coverage", "full cover", "high definition"],
}

SKIN_TYPE_KEYWORDS = {
    "dry": ["dry skin", "hydrating", "moisturizing", "for dry"],
    "oily": ["oily skin", "oil-control", "shine control", "for oily"],
    "combination": ["combination skin", "combo skin"],
    "normal": ["all skin types", "normal skin"],
}

MAX_WORKERS = 10          # concurrent Groq calls in flight
MAX_CALLS_PER_MINUTE = 55  # safety ceiling; lower this if you see 429 errors


class RateLimiter:
    """Thread-safe sliding-window limiter shared across worker threads.
    Lets calls burst up to the cap, then throttles — much faster than a
    fixed per-call sleep because worker threads overlap network latency."""

    def __init__(self, max_per_minute):
        self.max_per_minute = max_per_minute
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0] > 60:
                    self.calls.popleft()
                if len(self.calls) < self.max_per_minute:
                    self.calls.append(now)
                    return
                wait_time = 60 - (now - self.calls[0]) + 0.05
            time.sleep(wait_time)


def keyword_tag(text, keyword_map):
    text = (text or "").lower()
    matches = [tag for tag, keywords in keyword_map.items()
               if any(kw in text for kw in keywords)]
    return matches


def slug_text(url):
    """Pull the last path segment of a product URL and turn dashes/underscores
    into spaces, since the real product name (e.g. 'matte') sometimes only
    survives in the link slug, not the truncated 'product' field."""
    if not url:
        return ""
    slug = url.rstrip("/").split("/")[-1]
    return re.sub(r"[-_]", " ", slug)


def keyword_classify(entry):
    # This dataset has no description/tags fields — the only usable text is
    # the product name, shade name, and the product_link URL slug.
    text = " ".join([
        entry.get("product") or "",
        entry.get("shade") or "",
        slug_text(entry.get("product_link") or ""),
    ])

    finish = keyword_tag(text, FINISH_KEYWORDS)
    coverage = keyword_tag(text, COVERAGE_KEYWORDS)
    skin_type = keyword_tag(text, SKIN_TYPE_KEYWORDS)

    return {
        "finish": finish[0] if finish else None,
        "coverage": coverage[0] if coverage else None,
        "skin_type_fit": skin_type if skin_type else None,
    }


def groq_classify(entry, client, rate_limiter, model="openai/gpt-oss-120b", max_retries=3):
    """Fallback classification using only the provided text. Returns
    'unknown' for any field with no textual basis in the description."""
    brand = entry.get("brand", "")
    name = entry.get("product") or ""
    shade = entry.get("shade") or ""
    link_hint = slug_text(entry.get("product_link") or "")

    prompt = f"""You are classifying a makeup foundation product based ONLY on the
text provided below. This dataset has no product description — you only have the
brand, product name, shade name, and a hint extracted from the product's URL.
Do not use brand reputation or general knowledge to guess — if the text gives no
basis for a field, return "unknown" for it. skin_type_fit will often legitimately
be "unknown" since this data rarely states it — that's fine, don't force a guess.

Brand: {brand}
Product name: {name}
Shade name: {shade}
URL hint: {link_hint}

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"finish": "dewy|matte|satin|natural|unknown", "coverage": "light|medium|full|unknown", "skin_type_fit": ["dry"|"oily"|"combination"|"normal"] or "unknown"}}
"""

    for attempt in range(max_retries):
        rate_limiter.acquire()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
                reasoning_effort="low",
                reasoning_format="hidden",
            )
            raw = response.choices[0].message.content
            if not raw or not raw.strip():
                return {"finish": None, "coverage": None, "skin_type_fit": None}
            raw = raw.strip()
            raw = re.sub(r"^```json\s*|\s*```$", "", raw)
            parsed = json.loads(raw)
            return {
                "finish": None if parsed.get("finish") == "unknown" else parsed.get("finish"),
                "coverage": None if parsed.get("coverage") == "unknown" else parsed.get("coverage"),
                "skin_type_fit": None if parsed.get("skin_type_fit") == "unknown" else parsed.get("skin_type_fit"),
            }
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                backoff = 5 * (attempt + 1)
                print(f"  ⏳ Rate limited on {brand} {name}, retrying in {backoff}s...", file=sys.stderr)
                time.sleep(backoff)
                continue
            print(f"  ⚠️  Groq classification failed for {brand} {name}: {e}", file=sys.stderr)
            return {"finish": None, "coverage": None, "skin_type_fit": None}
    return {"finish": None, "coverage": None, "skin_type_fit": None}


def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: could not find {INPUT_PATH}. Edit INPUT_PATH at the top "
              f"of this script if foundations.json lives somewhere else.")
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        foundations = json.load(f)

    groq_client = None
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=api_key)
        except ImportError:
            print("⚠️  'groq' package not installed — pip install groq. "
                  "Continuing with keyword-only tagging.")
    else:
        print("⚠️  GROQ_API_KEY not set — continuing with keyword-only tagging.")

    total = len(foundations)
    enriched = [None] * total
    needs_groq = []  # indices that keyword matching couldn't resolve

    # Pass 1: keyword tagging — fast, sequential, no need to parallelize
    for i, entry in enumerate(foundations):
        tags = keyword_classify(entry)
        used_keyword = any(v is not None for v in tags.values())
        merged = dict(entry)
        merged["finish"] = tags["finish"]
        merged["coverage"] = tags["coverage"]
        merged["skin_type_fit"] = tags["skin_type_fit"]
        enriched[i] = merged
        if not used_keyword:
            needs_groq.append(i)

    keyword_tagged = total - len(needs_groq)
    print(f"Keyword pass done: {keyword_tagged}/{total} tagged, "
          f"{len(needs_groq)} need Groq.")

    # Pass 2: Groq fallback, run concurrently with a shared rate limiter
    groq_tagged = 0
    if groq_client is not None and needs_groq:
        rate_limiter = RateLimiter(MAX_CALLS_PER_MINUTE)
        start = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(groq_classify, foundations[i], groq_client, rate_limiter): i
                for i in needs_groq
            }
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                tags = future.result()
                enriched[i]["finish"] = tags["finish"]
                enriched[i]["coverage"] = tags["coverage"]
                enriched[i]["skin_type_fit"] = tags["skin_type_fit"]
                if any(v is not None for v in tags.values()):
                    groq_tagged += 1
                completed += 1
                if completed % 25 == 0:
                    elapsed = time.time() - start
                    print(f"  ...{completed}/{len(needs_groq)} Groq calls done "
                          f"({elapsed:.0f}s elapsed)")

    counts = {
        "keyword_tagged": keyword_tagged,
        "groq_tagged": groq_tagged,
        "still_unknown_finish": sum(1 for e in enriched if e["finish"] is None),
        "still_unknown_coverage": sum(1 for e in enriched if e["coverage"] is None),
        "still_unknown_skin_type": sum(1 for e in enriched if not e["skin_type_fit"]),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print("\n--- Enrichment summary ---")
    print(f"Total entries:        {total}")
    print(f"Keyword-tagged:       {counts['keyword_tagged']}")
    print(f"Groq-tagged:          {counts['groq_tagged']}")
    print(f"Still unknown finish: {counts['still_unknown_finish']}")
    print(f"Still unknown cover.: {counts['still_unknown_coverage']}")
    print(f"Still unknown skin:   {counts['still_unknown_skin_type']}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()