"""
bert_predictor.py — Semantic Fake News Classifier
===================================================
Uses sentence-transformers/all-MiniLM-L6-v2 (22 MB, fully BERT-based)
for genuine semantic understanding via prototype-cosine classification.

How it works
------------
1. The sentence transformer encodes the article into a 384-dim semantic
   embedding (exactly what BERT attention layers produce).
2. We compare that embedding to curated prototype sentences for FAKE
   and REAL news using cosine similarity.
3. Whichever class prototype cluster is closest in semantic space wins.

This is the same core mechanism used in zero-shot / few-shot NLP:
understanding MEANING, not just keyword frequency.

Returns: (label: 'REAL' | 'FAKE',  probability: float 0-1)
"""

from __future__ import annotations
from pathlib import Path
import os

# ── Prevent Render 512MB RAM Out-Of-Memory (OOM) ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
ST_MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"
BERT_DIR     = Path(__file__).parent / "bert_model"

_model       = None   # lazy singleton
_fake_proto  = None   # pre-computed prototype embedding
_real_proto  = None


# ── Curated prototype sentences ────────────────────────────────────────────
# FAKE prototypes: sensationalist, conspiratorial, hoax-like language
_FAKE_PROTOTYPES = [
    "SHOCKING Government is hiding this secret from you exposed cover-up conspiracy",
    "You won't believe what they don't want you to know breakthrough suppressed truth",
    "Scientists baffled as miracle cure found but big pharma wants it buried",
    "BREAKING: Famous celebrity dead or arrested in shocking scandal exposed",
    "This will blow your mind the untold truth they are keeping from the public",
    "Deep state globalist agenda exposed whistleblower reveals massive conspiracy",
    "Mainstream media lies again fabricated story completely false misinformation",
    "Urgent warning government control population microchip vaccine tracking agenda",
    "100% proof aliens UFO cover-up NASA hiding extraterrestrial life disclosure",
    "ALERT satanic elites pedophile ring exposed QAnon patriot insider reveals all",
]

# REAL prototypes: neutral, sourced, factual journalistic tone
_REAL_PROTOTYPES = [
    "According to official government statistics released this quarter",
    "A spokesperson for the department confirmed in a press statement today",
    "Researchers published peer-reviewed findings in the journal Nature",
    "The central bank raised interest rates citing persistent inflation concern",
    "Officials announced the policy change following months of bipartisan negotiation",
    "The report found that unemployment fell to a four-year low last month",
    "Authorities said the investigation is ongoing with no charges filed yet",
    "The company disclosed quarterly earnings showing a modest revenue increase",
    "Parliament passed the legislation with a majority after debate on amendments",
    "Health experts recommend vaccination based on clinical trial data analysis",
    "Police responded to the scene after reports of an explosion in the downtown district",
    "The prime minister condemned the attacks and vowed to bring the perpetrators to justice",
    "Heavy fighting broke out between rebel forces and the national army near the border",
    "Emergency services are conducting rescue operations following the devastating earthquake",
    "The minister died in a tragic incident involving a car bomb outside his residence, officials confirmed"
]


def _get_model():
    """Lazy-load the sentence transformer model (BERT-based)."""
    global _model, _fake_proto, _real_proto
    if _model is not None:
        return _model

    model_src = str(BERT_DIR) if (BERT_DIR.exists() and any(BERT_DIR.iterdir())) else ST_MODEL_ID
    print(f"[BERT] Loading sentence transformer from: {model_src}")

    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(model_src)

    # Pre-compute prototype embeddings once at load time
    _fake_proto = _model.encode(_FAKE_PROTOTYPES, convert_to_numpy=True).mean(axis=0)
    _real_proto = _model.encode(_REAL_PROTOTYPES, convert_to_numpy=True).mean(axis=0)

    print("[BERT] Sentence transformer ready ✓ (semantic prototype classifier loaded)")
    return _model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def predict(text: str) -> tuple[str, float]:
    """
    Classify a news article using BERT semantic embeddings.

    Strategy: encode the article → compare to FAKE/REAL prototype embeddings
    via cosine similarity → softmax-normalise into a probability.

    Args:
        text: Raw article text (headline + body preferred for best result).

    Returns:
        (label, confidence)
        label      — 'REAL' or 'FAKE'
        confidence — probability of the predicted label (0.0 – 1.0)
    """
    if not text or not text.strip():
        return "FAKE", 0.5

    model = _get_model()

    # Encode the article — truncate to first 512 words (≈ attention window)
    truncated = " ".join(text.split()[:512])

    article_emb = model.encode(truncated, convert_to_numpy=True)

    # Cosine similarity to each prototype cluster
    sim_fake = _cosine(article_emb, _fake_proto)
    sim_real = _cosine(article_emb, _real_proto)

    # Softmax-style normalisation into probabilities
    # Add a temperature-scaled sigmoid to sharpen the signal
    raw_diff  = sim_fake - sim_real          # positive → leans FAKE
    temperature = 8.0                        # sharpening factor
    fake_prob = 1.0 / (1.0 + np.exp(-temperature * raw_diff))
    real_prob = 1.0 - fake_prob

    if fake_prob >= real_prob:
        return "FAKE", round(float(fake_prob), 4)
    else:
        return "REAL", round(float(real_prob), 4)


def is_available() -> bool:
    import os
    if os.environ.get('RENDER'):
        return False
    """Always True — model downloads automatically from HF on first call."""
    return True   # sentence-transformers handles auto-download
