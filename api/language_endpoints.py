"""
Multilingual Support Endpoints
 - Language identification (heuristic, offline)
 - Code-switching detection
 - Translation integration (model-backed when available, safe fallback otherwise)
 - Cultural adaptation (hints + voice suggestions)
"""

from fastapi import APIRouter, Form, HTTPException
from typing import Dict, Any, Optional
import re

from config.languages import SUPPORTED_LANGUAGES

try:
    from models.qwen_omni import generate_text_response  # Optional translation backend
    HAS_QWEN = True
except Exception:
    HAS_QWEN = False

router = APIRouter(prefix="/language", tags=["multilingual"])


# Simple Unicode script buckets for heuristic detection
SCRIPT_BUCKETS = {
    'latin': re.compile(r"[A-Za-z]"),
    'cyrillic': re.compile(r"[\u0400-\u04FF]"),
    'arabic': re.compile(r"[\u0600-\u06FF]"),
    'devanagari': re.compile(r"[\u0900-\u097F]"),
    'thai': re.compile(r"[\u0E00-\u0E7F]"),
    'hangul': re.compile(r"[\uAC00-\uD7AF]"),
    'hiragana': re.compile(r"[\u3040-\u309F]"),
    'katakana': re.compile(r"[\u30A0-\u30FF]"),
    'cjk': re.compile(r"[\u4E00-\u9FFF]"),
}


LANG_HINTS = [
    ("zh", ["cjk"]),
    ("ja", ["hiragana", "katakana", "cjk"]),
    ("ko", ["hangul"]),
    ("ar", ["arabic"]),
    ("hi", ["devanagari"]),
    ("th", ["thai"]),
    ("ru", ["cyrillic"]),
    ("en-US", ["latin"]),
    ("es", ["latin"]),
    ("fr", ["latin"]),
    ("de", ["latin"]),
    ("it", ["latin"]),
    ("pt", ["latin"]),
]


def _script_distribution(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in SCRIPT_BUCKETS}
    for ch in text:
        for name, rx in SCRIPT_BUCKETS.items():
            if rx.match(ch):
                counts[name] += 1
                break
    return counts


def _guess_language(text: str) -> Dict[str, Any]:
    dist = _script_distribution(text)
    total = sum(dist.values()) or 1
    # Score languages by matching script presence
    scores = []
    for lang, scripts in LANG_HINTS:
        score = sum(dist[s] for s in scripts)
        scores.append((lang, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    lang, score = scores[0]
    confidence = round(min(1.0, score / total), 3)
    return {"language": lang, "confidence": confidence, "scripts": dist}


@router.post("/detect")
async def detect_language(text: str = Form(...)):
    """Heuristic offline language detection with script distribution and confidence."""
    try:
        result = _guess_language(text)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/code-switch")
async def code_switch(text: str = Form(...), threshold: float = Form(0.15)):
    """Detect code-switching by checking multiple script presence above a threshold share."""
    try:
        dist = _script_distribution(text)
        total = sum(dist.values()) or 1
        shares = {k: v / total for k, v in dist.items() if v > 0}
        # Consider code-switching if 2+ scripts exceed threshold
        strong = {k: s for k, s in shares.items() if s >= threshold}
        is_code_switch = len(strong) >= 2
        return {"success": True, "code_switch": is_code_switch, "shares": shares, "strong_scripts": strong}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/translate")
async def translate_text(text: str = Form(...), target_lang: str = Form("en-US"), source_lang: Optional[str] = Form(None)):
    """Translate text using the available model backend (if present). Falls back to echo with hint."""
    try:
        if source_lang is None:
            source_lang = _guess_language(text)["language"]
        if source_lang == target_lang:
            return {"success": True, "source_lang": source_lang, "target_lang": target_lang, "translated": text, "note": "source=target"}

        if HAS_QWEN:
            prompt = f"Translate the following text from {source_lang} to {target_lang}. Keep meaning and tone.\n\nTEXT:\n{text}"
            out = generate_text_response(prompt, max_length=800)
            return {"success": True, "source_lang": source_lang, "target_lang": target_lang, "translated": out}
        else:
            return {"success": True, "source_lang": source_lang, "target_lang": target_lang, "translated": text, "note": "translation backend unavailable"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices")
async def list_voices():
    """List supported languages and voices from config."""
    return {"success": True, "languages": SUPPORTED_LANGUAGES, "count": len(SUPPORTED_LANGUAGES)}


@router.post("/culture-hints")
async def culture_hints(lang: str = Form("en-US")):
    """Return basic cultural adaptation hints by language region."""
    hints = {
        'en-US': ["Direct, concise messaging", "Emphasize benefits and outcomes"],
        'en-GB': ["Polite and reserved tone", "Understatement over hyperbole"],
        'es': ["Warm, relational tone", "Use inclusive language"],
        'fr': ["Elegant phrasing", "Clarity and structure"],
        'de': ["Precise and thorough", "Data-driven arguments"],
        'zh': ["Respect hierarchy and context", "Avoid direct confrontation"],
        'ja': ["Politeness and humility", "Indirect suggestions"],
        'ko': ["Honorifics and respect", "Collective perspective"],
        'ar': ["Formal greetings", "Relationship-centric"],
    }
    out = hints.get(lang, ["Adapt tone to audience context", "Avoid idioms that may not translate"]) 
    return {"success": True, "lang": lang, "hints": out}

