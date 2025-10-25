"""Centralized text preprocessing utilities shared across training and inference.

This module keeps the text standardisation pipeline consistent everywhere. It
covers casing, removal of punctuation/artefacts, stopword filtering, stemming,
and domain-specific synonym normalisation used by the halal/haram classifier.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable, Sequence

import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

__all__ = [
    "preprocess_text",
    "ensure_nltk_resources",
]

# ---------------------------------------------------------------------------
# NLTK resource management
# ---------------------------------------------------------------------------


def ensure_nltk_resources(packages: Sequence[str] | None = None) -> None:
    """Ensure required NLTK corpora are available before using them."""
    required = packages or ("stopwords", "wordnet", "omw-1.4")
    for package in required:
        lookup_path = f"corpora/{package}"
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(package)


ensure_nltk_resources()

# ---------------------------------------------------------------------------
# Stopwords & Stemmer initialisation
# ---------------------------------------------------------------------------

factory = StemmerFactory()
_stemmer = factory.create_stemmer()

_STOPWORDS: set[str] = set()
for language in ("indonesian", "english"):
    _STOPWORDS.update(stopwords.words(language))

_EXTRA_STOPWORDS = {
    "and",
    "with",
    "contains",
    "containing",
    "include",
    "including",
    "made",
    "make",
    "product",
    "products",
    "may",
    "also",
    "less",
    "than",
    "from",
    "per",
    "based",
}
_STOPWORDS.update(_EXTRA_STOPWORDS)

# ---------------------------------------------------------------------------
# Normalisation dictionaries & regex rules
# ---------------------------------------------------------------------------

# Domain-specific mapping for ingredients frequently seen in labels.
_BASE_NORMALISATION_MAP: dict[str, str] = {
    "pork": "babi",
    "pig": "babi",
    "gelatin": "babi",
    "gelatine": "babi",
    "lard": "babi",
    "bacon": "babi",
    "ham": "babi",
    "swine": "babi",
    "mechanically separated chicken": "ayam mekanis",
    "mechanically separated turkey": "kalkun mekanis",
    "mechanically separated meat": "daging mekanis",
    "alcohol": "alkohol",
    "wine": "anggur alkohol",
    "beer": "bir",
    "rum": "rum",
    "vodka": "vodka",
    "sake": "sake",
    "bourbon": "bourbon",
    "rennet": "rennet",
    "lipase": "lipase",
    "pepsin": "pepsin",
    "tallow": "lemak hewan",
    "animal fat": "lemak hewan",
    "beef fat": "lemak sapi",
    "chicken fat": "lemak ayam",
    "duck fat": "lemak bebek",
    "lecithin soy": "soy lecithin",
    "soya lecithin": "soy lecithin",
    "lecithin (soy)": "soy lecithin",
    "lecithin (soya)": "soy lecithin",
    "lecithin (from soy)": "soy lecithin",
    "lecithin (from soya)": "soy lecithin",
    "soybean lecithin": "soy lecithin",
    "soya bean lecithin": "soy lecithin",
}

# Regex rules allow us to catch variations that include punctuation or hyphenated
# phrasing before we strip them out.
_NORMALISATION_REGEX = [
    (re.compile(r"lecithin\s*\(\s*soy\s*\)"), "soy lecithin"),
    (re.compile(r"lecithin\s*\(\s*soya\s*\)"), "soy lecithin"),
    (re.compile(r"lecithin\s+from\s+soy"), "soy lecithin"),
    (re.compile(r"lecithin\s+from\s+soya"), "soy lecithin"),
    (re.compile(r"soy[-\s]?bean\s+lecithin"), "soy lecithin"),
    (re.compile(r"soya[-\s]?bean\s+lecithin"), "soy lecithin"),
]

# Sort keys by length to avoid partial replacements shadowing longer phrases.
_ORDERED_NORMALISATIONS = sorted(
    _BASE_NORMALISATION_MAP.items(), key=lambda item: len(item[0]), reverse=True
)

_ASCII_CLEANER = re.compile(r"[^\x00-\x7F]")
_PUNCT_DIGIT = re.compile(r"[^a-z\s]")
_MULTI_SPACE = re.compile(r"\s+")


def _normalise_synonyms(text: str) -> str:
    """Apply regex and dictionary based substitutions for known variants."""
    for pattern, replacement in _NORMALISATION_REGEX:
        text = pattern.sub(replacement, text)

    for source, target in _ORDERED_NORMALISATIONS:
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return text


@lru_cache(maxsize=1024)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def _strip_unicode_artifacts(text: str) -> str:
    """Normalise unicode text and remove non ASCII artefacts."""
    normalised = unicodedata.normalize("NFKD", text)
    return _ASCII_CLEANER.sub(" ", normalised)


def _filter_tokens(tokens: Iterable[str]) -> list[str]:
    """Remove stopwords and overly short tokens before stemming."""
    filtered = [tok for tok in tokens if tok and tok not in _STOPWORDS]
    return filtered


def preprocess_text(text: str | None) -> str:
    """Standardise raw ingredient text for modelling.

    Steps: lowercase, unicode normalisation, synonym harmonisation, punctuation
    stripping, stopword removal, and stemming. Returns an empty string if the
    input has no informative content after cleaning.
    """
    if text is None:
        return ""

    text = str(text).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    if not text:
        return ""

    text = text.lower()
    text = _strip_unicode_artifacts(text)
    text = _normalise_synonyms(text)
    text = _PUNCT_DIGIT.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()

    if not text:
        return ""

    tokens = text.split()
    tokens = _filter_tokens(tokens)
    if not tokens:
        return ""

    stemmed = [_stem(token) for token in tokens]
    return " ".join(stemmed)
