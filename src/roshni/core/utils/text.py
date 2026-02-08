"""Text processing utilities: cleaning, normalization, keyword extraction."""

import re
import string


def clean_text(text: str, stop_words: set[str] | None = None) -> str:
    """Clean text for ML/analysis: lowercase, strip punctuation, remove stop words."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    if stop_words:
        text = " ".join(w for w in text.split() if w not in stop_words)
    return text


def normalize_text(text: str) -> str:
    """Normalize whitespace and newlines."""
    if not text or not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(text: str, min_word_length: int = 3) -> list[str]:
    """Extract keywords (words above min length, stripped of punctuation)."""
    if not text or not isinstance(text, str):
        return []
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if len(w) >= min_word_length]


def truncate_text(text: str, max_length: int = 100, ellipsis: str = "...") -> str:
    """Truncate text to max_length, appending ellipsis if truncated."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(ellipsis)] + ellipsis
