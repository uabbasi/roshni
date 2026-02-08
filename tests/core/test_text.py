"""Tests for roshni.core.utils.text."""

from roshni.core.utils.text import clean_text, extract_keywords, normalize_text, truncate_text


def test_clean_text_basic():
    assert clean_text("Hello, World!") == "hello world"


def test_clean_text_stop_words():
    result = clean_text("the cat sat on the mat", stop_words={"the", "on"})
    assert result == "cat sat mat"


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_normalize_text():
    assert normalize_text("hello\r\nworld") == "hello world"


def test_normalize_text_multiple_newlines():
    result = normalize_text("a\n\n\n\nb")
    assert "\n\n\n" not in result


def test_extract_keywords():
    keywords = extract_keywords("The quick brown fox")
    assert "quick" in keywords
    assert "brown" in keywords
    # "The" is 3 chars -> included (lowercased)
    assert "the" in keywords


def test_extract_keywords_min_length():
    keywords = extract_keywords("I am ok", min_word_length=3)
    assert "i" not in keywords
    assert "am" not in keywords


def test_truncate_text():
    assert truncate_text("short") == "short"
    assert truncate_text("a" * 200, max_length=10) == "aaaaaaa..."
    assert truncate_text("") == ""
