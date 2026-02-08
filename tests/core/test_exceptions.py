"""Tests for roshni.core.exceptions."""

from roshni.core.exceptions import (
    APIError,
    AuthenticationError,
    CacheError,
    ConfigurationError,
    DataProcessingError,
    FileIOError,
    LLMError,
    RoshniError,
    SecretNotFoundError,
)


def test_hierarchy():
    """All exceptions should inherit from RoshniError."""
    for exc_cls in [
        ConfigurationError,
        APIError,
        DataProcessingError,
        FileIOError,
        CacheError,
        AuthenticationError,
        SecretNotFoundError,
    ]:
        assert issubclass(exc_cls, RoshniError)


def test_llm_error_is_api_error():
    assert issubclass(LLMError, APIError)
    assert issubclass(LLMError, RoshniError)


def test_exception_message():
    err = ConfigurationError("missing key: llm.provider")
    assert "missing key" in str(err)


def test_catch_base():
    """Catching RoshniError should catch all subtypes."""
    try:
        raise LLMError("rate limited")
    except RoshniError as e:
        assert "rate limited" in str(e)
