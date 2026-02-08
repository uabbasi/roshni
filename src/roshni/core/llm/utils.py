"""Utility functions for LLM response handling."""

from typing import Any

from loguru import logger


def extract_text_from_response(content: Any, log_failures: bool = False) -> str:
    """Extract text content from an LLM response.

    LiteLLM normalises responses to OpenAI format, so most of the time
    ``content`` is just a string.  This helper also handles edge cases
    (Gemini content-block lists, objects with ``.parts``, etc.) so callers
    don't have to.
    """
    result = _extract_text_impl(content)

    if not result and log_failures:
        logger.warning(f"Empty response extraction. Type: {type(content).__name__}, Value: {repr(content)[:500]}")

    return result


def _extract_text_impl(content: Any) -> str:
    if isinstance(content, str):
        return content

    # Objects with .parts (Gemini native)
    if hasattr(content, "parts"):
        text_parts = []
        for part in content.parts:
            if hasattr(part, "text"):
                text_parts.append(part.text)
            elif isinstance(part, str):
                text_parts.append(part)
        if text_parts:
            return "".join(text_parts)

    if hasattr(content, "text"):
        return content.text

    # List format (content blocks)
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    text_parts.append(part["text"])
                elif part.get("type") in ("tool_use", "tool_result", "thinking"):
                    continue
                elif "text" in part:
                    text_parts.append(part["text"])
            elif hasattr(part, "type") and hasattr(part, "text"):
                if getattr(part, "type", None) == "text":
                    text_parts.append(part.text)
            elif hasattr(part, "text"):
                text_parts.append(part.text)
        return "".join(text_parts)

    return str(content) if content else ""
