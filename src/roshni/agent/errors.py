"""Shared error sanitization for user-facing error messages.

Ensures internal details (tracebacks, file paths, API keys) are never
leaked to end users via Telegram, the Router, or any other surface.
"""

from __future__ import annotations


def friendly_error_message(error: Exception) -> str:
    """Return a short, user-readable message for common error types.

    Strips internal details and maps known error classes to safe messages.
    """
    error_type = type(error).__name__.lower()
    error_str = str(error).lower()

    if "budget" in error_str or "budget" in error_type:
        return "Daily token budget exceeded. Try again tomorrow."
    if "notfound" in error_type:
        return "I'm having trouble reaching the AI model. Let me know if this persists."
    if "ratelimit" in error_type:
        return "The AI service is busy right now. Please try again in a moment."
    if "apiconnection" in error_type or "connection" in error_type:
        return "Having trouble connecting. Please check if the service is available."
    if "badrequest" in error_type:
        return "Something went wrong with my request format. This has been logged for investigation."
    if "queue" in error_str and ("full" in error_str or "saturated" in error_str):
        return "I'm handling too many requests right now. Please try again in a moment."
    if "timeout" in error_type or "timeout" in error_str:
        return "The request took too long. Please try again."
    return "Something unexpected happened. Please try again."
