"""Response continuation utilities for handling truncated LLM responses.

This module provides:
- Detection of truncated/incomplete responses
- Automatic continuation logic for completing responses
- Mixin class for adding continuation capability to any LLM orchestrator

The continuation logic is generic and can work with any LLM backend by
accepting a callable for the actual LLM invocation.

Ported from weeklies' core/llm/response_continuation.py — roshni is now
the canonical source for these utilities.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ContinuationConfig:
    """Configuration for response continuation behavior."""

    max_total_attempts: int = 3
    """Maximum number of LLM calls (including initial + continuations)."""

    continuation_overlap_tokens: int = 50
    """Approximate tokens of context to include from partial response."""

    min_continuation_growth: int = 50
    """Minimum characters a continuation must add to be considered useful."""


@dataclass
class ContinuationResult:
    """Result of a continuation attempt or complete response generation."""

    response: str
    """The final (possibly continued) response."""

    total_time: float
    """Total time across all LLM calls."""

    continuation_count: int = 0
    """Number of continuation attempts made."""

    was_truncated: bool = False
    """Whether the initial response was detected as truncated."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata about the continuation process."""


def is_response_truncated(response: str) -> bool:
    """
    Detect if a response appears to be truncated/incomplete.

    Uses conservative indicators to avoid false positives. Checks for:
    - Very short responses (< 30 chars)
    - Ellipsis endings
    - Incomplete sentence patterns
    - Dangling connectors (and, or, but)
    - Unclosed markdown formatting

    Args:
        response: The response text to check

    Returns:
        True if the response appears truncated, False otherwise
    """
    response = response.strip()

    # Very short responses are likely truncated
    if len(response) < 30:
        return True

    # Clear truncation indicators
    if response.endswith(("...", "..", "…")):
        return True

    # Check if response ends mid-sentence (very conservatively)
    lines = response.split("\n")
    if lines:
        last_line = lines[-1].strip()

        # Only flag if last line is clearly incomplete
        incomplete_patterns = [
            # Ends with comma followed by very short fragment
            last_line.endswith(",") and len(last_line) < 15,
            # Ends with "and" or "or" as a dangling connector (but only if very short)
            last_line.endswith((" and", " or")) and len(last_line) < 20,
            # Ends with incomplete sentence starters that are very short
            (
                last_line.startswith(("In", "The", "This", "That", "For", "However", "Therefore"))
                and len(last_line) < 12
                and not last_line.endswith((".", "!", "?"))
            ),
            # Ends with clearly incomplete markdown
            last_line.endswith("**") and last_line.count("**") % 2 == 1,  # Unclosed bold
            last_line.endswith("*")
            and last_line.count("*") % 2 == 1
            and not last_line.endswith("**"),  # Unclosed italic
        ]

        if any(incomplete_patterns):
            return True

    # Check for abrupt cutoff in the middle of common sentence structures
    # Only if the response doesn't end with proper punctuation
    if not response.endswith((".", "!", "?", ":", ")", "]", "}", '"', "'")):
        # Look for patterns that suggest mid-sentence cutoff
        concerning_endings = [
            response.endswith((" the", " a", " an", " of", " to", " for", " in", " on", " at")),
            response.endswith((" is", " are", " was", " were", " will", " would", " should")),
            response.endswith((" and", " or", " but")) and len(response.split()[-1]) < 5,
        ]

        if any(concerning_endings):
            return True

    return False


def build_continuation_prompt(
    original_query: str,
    partial_response: str,
    context: str | None = None,
    overlap_lines: int = 5,
) -> str:
    """
    Build a continuation prompt from a partial response.

    Args:
        original_query: The original user query
        partial_response: The response so far
        context: Optional document context to include
        overlap_lines: Number of lines from partial response to include

    Returns:
        Prompt string for continuation
    """
    # Get last N lines for context
    response_lines = partial_response.split("\n")
    context_lines = response_lines[-overlap_lines:] if len(response_lines) > overlap_lines else response_lines
    response_context = "\n".join(context_lines)

    if context:
        return (
            f"Original Query: {original_query}\n\n"
            f"Document Context:\n{context}\n\n"
            f"Partial Response So Far (last part):\n{response_context}\n\n"
            f"Please continue the response from where it left off."
        )
    else:
        return (
            f"Original Query: {original_query}\n\n"
            f"Partial Response So Far (last part):\n{response_context}\n\n"
            f"Please continue the response from where it left off."
        )


def merge_responses(partial: str, continuation: str) -> str:
    """
    Merge a partial response with its continuation.

    Handles spacing/newline issues at the join point.

    Args:
        partial: The original partial response
        continuation: The continuation text

    Returns:
        Merged response string
    """
    if not partial:
        return continuation
    if not continuation:
        return partial

    # Add appropriate spacing if needed
    if not partial.endswith(("\n", " ")):
        return partial + " " + continuation
    return partial + continuation


class ResponseContinuationMixin:
    """
    Mixin class for adding response continuation capability.

    This mixin provides generic continuation logic that can be used with
    any LLM backend. The actual LLM invocation is handled by a callable
    passed to the methods.

    Example usage:
        class MyOrchestrator(ResponseContinuationMixin):
            def generate_response(self, query: str, context: str) -> str:
                def llm_call(prompt: str) -> tuple[str, float]:
                    return self.llm.chat(prompt)

                result = self.generate_with_continuation(
                    initial_prompt=self.build_prompt(query, context),
                    llm_call=llm_call,
                    continuation_builder=lambda partial: self.build_continuation_prompt(
                        query, partial, context
                    ),
                )
                return result.response
    """

    continuation_config: ContinuationConfig | None = None

    def get_continuation_config(self) -> ContinuationConfig:
        """Get continuation config, creating default if not set."""
        if self.continuation_config is None:
            self.continuation_config = ContinuationConfig()
        return self.continuation_config

    def generate_with_continuation(
        self,
        initial_prompt: str,
        llm_call: Callable[[str], tuple[str, float]],
        continuation_builder: Callable[[str], str],
        config: ContinuationConfig | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> ContinuationResult:
        """
        Generate a response with automatic continuation if truncated.

        Args:
            initial_prompt: The initial prompt to send to the LLM
            llm_call: Callable that takes a prompt and returns (response, time)
            continuation_builder: Callable that takes partial response and returns
                                  a continuation prompt
            config: Optional override for continuation config
            progress_callback: Optional callback for progress updates

        Returns:
            ContinuationResult with final response and metadata
        """
        cfg = config or self.get_continuation_config()

        # Make initial call
        response, initial_time = llm_call(initial_prompt)
        total_time = initial_time

        if progress_callback:
            progress_callback("Checking response completeness...")

        logger.debug(f"Initial response length: {len(response)} chars, truncated: {is_response_truncated(response)}")

        was_truncated = is_response_truncated(response)
        continuation_count = 0
        max_continuations = cfg.max_total_attempts - 1  # Subtract for initial call

        while is_response_truncated(response) and continuation_count < max_continuations:
            if progress_callback:
                progress_callback(f"Continuation {continuation_count + 1}/{max_continuations}...")

            logger.debug(f"Attempting continuation {continuation_count + 1}/{max_continuations}")

            # Build continuation prompt and call LLM
            continuation_prompt = continuation_builder(response)
            continuation, continuation_time = llm_call(continuation_prompt)
            total_time += continuation_time

            # Merge responses
            continued_response = merge_responses(response, continuation)

            # Check if continuation actually added meaningful content
            response_growth = len(continued_response) - len(response)
            if response_growth < cfg.min_continuation_growth:
                logger.debug(f"Continuation added only {response_growth} chars, stopping")
                if progress_callback:
                    progress_callback("Response completion finished")
                break

            response = continued_response
            continuation_count += 1
            logger.debug(
                f"After continuation {continuation_count}: length={len(response)}, "
                f"truncated={is_response_truncated(response)}"
            )

        if continuation_count > 0:
            logger.debug(f"Completed with {continuation_count} continuations, final length: {len(response)}")
            if progress_callback:
                progress_callback(f"Response completed with {continuation_count} continuations")
        else:
            if progress_callback:
                progress_callback("Response complete")

        return ContinuationResult(
            response=response,
            total_time=total_time,
            continuation_count=continuation_count,
            was_truncated=was_truncated,
            metadata={
                "initial_time": initial_time,
                "final_length": len(response),
            },
        )
