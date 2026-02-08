"""Tests for core.llm.response_continuation — truncation detection, continuation, merging."""

from roshni.core.llm.response_continuation import (
    ContinuationConfig,
    ContinuationResult,
    ResponseContinuationMixin,
    build_continuation_prompt,
    is_response_truncated,
    merge_responses,
)


class TestIsResponseTruncated:
    def test_complete_response(self):
        assert not is_response_truncated("This is a complete response with proper ending.")

    def test_short_response(self):
        # Very short responses are flagged
        assert is_response_truncated("Hi")

    def test_ellipsis_ending(self):
        assert is_response_truncated("This response was cut off because of the limit...")
        assert is_response_truncated("This response was cut off\u2026")  # Unicode ellipsis

    def test_dangling_article(self):
        assert is_response_truncated("The analysis shows that the results indicate the")

    def test_dangling_preposition(self):
        assert is_response_truncated("This feature is important for")

    def test_dangling_verb(self):
        assert is_response_truncated("The system components are")

    def test_proper_ending_not_truncated(self):
        assert not is_response_truncated("The analysis is complete and all items are accounted for.")
        assert not is_response_truncated("Here is the full list of items!")
        assert not is_response_truncated("What do you think about this approach?")
        # Short strings (< 30 chars) are always flagged as truncated, even with proper endings
        assert is_response_truncated('She said "hello"')
        # But longer quoted strings are fine
        assert not is_response_truncated('She said "hello" and then walked away from the conversation.')

    def test_code_block_not_false_positive(self):
        # Short code blocks trip the < 30 char heuristic; longer ones are fine
        assert not is_response_truncated("Here is the code example:\n```python\nprint('hello world')\n```")

    def test_parenthetical_ending(self):
        assert not is_response_truncated("This is documented (see RFC 1234)")


class TestBuildContinuationPrompt:
    def test_basic_prompt(self):
        prompt = build_continuation_prompt(
            original_query="What is Python?",
            partial_response="Python is a programming language that",
        )
        assert "What is Python?" in prompt
        assert "Python is a programming language that" in prompt
        assert "continue" in prompt.lower()

    def test_with_context(self):
        prompt = build_continuation_prompt(
            original_query="Summarize this",
            partial_response="The document discusses",
            context="Python is great.",
        )
        assert "Summarize this" in prompt
        assert "Python is great." in prompt
        assert "The document discusses" in prompt

    def test_overlap_lines(self):
        partial = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7"
        prompt = build_continuation_prompt("query", partial, overlap_lines=3)
        # Should include last 3 lines
        assert "Line 5" in prompt
        assert "Line 6" in prompt
        assert "Line 7" in prompt


class TestMergeResponses:
    def test_basic_merge(self):
        result = merge_responses("Hello world", "and goodbye")
        assert result == "Hello world and goodbye"

    def test_already_spaced(self):
        result = merge_responses("Hello world ", "and goodbye")
        assert result == "Hello world and goodbye"

    def test_newline_ending(self):
        result = merge_responses("Hello world\n", "and goodbye")
        assert result == "Hello world\nand goodbye"

    def test_empty_partial(self):
        assert merge_responses("", "continuation") == "continuation"

    def test_empty_continuation(self):
        assert merge_responses("partial", "") == "partial"


class TestContinuationConfig:
    def test_defaults(self):
        cfg = ContinuationConfig()
        assert cfg.max_total_attempts == 3
        assert cfg.continuation_overlap_tokens == 50
        assert cfg.min_continuation_growth == 50

    def test_custom_values(self):
        cfg = ContinuationConfig(max_total_attempts=5, min_continuation_growth=100)
        assert cfg.max_total_attempts == 5
        assert cfg.min_continuation_growth == 100


class TestContinuationResult:
    def test_basic_result(self):
        result = ContinuationResult(response="Hello", total_time=1.5)
        assert result.response == "Hello"
        assert result.total_time == 1.5
        assert result.continuation_count == 0
        assert result.was_truncated is False
        assert result.metadata == {}


class TestResponseContinuationMixin:
    def test_no_continuation_needed(self):
        class TestOrchestrator(ResponseContinuationMixin):
            pass

        orchestrator = TestOrchestrator()

        # LLM returns a complete response
        def llm_call(prompt: str) -> tuple[str, float]:
            return "This is a complete response with proper ending.", 0.5

        result = orchestrator.generate_with_continuation(
            initial_prompt="Tell me something.",
            llm_call=llm_call,
            continuation_builder=lambda partial: f"Continue: {partial}",
        )

        assert result.continuation_count == 0
        assert result.total_time == 0.5
        assert "complete response" in result.response

    def test_continuation_triggered(self):
        class TestOrchestrator(ResponseContinuationMixin):
            pass

        orchestrator = TestOrchestrator()
        call_count = 0

        def llm_call(prompt: str) -> tuple[str, float]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call returns truncated response (ends with dangling article)
                return "The analysis shows that the data indicates the", 0.5
            else:
                # Continuation must exceed min_continuation_growth (50 chars)
                return ("results are fully consistent with our original hypothesis and expectations."), 0.3

        result = orchestrator.generate_with_continuation(
            initial_prompt="Analyze the data.",
            llm_call=llm_call,
            continuation_builder=lambda partial: f"Continue: {partial}",
        )

        assert result.continuation_count == 1
        assert result.was_truncated is True
        assert "hypothesis" in result.response

    def test_max_attempts_respected(self):
        class TestOrchestrator(ResponseContinuationMixin):
            pass

        orchestrator = TestOrchestrator()
        call_count = 0

        def llm_call(prompt: str) -> tuple[str, float]:
            nonlocal call_count
            call_count += 1
            # Always return truncated response
            return f"Part {call_count} of the analysis shows the", 0.2

        config = ContinuationConfig(max_total_attempts=3)
        orchestrator.generate_with_continuation(
            initial_prompt="Analyze.",
            llm_call=llm_call,
            continuation_builder=lambda partial: f"Continue: {partial}",
            config=config,
        )

        # max_total_attempts=3 means 1 initial + 2 continuations max
        assert call_count <= 3

    def test_min_growth_stops_continuation(self):
        class TestOrchestrator(ResponseContinuationMixin):
            pass

        orchestrator = TestOrchestrator()
        call_count = 0

        def llm_call(prompt: str) -> tuple[str, float]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "The analysis shows the", 0.5
            else:
                # Return very small continuation (below min_continuation_growth)
                return "end.", 0.1

        config = ContinuationConfig(min_continuation_growth=100)
        orchestrator.generate_with_continuation(
            initial_prompt="Analyze.",
            llm_call=llm_call,
            continuation_builder=lambda partial: f"Continue: {partial}",
            config=config,
        )

        # Should stop after small continuation
        assert call_count == 2

    def test_progress_callback(self):
        class TestOrchestrator(ResponseContinuationMixin):
            pass

        orchestrator = TestOrchestrator()
        progress_messages: list[str] = []

        def llm_call(prompt: str) -> tuple[str, float]:
            return "This is a complete response with proper ending.", 0.5

        orchestrator.generate_with_continuation(
            initial_prompt="Tell me something.",
            llm_call=llm_call,
            continuation_builder=lambda partial: f"Continue: {partial}",
            progress_callback=progress_messages.append,
        )

        assert len(progress_messages) >= 1
        assert any("complete" in msg.lower() for msg in progress_messages)

    def test_get_continuation_config_default(self):
        class TestOrchestrator(ResponseContinuationMixin):
            pass

        orchestrator = TestOrchestrator()
        config = orchestrator.get_continuation_config()
        assert isinstance(config, ContinuationConfig)
        assert config.max_total_attempts == 3
