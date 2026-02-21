"""Tests for LLMClient fallback behavior."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# --- litellm mock setup ---
# litellm is an optional dep not installed in dev. We create a mock module
# with the exception classes that client.py references.

_litellm_mock = types.ModuleType("litellm")


class _MockRateLimitError(Exception):
    def __init__(self, message="", llm_provider="", model=""):
        super().__init__(message)


class _MockAPIError(Exception):
    def __init__(self, message="", llm_provider="", model="", status_code=500):
        super().__init__(message)


class _MockAPIConnectionError(Exception):
    def __init__(self, message="", llm_provider="", model=""):
        super().__init__(message)


class _MockBadRequestError(_MockAPIError):
    def __init__(self, message="", llm_provider="", model="", status_code=400):
        super().__init__(message)


class _MockServiceUnavailableError(Exception):
    def __init__(self, message="", llm_provider="", model="", status_code=503):
        super().__init__(message)


class _MockNotFoundError(Exception):
    def __init__(self, message="", llm_provider="", model="", status_code=404):
        super().__init__(message)


class _MockInternalServerError(Exception):
    def __init__(self, message="", llm_provider="", model="", status_code=500):
        super().__init__(message)


_litellm_mock.RateLimitError = _MockRateLimitError
_litellm_mock.APIError = _MockAPIError
_litellm_mock.APIConnectionError = _MockAPIConnectionError
_litellm_mock.BadRequestError = _MockBadRequestError
_litellm_mock.ServiceUnavailableError = _MockServiceUnavailableError
_litellm_mock.NotFoundError = _MockNotFoundError
_litellm_mock.InternalServerError = _MockInternalServerError
_litellm_mock.completion = MagicMock()
_litellm_mock.acompletion = MagicMock()


@pytest.fixture(autouse=True)
def _mock_litellm():
    """Inject our mock litellm into sys.modules for all tests."""
    old = sys.modules.get("litellm")
    sys.modules["litellm"] = _litellm_mock
    yield
    if old is not None:
        sys.modules["litellm"] = old
    else:
        sys.modules.pop("litellm", None)


@pytest.fixture(autouse=True)
def _reset_model_health():
    """Reset model health tracker between tests to prevent state leakage."""
    from roshni.core.llm.model_health import reset_model_health

    reset_model_health()
    yield
    reset_model_health()


def _make_response(content="Hello", prompt_tokens=10, completion_tokens=5):
    """Build a mock litellm response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    # Explicit cache fields — MagicMock auto-creates attrs, which breaks getattr defaults
    usage.cache_creation_input_tokens = 0
    usage.cached_content_token_count = 0
    usage.prompt_tokens_details = None
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock(choices=[choice])
    resp.usage = usage
    return resp


class TestFallbackInit:
    def test_no_fallback_by_default(self):
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-4o-mini")
        assert client.fallback_model is None
        assert client.fallback_provider is None

    def test_fallback_model_set(self):
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-4o-mini", fallback_model="anthropic/claude-sonnet-4-20250514")
        assert client.fallback_model == "anthropic/claude-sonnet-4-20250514"
        assert client.fallback_provider == "anthropic"

    def test_fallback_provider_explicit(self):
        from roshni.core.llm.client import LLMClient

        client = LLMClient(
            model="gpt-4o-mini",
            fallback_model="anthropic/claude-sonnet-4-20250514",
            fallback_provider="anthropic",
        )
        assert client.fallback_provider == "anthropic"

    def test_config_info_includes_fallback(self):
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-4o-mini", fallback_model="deepseek/deepseek-chat")
        info = client.get_config_info()
        assert info["fallback_model"] == "deepseek/deepseek-chat"
        assert info["fallback_provider"] == "deepseek"

    def test_config_info_no_fallback(self):
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-4o-mini")
        info = client.get_config_info()
        assert "fallback_model" not in info


class TestFallbackCompletion:
    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_primary_succeeds_no_fallback_used(self, mock_record, mock_budget):
        """When primary succeeds, fallback is never attempted."""
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-4o-mini", fallback_model="anthropic/claude-sonnet-4-20250514")
        _litellm_mock.completion = MagicMock(return_value=_make_response("Primary OK"))

        messages = [{"role": "user", "content": "hi"}]
        resp = client.completion(messages)

        assert resp.choices[0].message.content == "Primary OK"
        assert _litellm_mock.completion.call_count == 1
        assert _litellm_mock.completion.call_args.kwargs["model"] == "gpt-4o-mini"

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_primary_fails_fallback_succeeds(self, mock_record, mock_budget):
        """When primary fails with a retryable error, fallback fires."""
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-4o-mini", fallback_model="anthropic/claude-sonnet-4-20250514")

        calls = []

        def side_effect(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gpt-4o-mini":
                raise _MockRateLimitError("Rate limited")
            return _make_response("Fallback OK")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)

        messages = [{"role": "user", "content": "hi"}]
        resp = client.completion(messages)

        assert resp.choices[0].message.content == "Fallback OK"
        assert calls == ["gpt-4o-mini", "anthropic/claude-sonnet-4-20250514"]

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_primary_fails_no_fallback_configured(self, mock_record, mock_budget):
        """When primary fails and no fallback is configured, error propagates."""
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-4o-mini")  # no fallback

        _litellm_mock.completion = MagicMock(side_effect=_MockRateLimitError("Rate limited"))

        with pytest.raises(_MockRateLimitError):
            client.completion([{"role": "user", "content": "hi"}])

    @patch("roshni.core.llm.client.check_budget", return_value=(False, 0))
    def test_budget_exceeded_not_retried(self, mock_budget):
        """RuntimeError (budget exceeded) is NOT retried with fallback."""
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-4o-mini", fallback_model="anthropic/claude-sonnet-4-20250514")
        _litellm_mock.completion = MagicMock()  # should never be called

        with pytest.raises(RuntimeError, match="budget exceeded"):
            client.completion([{"role": "user", "content": "hi"}])

        _litellm_mock.completion.assert_not_called()

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_fallback_caps_max_tokens(self, mock_record, mock_budget):
        """Fallback should cap max_tokens to the fallback model's limit."""
        from roshni.core.llm.client import LLMClient

        # gpt-4o has 16384 limit, deepseek-chat has 8192
        client = LLMClient(model="gpt-4o", fallback_model="deepseek/deepseek-chat")

        fallback_kwargs = {}

        def side_effect(**kwargs):
            if kwargs["model"] == "gpt-4o":
                raise _MockAPIError("Server error")
            fallback_kwargs.update(kwargs)
            return _make_response("OK")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)
        client.completion([{"role": "user", "content": "hi"}])

        assert fallback_kwargs["model"] == "deepseek/deepseek-chat"
        assert fallback_kwargs["max_tokens"] <= 8192

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_api_connection_error_triggers_fallback(self, mock_record, mock_budget):
        """APIConnectionError should also trigger fallback."""
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-4o-mini", fallback_model="deepseek/deepseek-chat")

        def side_effect(**kwargs):
            if kwargs["model"] == "gpt-4o-mini":
                raise _MockAPIConnectionError("Connection refused")
            return _make_response("Fallback OK")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)

        resp = client.completion([{"role": "user", "content": "hi"}])
        assert resp.choices[0].message.content == "Fallback OK"

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_fallback_records_usage_with_fallback_provider(self, mock_record, mock_budget):
        """Usage recording should use fallback provider/model when fallback fires."""
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-4o-mini", fallback_model="deepseek/deepseek-chat")

        def side_effect(**kwargs):
            if kwargs["model"] == "gpt-4o-mini":
                raise _MockRateLimitError("Rate limited")
            return _make_response("OK")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)
        client.completion([{"role": "user", "content": "hi"}])

        mock_record.assert_called_once_with(
            input_tokens=10,
            output_tokens=5,
            provider="deepseek",
            model="deepseek/deepseek-chat",
            cache_creation_tokens=0,
            cache_read_tokens=0,
            cost_usd=0.0,
        )


class TestCompatibilityRecovery:
    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_deepseek_reasoner_omits_temperature(self, mock_record, mock_budget):
        from roshni.core.llm.client import LLMClient

        captured = {}

        def side_effect(**kwargs):
            captured.update(kwargs)
            return _make_response("OK")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)
        client = LLMClient(model="deepseek/deepseek-reasoner")
        client.completion([{"role": "user", "content": "hi"}])

        assert "temperature" not in captured

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_bad_request_temperature_retries_without_temperature(self, mock_record, mock_budget):
        from roshni.core.llm.client import LLMClient

        calls = []

        def side_effect(**kwargs):
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _MockBadRequestError("temperature is not supported for this model")
            return _make_response("Recovered")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)
        client = LLMClient(model="gpt-4o-mini")
        resp = client.completion([{"role": "user", "content": "hi"}])

        assert resp.choices[0].message.content == "Recovered"
        assert len(calls) == 2
        assert "temperature" in calls[0]
        assert "temperature" not in calls[1]

    def test_temperature_contract_with_prefixed_and_unprefixed_reasoning_models(self):
        from roshni.core.llm.client import LLMClient

        assert LLMClient._supports_custom_temperature("o3") is False
        assert LLMClient._supports_custom_temperature("openai/o3") is False
        assert LLMClient._supports_custom_temperature("deepseek/deepseek-reasoner") is False
        assert LLMClient._supports_custom_temperature("gpt-5.2-chat-latest") is True


class TestToolCallRepairRecovery:
    """Tests for self-healing tool_call message repair in the recovery loop."""

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_bad_request_tool_call_ids_self_heals(self, mock_record, mock_budget):
        """BadRequest with 'tool_call_ids did not have response' triggers repair and retries."""
        from roshni.core.llm.client import LLMClient

        calls = []

        def side_effect(**kwargs):
            calls.append(dict(kwargs))
            messages = kwargs.get("messages", [])
            # Check if there are orphaned tool_calls without results
            for i, msg in enumerate(messages):
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tc_id = tc["id"]
                        # Check if next message is a tool result for this ID
                        has_result = any(
                            m.get("role") == "tool" and m.get("tool_call_id") == tc_id for m in messages[i + 1 :]
                        )
                        if not has_result:
                            raise _MockBadRequestError("tool_call_ids did not have response messages: ['call_orphan']")
            return _make_response("Recovered after repair")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)

        client = LLMClient(model="gpt-4o-mini")
        # Messages with an orphaned tool_call (missing tool result)
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_orphan", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            {"role": "user", "content": "what happened?"},
        ]

        resp = client.completion(messages)
        assert resp.choices[0].message.content == "Recovered after repair"
        # Should have retried after repairing
        assert len(calls) >= 2

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_bad_request_must_be_followed_by_self_heals(self, mock_record, mock_budget):
        """BadRequest with 'must be followed by' also triggers repair."""
        from roshni.core.llm.client import LLMClient

        attempt = [0]

        def side_effect(**kwargs):
            attempt[0] += 1
            if attempt[0] == 1:
                raise _MockBadRequestError("Messages with tool_calls must be followed by tool results")
            return _make_response("Fixed")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)

        client = LLMClient(model="gpt-4o-mini")
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
            },
        ]

        resp = client.completion(messages)
        assert resp.choices[0].message.content == "Fixed"
        assert attempt[0] == 2


class TestNotFoundRecovery:
    """Tests for model name auto-correction on NotFoundError."""

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_not_found_error_resolves_model_name(self, mock_record, mock_budget):
        """NotFound with 'claude-haiku-4' auto-corrects to catalog model and retries."""
        from roshni.core.llm.client import LLMClient

        calls = []

        def side_effect(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "claude-haiku-4":
                raise _MockNotFoundError("model: claude-haiku-4 not found")
            return _make_response("Resolved OK")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)

        client = LLMClient(model="claude-haiku-4")
        messages = [{"role": "user", "content": "hi"}]
        resp = client.completion(messages)

        assert resp.choices[0].message.content == "Resolved OK"
        assert calls[0] == "claude-haiku-4"
        # Second call should use the resolved model name
        assert calls[1] != "claude-haiku-4"
        assert "claude-haiku" in calls[1]

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_not_found_falls_back_when_no_catalog_match(self, mock_record, mock_budget):
        """NotFound with unknown model falls back to fallback_model."""
        from roshni.core.llm.client import LLMClient

        calls = []

        def side_effect(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "totally-fake-model":
                raise _MockNotFoundError("model: totally-fake-model not found")
            return _make_response("Fallback OK")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)

        client = LLMClient(model="totally-fake-model", fallback_model="deepseek/deepseek-chat")
        messages = [{"role": "user", "content": "hi"}]
        resp = client.completion(messages)

        assert resp.choices[0].message.content == "Fallback OK"
        # Should try fake model, then switch to fallback within recovery loop
        assert "totally-fake-model" in calls
        assert "deepseek/deepseek-chat" in calls

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_not_found_no_match_no_fallback_raises(self, mock_record, mock_budget):
        """NotFound with no catalog match and no fallback raises."""
        from roshni.core.llm.client import LLMClient

        _litellm_mock.completion = MagicMock(side_effect=_MockNotFoundError("model: totally-fake-model not found"))

        client = LLMClient(model="totally-fake-model")  # no fallback
        with pytest.raises(_MockNotFoundError):
            client.completion([{"role": "user", "content": "hi"}])


class TestAsyncFallbackAfterBadRequest:
    """Tests that async path allows fallback after unrecoverable BadRequest."""

    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    @pytest.mark.asyncio
    async def test_async_bad_request_can_fallback(self, mock_record, mock_budget):
        """Verify acompletion allows fallback after unrecoverable BadRequest."""
        from roshni.core.llm.client import LLMClient

        calls = []

        async def async_side_effect(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gpt-4o-mini":
                raise _MockBadRequestError("Some unrecoverable bad request error")
            return _make_response("Fallback OK")

        _litellm_mock.acompletion = MagicMock(side_effect=async_side_effect)

        client = LLMClient(model="gpt-4o-mini", fallback_model="anthropic/claude-sonnet-4-6")
        messages = [{"role": "user", "content": "hi"}]
        resp = await client.acompletion(messages)

        # Should have fallen back to the fallback model
        assert resp.choices[0].message.content == "Fallback OK"
        assert "gpt-4o-mini" in calls
        assert "anthropic/claude-sonnet-4-6" in calls


class TestAuthProfileFailover:
    @patch("roshni.core.llm.client.check_budget", return_value=(True, 100_000))
    @patch("roshni.core.llm.client.record_usage")
    def test_tries_all_auth_profiles_before_fallback(self, mock_record, mock_budget):
        from roshni.core.llm.auth_profiles import AuthProfile
        from roshni.core.llm.client import LLMClient

        profiles = [
            AuthProfile(name="p1", provider="anthropic", api_key="key-1"),
            AuthProfile(name="p2", provider="anthropic", api_key="key-2"),
            AuthProfile(name="p3", provider="anthropic", api_key="key-3"),
        ]
        attempted_keys = []

        def side_effect(**kwargs):
            attempted_keys.append(kwargs.get("api_key"))
            if kwargs.get("api_key") != "key-3":
                raise _MockRateLimitError("Rate limited")
            return _make_response("Recovered with profile 3")

        _litellm_mock.completion = MagicMock(side_effect=side_effect)

        client = LLMClient(
            model="anthropic/claude-sonnet-4-20250514",
            auth_profiles=profiles,
        )
        resp = client.completion([{"role": "user", "content": "hi"}])

        assert resp.choices[0].message.content == "Recovered with profile 3"
        assert attempted_keys == ["key-1", "key-2", "key-3"]
