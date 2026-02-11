"""Tests for the Advisor/AfterChatHook protocols and built-in implementations."""

from unittest.mock import MagicMock, patch

import pytest

from roshni.agent.advisor import Advisor, AfterChatHook, FunctionAdvisor, FunctionAfterChatHook
from roshni.agent.advisors import MemoryAdvisor, SystemHealthAdvisor
from roshni.agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from roshni.agent.hooks import LoggingHook, MemoryExtractionHook, MetricsHook

# ── Protocol conformance ──────────────────────────────────────────────


class TestProtocolConformance:
    def test_function_advisor_is_advisor(self):
        fa = FunctionAdvisor("test", lambda: "hi")
        assert isinstance(fa, Advisor)

    def test_function_after_chat_hook_is_hook(self):
        fh = FunctionAfterChatHook("test", lambda: None)
        assert isinstance(fh, AfterChatHook)

    def test_memory_advisor_is_advisor(self):
        mm = MagicMock()
        ma = MemoryAdvisor(mm)
        assert isinstance(ma, Advisor)

    def test_system_health_advisor_is_advisor(self):
        sha = SystemHealthAdvisor()
        assert isinstance(sha, Advisor)

    def test_logging_hook_is_hook(self):
        lh = LoggingHook(lambda m, r, t: None)
        assert isinstance(lh, AfterChatHook)

    def test_metrics_hook_is_hook(self):
        cb = CircuitBreaker()
        mh = MetricsHook(cb)
        assert isinstance(mh, AfterChatHook)

    def test_memory_extraction_hook_is_hook(self):
        meh = MemoryExtractionHook(MagicMock(), MagicMock())
        assert isinstance(meh, AfterChatHook)


# ── FunctionAdvisor ───────────────────────────────────────────────────


class TestFunctionAdvisor:
    def test_no_args_function(self):
        fa = FunctionAdvisor("simple", lambda: "context")
        assert fa.advise(message="hi") == "context"

    def test_message_arg(self):
        fa = FunctionAdvisor("msg", lambda message: f"got: {message}")
        assert fa.advise(message="hello") == "got: hello"

    def test_message_and_channel(self):
        def fn(message, channel):
            return f"{message}@{channel}"

        fa = FunctionAdvisor("both", fn)
        assert fa.advise(message="hi", channel="telegram") == "hi@telegram"

    def test_channel_only(self):
        fa = FunctionAdvisor("ch", lambda channel: f"ch={channel}")
        assert fa.advise(message="ignored", channel="web") == "ch=web"

    def test_name_attribute(self):
        fa = FunctionAdvisor("my_advisor", lambda: "")
        assert fa.name == "my_advisor"


# ── FunctionAfterChatHook ────────────────────────────────────────────


class TestFunctionAfterChatHook:
    def test_receives_message_and_response(self):
        captured = {}

        def fn(message, response):
            captured["msg"] = message
            captured["resp"] = response

        hook = FunctionAfterChatHook("test", fn)
        hook.run(message="hi", response="bye", tool_calls=[])
        assert captured == {"msg": "hi", "resp": "bye"}

    def test_receives_all_kwargs(self):
        captured = {}

        def fn(message, response, tool_calls, channel):
            captured.update(message=message, response=response, tool_calls=tool_calls, channel=channel)

        hook = FunctionAfterChatHook("full", fn)
        hook.run(message="a", response="b", tool_calls=[{"name": "x"}], channel="tg")
        assert captured["tool_calls"] == [{"name": "x"}]
        assert captured["channel"] == "tg"

    def test_no_args_function(self):
        called = []
        hook = FunctionAfterChatHook("noop", lambda: called.append(True))
        hook.run(message="a", response="b", tool_calls=[])
        assert called == [True]


# ── MemoryAdvisor ────────────────────────────────────────────────────


class TestMemoryAdvisor:
    def test_returns_memory_and_daily(self):
        mm = MagicMock()
        mm.get_context.return_value = "[MEMORY]\nstuff\n[/MEMORY]"
        mm.get_daily_context.return_value = "[DAILY NOTES]\nnotes\n[/DAILY NOTES]"

        ma = MemoryAdvisor(mm)
        result = ma.advise(message="hi")
        assert "[MEMORY]" in result
        assert "[DAILY NOTES]" in result

    def test_returns_empty_when_no_memory(self):
        mm = MagicMock()
        mm.get_context.return_value = ""
        mm.get_daily_context.return_value = ""

        ma = MemoryAdvisor(mm)
        assert ma.advise(message="hi") == ""

    def test_returns_only_memory_when_no_daily(self):
        mm = MagicMock()
        mm.get_context.return_value = "[MEMORY]\nstuff\n[/MEMORY]"
        mm.get_daily_context.return_value = ""

        ma = MemoryAdvisor(mm)
        result = ma.advise(message="hi")
        assert "[MEMORY]" in result
        assert "[DAILY NOTES]" not in result


# ── SystemHealthAdvisor ──────────────────────────────────────────────


class TestSystemHealthAdvisor:
    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_returns_empty_when_healthy(self, _mock):
        sha = SystemHealthAdvisor()
        assert sha.advise(message="hi") == ""

    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.75)
    def test_surfaces_moderate_budget_pressure(self, _mock):
        sha = SystemHealthAdvisor(budget_threshold=0.6)
        result = sha.advise(message="hi")
        assert "[BUDGET]" in result
        assert "75%" in result
        assert "efficiency" in result.lower()

    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.95)
    def test_surfaces_high_budget_pressure(self, _mock):
        sha = SystemHealthAdvisor(budget_threshold=0.6)
        result = sha.advise(message="hi")
        assert "[BUDGET]" in result
        assert "95%" in result
        assert "concise" in result.lower()

    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.5)
    def test_no_budget_below_threshold(self, _mock):
        sha = SystemHealthAdvisor(budget_threshold=0.6)
        assert sha.advise(message="hi") == ""

    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_surfaces_broken_circuit(self, _mock):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, open_duration=60.0))
        # Trip the circuit
        cb.record("search", success=False)
        cb.record("search", success=False)
        assert not cb.is_available("search")

        sha = SystemHealthAdvisor(circuit_breaker=cb)
        result = sha.advise(message="hi")
        assert "[SERVICES DOWN]" in result
        assert "search" in result

    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_surfaces_degraded_service(self, _mock):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=10))  # High threshold so it doesn't trip
        # Record mostly failures but not enough to trip
        cb.record("weather", success=False)
        cb.record("weather", success=False)
        cb.record("weather", success=True)  # 66% failure rate, 3+ calls

        sha = SystemHealthAdvisor(circuit_breaker=cb)
        result = sha.advise(message="hi")
        assert "[DEGRADED]" in result
        assert "weather" in result


# ── MetricsHook ──────────────────────────────────────────────────────


class TestMetricsHook:
    def test_records_successful_tool_calls(self):
        cb = CircuitBreaker()
        hook = MetricsHook(cb)
        hook.run(
            message="hi",
            response="done",
            tool_calls=[{"name": "search", "result": "found 3 items"}],
        )
        status = cb.get_status()
        assert "search" in status
        assert status["search"]["successes"] == 1

    def test_records_failed_tool_calls(self):
        cb = CircuitBreaker()
        hook = MetricsHook(cb)
        hook.run(
            message="hi",
            response="sorry",
            tool_calls=[{"name": "search", "result": "Error: connection timeout"}],
        )
        status = cb.get_status()
        assert status["search"]["failures"] == 1

    def test_records_multiple_tools(self):
        cb = CircuitBreaker()
        hook = MetricsHook(cb)
        hook.run(
            message="hi",
            response="done",
            tool_calls=[
                {"name": "search", "result": "ok"},
                {"name": "write", "result": "Error: denied"},
            ],
        )
        status = cb.get_status()
        assert status["search"]["successes"] == 1
        assert status["write"]["failures"] == 1


# ── LoggingHook ──────────────────────────────────────────────────────


class TestLoggingHook:
    def test_calls_callback(self):
        captured = {}

        def cb(msg, resp, tools):
            captured.update(msg=msg, resp=resp, tools=tools)

        hook = LoggingHook(cb)
        hook.run(message="hi", response="bye", tool_calls=[{"name": "x"}])
        assert captured["msg"] == "hi"
        assert captured["resp"] == "bye"


# ── MemoryExtractionHook ─────────────────────────────────────────────


class TestMemoryExtractionHook:
    def test_skips_when_no_trigger(self):
        mm = MagicMock()
        mm.detect_trigger.return_value = False
        llm = MagicMock()

        hook = MemoryExtractionHook(mm, llm)
        hook.run(message="hello", response="hi", tool_calls=[])
        llm.completion.assert_not_called()

    def test_skips_when_save_memory_already_called(self):
        mm = MagicMock()
        mm.detect_trigger.return_value = True
        llm = MagicMock()

        hook = MemoryExtractionHook(mm, llm)
        hook.run(message="always do X", response="ok", tool_calls=[{"name": "save_memory"}])
        llm.completion.assert_not_called()

    def test_extracts_and_saves_preference(self):
        mm = MagicMock()
        mm.detect_trigger.return_value = True

        resp_mock = MagicMock()
        resp_mock.choices = [MagicMock()]
        resp_mock.choices[0].message.content = "Use dark mode"
        llm = MagicMock()
        llm.completion.return_value = resp_mock

        hook = MemoryExtractionHook(mm, llm)
        hook.run(message="I always prefer dark mode", response="ok", tool_calls=[])
        mm.save.assert_called_once_with("preferences", "Use dark mode")

    def test_extracts_and_saves_decision(self):
        mm = MagicMock()
        mm.detect_trigger.return_value = True

        resp_mock = MagicMock()
        resp_mock.choices = [MagicMock()]
        resp_mock.choices[0].message.content = "Switch to PostgreSQL"
        llm = MagicMock()
        llm.completion.return_value = resp_mock

        hook = MemoryExtractionHook(mm, llm)
        hook.run(message="remember we switched to PostgreSQL", response="ok", tool_calls=[])
        mm.save.assert_called_once_with("decisions", "Switch to PostgreSQL")

    def test_skips_none_extraction(self):
        mm = MagicMock()
        mm.detect_trigger.return_value = True

        resp_mock = MagicMock()
        resp_mock.choices = [MagicMock()]
        resp_mock.choices[0].message.content = "NONE"
        llm = MagicMock()
        llm.completion.return_value = resp_mock

        hook = MemoryExtractionHook(mm, llm)
        hook.run(message="remember to buy milk", response="ok", tool_calls=[])
        mm.save.assert_not_called()


# ── Advisor failure isolation ────────────────────────────────────────


class TestAdvisorFailureIsolation:
    """Advisors that throw should not break other advisors."""

    def test_failing_advisor_returns_error_gracefully(self):
        """FunctionAdvisor doesn't catch errors — that's DefaultAgent's job.
        But we test that the protocol works correctly when called normally."""

        def bad_fn():
            raise RuntimeError("boom")

        fa = FunctionAdvisor("bad", bad_fn)
        with pytest.raises(RuntimeError, match="boom"):
            fa.advise(message="hi")

    def test_hook_failure_doesnt_propagate(self):
        """FunctionAfterChatHook doesn't catch — DefaultAgent catches in _fire_after_chat_hooks."""

        def bad_fn():
            raise RuntimeError("boom")

        hook = FunctionAfterChatHook("bad", bad_fn)
        with pytest.raises(RuntimeError, match="boom"):
            hook.run(message="hi", response="bye", tool_calls=[])
