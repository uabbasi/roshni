"""AgentSDKAgent — agent backed by Claude Agent SDK.

Routes Claude model usage through the claude-agent-sdk package,
which wraps Claude Code's agentic capabilities including
tool use via in-process MCP servers.

Requires:
    pip install claude-agent-sdk

The Claude Agent SDK wraps the Claude Code CLI.  Authentication is
inherited from Claude Code's own login — typically an OAuth token
obtained via ``claude login``.  No ``ANTHROPIC_API_KEY`` is needed
unless you explicitly want to use an API key instead.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Callable
from time import time
from typing import Any

from loguru import logger

from roshni.agent.base import BaseAgent, ChatResult
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager

# ---------------------------------------------------------------------------
# Async/sync bridge
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from synchronous code.

    Creates a new event loop if none is running, otherwise
    delegates to a background thread to avoid nested-loop issues.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside a running loop — execute in a fresh thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

_JSON_TYPE_TO_PYTHON = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _build_mcp_server(tools: list[ToolDefinition]) -> Any:
    """Convert roshni :class:`ToolDefinition` objects to an in-process MCP server.

    Each tool's JSON Schema parameters are simplified to a Python-type mapping
    expected by the ``@tool`` decorator from ``claude_agent_sdk``.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    sdk_tools = []
    for td in tools:
        props = td.parameters.get("properties", {})
        schema: dict[str, type] = {}
        for param_name, param_schema in props.items():
            json_type = param_schema.get("type", "string")
            schema[param_name] = _JSON_TYPE_TO_PYTHON.get(json_type, str)

        # Capture *td.function* in a default arg to avoid late-binding closure issues
        fn = td.function

        @tool(td.name, td.description, schema)
        async def _wrapper(args: dict[str, Any], _fn: Callable[..., str] = fn) -> dict[str, Any]:
            try:
                result = _fn(**args)
                return {"content": [{"type": "text", "text": str(result)}]}
            except Exception as exc:
                return {"content": [{"type": "text", "text": f"Error: {exc}"}]}

        sdk_tools.append(_wrapper)

    return create_sdk_mcp_server(
        name="roshni-tools",
        version="1.0.0",
        tools=sdk_tools,
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentSDKAgent(BaseAgent):
    """Agent that routes Claude model usage through the Claude Agent SDK.

    This agent delegates LLM interaction to the Claude Agent SDK
    (``claude-agent-sdk``), which wraps the Claude Code CLI.  Existing
    roshni :class:`ToolDefinition` objects are automatically converted
    to MCP tools so Claude can invoke them.

    Simple (tool-free) queries use the lightweight ``query()`` API;
    tool-bearing queries use ``ClaudeSDKClient`` for full agentic support.

    Example::

        from roshni.agent.agent_sdk import AgentSDKAgent
        from roshni.core.config import Config
        from roshni.core.secrets import SecretsManager

        agent = AgentSDKAgent(
            config=Config(data_dir="/tmp"),
            secrets=SecretsManager(providers=[]),
            system_prompt="You are a helpful assistant.",
        )
        result = agent.chat("What is 2 + 2?")
        print(result.text)
    """

    def __init__(
        self,
        config: Config,
        secrets: SecretsManager,
        tools: list[ToolDefinition] | None = None,
        *,
        system_prompt: str | None = None,
        persona_dir: str | None = None,
        name: str = "assistant",
        max_turns: int = 5,
        cwd: str | None = None,
        permission_mode: str | None = None,
        model: str | None = None,
    ):
        super().__init__(name=name)

        try:
            from claude_agent_sdk import ClaudeAgentOptions
        except ImportError:
            raise ImportError("Claude Agent SDK not installed. Install with:  pip install claude-agent-sdk")

        self.config = config
        self.secrets = secrets
        self.tools = list(tools) if tools else []
        self._max_turns = max_turns
        self._model = model

        # --- System prompt ---
        if system_prompt:
            self._system_prompt = system_prompt
        elif persona_dir:
            from roshni.agent.persona import get_system_prompt

            self._system_prompt = get_system_prompt(persona_dir)
        else:
            self._system_prompt = "You are a helpful personal AI assistant."

        # --- MCP server from roshni tools ---
        self._mcp_server: Any | None = None
        self._allowed_tools: list[str] = []
        if self.tools:
            self._mcp_server = _build_mcp_server(self.tools)
            self._allowed_tools = [f"mcp__roshni-tools__{t.name}" for t in self.tools]

        # --- Build options ---
        mcp_servers: dict[str, Any] = {}
        if self._mcp_server is not None:
            mcp_servers["roshni-tools"] = self._mcp_server

        options_kwargs: dict[str, Any] = {
            "system_prompt": self._system_prompt,
            "max_turns": max_turns,
        }
        if mcp_servers:
            options_kwargs["mcp_servers"] = mcp_servers
        if self._allowed_tools:
            options_kwargs["allowed_tools"] = self._allowed_tools
        if cwd:
            options_kwargs["cwd"] = cwd
        if permission_mode:
            options_kwargs["permission_mode"] = permission_mode
        if model:
            options_kwargs["model"] = model

        self._options = ClaudeAgentOptions(**options_kwargs)

        logger.debug(f"AgentSDKAgent: name={name}  tools={len(self.tools)}  max_turns={max_turns}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Model identifier — returns the configured model or a default label."""
        return self._model or "claude-agent-sdk"

    @property
    def provider(self) -> str:
        return "anthropic"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        *,
        mode: str | None = None,
        call_type: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        max_iterations: int = 5,
        on_tool_start: Callable[[str, int, dict | None], None] | None = None,
        on_stream: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Process a user message via the Claude Agent SDK."""
        self._busy.set()
        start = time()

        try:
            result = _run_async(self._achat(message, on_stream=on_stream))
            return ChatResult(
                text=result["text"],
                duration=time() - start,
                tool_calls=result.get("tool_calls", []),
                model="claude-agent-sdk",
            )
        except Exception as e:
            logger.error(f"AgentSDKAgent error ({type(e).__name__}): {e}")
            return ChatResult(
                text=f"An error occurred: {e}",
                duration=time() - start,
                tool_calls=[],
                model="claude-agent-sdk",
            )
        finally:
            self._busy.clear()

    # ------------------------------------------------------------------
    # Internal async implementation
    # ------------------------------------------------------------------

    async def _achat(self, message: str, *, on_stream: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Dispatch to the appropriate SDK path based on tool availability."""
        if self.tools:
            return await self._achat_with_client(message, on_stream=on_stream)
        return await self._achat_simple(message, on_stream=on_stream)

    async def _achat_simple(self, message: str, *, on_stream: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Lightweight path using ``query()`` for tool-free interactions."""
        from claude_agent_sdk import query

        texts: list[str] = []

        async for msg in query(prompt=message, options=self._options):
            text = _extract_text_from_sdk_message(msg)
            if text:
                texts.append(text)
                if on_stream:
                    on_stream(text)

        # ResultMessage.result often echoes the last AssistantMessage text
        deduped: list[str] = []
        for t in texts:
            if not deduped or t != deduped[-1]:
                deduped.append(t)

        return {"text": "\n".join(deduped) if deduped else "", "tool_calls": []}

    async def _achat_with_client(
        self, message: str, *, on_stream: Callable[[str], None] | None = None
    ) -> dict[str, Any]:
        """Full path using ``ClaudeSDKClient`` for tool-bearing interactions."""
        from claude_agent_sdk import ClaudeSDKClient

        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        async with ClaudeSDKClient(options=self._options) as client:
            await client.query(message)
            async for msg in client.receive_response():
                text = _extract_text_from_sdk_message(msg)
                if text:
                    texts.append(text)
                    if on_stream:
                        on_stream(text)

                msg_tool_calls = _extract_tool_calls_from_sdk_message(msg)
                tool_calls.extend(msg_tool_calls)

        # Filter duplicate text: ResultMessage.result often echoes the last
        # AssistantMessage text, so deduplicate adjacent identical entries.
        deduped: list[str] = []
        for t in texts:
            if not deduped or t != deduped[-1]:
                deduped.append(t)

        return {
            "text": "\n".join(deduped) if deduped else "",
            "tool_calls": tool_calls,
        }


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _extract_text_from_sdk_message(msg: Any) -> str:
    """Extract plain text content from a Claude Agent SDK message.

    Handles:
    - ``AssistantMessage`` — ``.content`` is a list of text/tool-use blocks
    - ``ResultMessage`` — ``.result`` is a plain string (final summary)
    - ``StreamEvent`` — ignored (no user-visible text)
    - Plain string ``.content`` — returned directly
    """
    # ResultMessage has .result (str | None) — the final output
    result = getattr(msg, "result", None)
    if result and isinstance(result, str):
        return result

    # AssistantMessage has .content which is a list of blocks
    content = getattr(msg, "content", None)
    if content is None:
        return ""

    # If content is a string, return directly
    if isinstance(content, str):
        return content

    # If content is a list of blocks, extract text from TextBlock instances
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            # TextBlock has .text attribute
            text = getattr(block, "text", None)
            if text and isinstance(text, str):
                # Skip tool use blocks — they have 'name' and 'input' attrs
                if not hasattr(block, "name") or not hasattr(block, "input"):
                    parts.append(text)
        return "".join(parts)

    return ""


def _extract_tool_calls_from_sdk_message(msg: Any) -> list[dict[str, Any]]:
    """Extract tool call information from a Claude Agent SDK message."""
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return []

    calls: list[dict[str, Any]] = []
    for block in content:
        # ToolUseBlock has .name and .input
        block_name = getattr(block, "name", None)
        block_input = getattr(block, "input", None)
        if block_name is not None and block_input is not None:
            calls.append(
                {
                    "name": block_name,
                    "args": block_input,
                    "result": "",
                }
            )
    return calls
