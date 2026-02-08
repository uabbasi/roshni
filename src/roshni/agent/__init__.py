"""Agent framework — base agent, routing, circuit breaker, persona loading."""

# Re-export ModelSelector from its canonical location for backward compat
from roshni.core.llm.model_selector import ModelSelector

from .base import BaseAgent, ChatResult
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .default import DefaultAgent
from .memory import MemoryManager
from .persona import get_system_prompt
from .router import CommandParseResult, Router, RouteResult

__all__ = [
    "BaseAgent",
    "ChatResult",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CommandParseResult",
    "DefaultAgent",
    "MemoryManager",
    "ModelSelector",
    "RouteResult",
    "Router",
    "get_system_prompt",
]
