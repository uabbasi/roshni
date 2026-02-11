"""Agent framework — base agent, routing, circuit breaker, persona loading."""

# Re-export ModelSelector from its canonical location for backward compat
from roshni.core.events import EventBus
from roshni.core.llm.model_selector import ModelSelector

from .advisor import Advisor, AfterChatHook, FunctionAdvisor, FunctionAfterChatHook
from .advisors import MemoryAdvisor, SystemHealthAdvisor
from .base import BaseAgent, ChatResult
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .default import DefaultAgent
from .memory import MemoryManager
from .persona import get_system_prompt
from .router import CommandParseResult, Router, RouteResult
from .session import JSONLSessionStore, Session, SessionStore, Turn

__all__ = [
    "Advisor",
    "AfterChatHook",
    "BaseAgent",
    "ChatResult",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CommandParseResult",
    "DefaultAgent",
    "EventBus",
    "FunctionAdvisor",
    "FunctionAfterChatHook",
    "JSONLSessionStore",
    "MemoryAdvisor",
    "MemoryManager",
    "ModelSelector",
    "RouteResult",
    "Router",
    "Session",
    "SessionStore",
    "SystemHealthAdvisor",
    "Turn",
    "get_system_prompt",
]
