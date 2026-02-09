# CLAUDE.md — roshni

Personal data infrastructure framework. Modular, protocol-based, plugin-oriented.

## Commands

```bash
# Setup
uv sync --extra dev              # Install with dev dependencies
uv sync --extra all              # Install everything

# Test
uv run pytest tests/                        # All unit tests
uv run pytest tests/ -m smoke               # Smoke tests only (< 10s)
uv run pytest tests/ -m integration         # Integration tests (need creds)
uv run pytest tests/core/test_config.py     # Single file

# Lint & Format
uv run ruff check src/ tests/              # Lint
uv run ruff check src/ tests/ --fix        # Lint + autofix
uv run ruff format src/ tests/             # Format
uv run ruff format --check src/ tests/     # Format check (CI)

# Type check
uv run mypy src/roshni/                    # Currently non-blocking in CI
```

## Architecture

```
src/roshni/
├── core/           # Config, secrets, storage, LLM abstraction, auth, utils
│   ├── auth/       #   Google OAuth + service account helpers
│   ├── cli/        #   CLI utilities
│   ├── llm/        #   LiteLLM wrapper, model selection, token budgets
│   ├── storage/    #   StorageBackend ABC, LocalStorage, compression
│   └── utils/      #   Async helpers, caching, file I/O, logging, text
├── agent/          # AI agent framework: base agent, router, circuit breaker, persona
├── gateway/        # Messaging platform integrations (Telegram plugin)
├── health/         # Health data collection: protocols, ETL base, registry, Fitbit plugin
├── financial/      # Calculators (mortgage, zakat, tax, life events) + market data
├── journal/        # Journal RAG: store protocol, search, retrieval strategies
└── integrations/   # Google API wrappers (Sheets, Gmail, Drive, GCS)
```

Modules are self-contained — import from submodules directly (`from roshni.core.config import Config`).

## Key Design Patterns

**Protocols & ABCs** — Extension points for DI. All are `runtime_checkable` where applicable:
- `SecretProvider` (`core/secrets.py`) — Secret backends (env, YAML, JSON, dotenv, GCP)
- `StorageBackend` (`core/storage/base.py`) — Async storage (local filesystem built-in)
- `HealthCollector` (`health/collector.py`) — Health data source plugins
- `BaseETL` (`health/etl_base.py`) — Extract/transform/load processors
- `JournalStore` (`journal/store.py`) — Read-only journal entry access
- `BaseAgent` (`agent/base.py`) — Tool-calling conversational agents
- `BotGateway` (`gateway/base.py`) — Messaging platform adapters

**Chain of Responsibility** — `SecretsManager` queries providers in order; first non-None wins.

**Registry + Entry Points** — `HealthCollectorRegistry` discovers plugins via `roshni.health_collectors` entry point group.

**Router** — `agent/router.py` dispatches messages via slash commands, prefix routes, keyword patterns.

**Circuit Breaker** — `agent/circuit_breaker.py` tracks service health with configurable thresholds.

**Persona Assembly** — `agent/persona.py` builds system prompts from markdown files (IDENTITY.md, SOUL.md, USER.md, AGENTS.md).

## Exception Hierarchy

```
RoshniError                          # core/exceptions.py — catch-all base
├── ConfigurationError
├── APIError
│   ├── GoogleAPIError
│   └── LLMError
├── DataProcessingError
├── FileIOError
├── CacheError
├── AuthenticationError
└── SecretNotFoundError

StorageError                         # core/storage/base.py — separate tree
├── StorageKeyError (+ KeyError)
├── StoragePermissionError
└── StorageQuotaError
```

## Dependencies

Core install includes everything needed to run: LLM (litellm), Telegram, Google APIs, web search (duckduckgo-search).

**Optional extras** for specialized/heavy features:

| Extra | What it adds |
|-------|-------------|
| `health` | pandas, requests — health collection base |
| `fitbit` | fitbit, requests-oauthlib — Fitbit collector |
| `journal` | sentence-transformers, scikit-learn — embeddings |
| `journal-faiss` | faiss-cpu |
| `journal-chroma` | chromadb |
| `financial` | pandas |
| `financial-full` | yfinance, duckdb, cvxpy — market data + optimization |
| `agent-langchain` | langchain, langchain-core |
| `all` | Everything above |
| `dev` | pytest, ruff, mypy + subset of extras |

## Testing

- **async**: `asyncio_mode = "auto"` — async tests just work
- **markers**: `smoke` (critical path, fast), `integration` (needs external services, skipped by default)
- **shared fixtures** in `tests/conftest.py`: `tmp_dir`, `tmp_config_file`
- **structure**: mirrors `src/roshni/` — `tests/core/`, `tests/agent/`, etc.
- **CI matrix**: Python 3.11, 3.12, 3.13 on Ubuntu

## Pre-Commit Checklist

Before every commit, always run these and fix any issues:
```bash
uv run ruff check src/ tests/ --fix        # Lint + autofix
uv run ruff format src/ tests/             # Format
```

## Tool Config

- **ruff**: line-length 120, rules E/F/I/B/UP/RUF
- **mypy**: Python 3.11, strict optional, `ignore_missing_imports = true`
- **hatchling**: build backend, PEP 561 typed (`py.typed`)
