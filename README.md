# Roshni

> *roshni* (روشنی) — "light" in Urdu

Your personal AI assistant on Telegram.

## Get Started

You need Python 3.11+ installed.

### 1. Install

    pip install "roshni[bot]"

### 2. Set Up

    roshni init

The wizard walks you through everything:
- Pick an AI provider (Anthropic, OpenAI, or Google)
- Create a Telegram bot (instructions included)
- Give your bot a personality
- Connect your services (Gmail, Obsidian)

### 3. Run

    roshni run

Open Telegram and message your bot.

### Try it in the terminal first

    roshni chat

No Telegram setup needed — just chat in your terminal.

## What Your Bot Can Do

- **Chat** — Ask anything, get thoughtful responses
- **Email** — "Send an email to Sarah about the meeting tomorrow"
- **Notes** — "Remember that the plumber is coming Thursday"
- **Search your notes** — "What did I write about the trip to Japan?"

## Services

| Service | What it does | Setup |
|---------|-------------|-------|
| Gmail | Send emails on your behalf | Google App Password |
| Obsidian | Search your vault | Point to the folder |
| Notes | Save and recall quick notes | Built-in, no setup |

*Coming soon: Google Calendar, Trello*

## Commands

In Telegram, just type normally. Your bot also understands:
- `/help` — see what your bot can do
- `/clear` — start a fresh conversation

## Reconfigure

    roshni init     # re-run the wizard anytime

## Framework

Roshni is also a modular Python framework you can build on. Each module is self-contained:

| Module | What it does | Install with |
|--------|-------------|--------------|
| **core** | Config, secrets, caching, storage, LLM helpers | *(always included)* |
| **financial** | Mortgage, zakat, tax, and life-event calculators | `roshni[financial]` |
| **health** | Health data collection framework + Fitbit plugin | `roshni[health]` or `roshni[fitbit]` |
| **journal** | Search your journal entries with AI (embeddings + RAG) | `roshni[journal-faiss]` |
| **agent** | Build AI agents with tool calling and routing | `roshni[agent]` |
| **gateway** | Connect agents to Telegram (or other platforms) | `roshni[gateway-telegram]` |
| **integrations** | Google Sheets, Drive, Gmail, Cloud Storage wrappers | `roshni[google]` |

<details>
<summary>Full list of optional extras</summary>

| Extra | What it adds |
|-------|-------------|
| `bot` | Agent + Telegram gateway (everything you need for `roshni run`) |
| `health` | pandas, requests — health collection base |
| `fitbit` | Fitbit API collector (includes `health`) |
| `journal` | sentence-transformers, scikit-learn — embedding engine |
| `journal-faiss` | FAISS vector search (includes `journal`) |
| `journal-chroma` | ChromaDB vector search (includes `journal`) |
| `financial` | pandas — financial data analysis |
| `financial-full` | yfinance, duckdb, cvxpy — market data + portfolio optimization |
| `llm` | litellm — multi-provider LLM client |
| `agent` | AI agent framework (includes `llm`) |
| `agent-langchain` | LangChain integration (includes `agent`) |
| `gateway-telegram` | Telegram bot gateway |
| `google` | Google Sheets, Drive, Gmail, Cloud Storage |
| `storage-gcs` | Google Cloud Storage backend |
| `all` | Everything above |

</details>

## Development

For developers extending roshni:

    git clone https://github.com/uabbasi/roshni.git
    cd roshni && uv sync --extra dev
    uv run pytest tests/

See [CLAUDE.md](CLAUDE.md) for architecture details.

## License

Apache 2.0
