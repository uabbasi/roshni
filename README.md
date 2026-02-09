# Roshni

> *roshni* (روشنی) — "light" in Urdu

Safety-first personal AI assistant with Telegram and terminal support.

## 5-Minute First Run

You need Python 3.11+.

### 1. Install

```bash
pip install "roshni[bot]"
```

### 2. Run Setup Wizard

```bash
roshni init
```

The wizard guides you through:
- Bot identity (name + personality — with arrow-key selection)
- AI provider (Gemini default, with step-by-step setup guides for every provider)
- Bot's Brain (vault folder for memory, tasks, and knowledge)
- Integrations (Google services, Trello, Notion, Apple Health, Fitbit — each with setup walkthrough)
- Safety level (one simple choice: Balanced / Read only / Full access)
- Platform (Telegram or terminal)

### 3. Try It Immediately

```bash
roshni chat
```

Terminal chat is the fastest first experience. No Telegram required.

### 4. Run on Telegram

```bash
roshni run
```

## Security Model

- **Balanced** (default): Can read and search everything, create drafts/tasks/notes. Asks before sending emails or deleting anything.
- **Read only**: Can search and read your data, but can't create or change anything.
- **Full access**: Can do everything including send emails and delete items.
- Telegram is deny-by-default unless your user ID is allowlisted.
- Gmail is draft-first by default (no send).

## Integrations

| Integration | Default | Risk | Notes |
|---|---|---|---|
| Gmail | Draft-only | Low | Saves drafts locally; optional send mode |
| Google Workspace profile | Off | Medium | Guided least-privilege scope choices |
| Obsidian | Off | Low | Read-only vault search |
| Trello | Off | Medium | Boards/lists/cards/labels/comments with approvals on write actions |
| Notion | Off | Medium | Database-backed page search/create/update/append |
| HealthKit import | Off | Low | Reads Apple Health `export.xml` summaries |
| Fitbit | Off | Low | Steps, sleep, heart rate via Fitbit API |
| Notes | On | Medium | Local note writes (approval-gated) |
| Weather | On | Low | Read-only |
| Web search/fetch | On | Low | Read-only |
| Reminders | On | Medium | Local reminder writes (approval-gated) |

## Local LLM (Ollama) Requirements

Recommended minimums:
- 8 GB RAM: small models only
- 16 GB RAM: solid everyday use
- 32 GB RAM: smoother larger local models

Quick start:
```bash
# Install Ollama first: https://ollama.com
ollama serve
ollama pull deepseek-r1
```

Then choose `Local (Ollama)` in `roshni init`.

## Delighter Features

- `morning_brief`
- `daily_plan`
- `weekly_review`
- `inbox_triage_bundle`
- `save_reminder` / `list_reminders` / `complete_reminder`

## Built-in Tools

- `get_weather`
- `search_web`
- `fetch_webpage`

## Reconfigure Anytime

```bash
roshni init
```

## Optional Extras

| Extra | What it adds |
|---|---|
| `bot` | Agent + Telegram gateway |
| `llm` | LiteLLM multi-provider client |
| `fitbit` | Fitbit collector plugin |
| `google` | Google API wrappers (Sheets/Drive/Gmail/GCS) |
| `all` | Everything |

## Development

```bash
git clone https://github.com/uabbasi/roshni.git
cd roshni
uv sync --extra dev
uv run pytest tests/
```

See `CLAUDE.md` for architecture details.
