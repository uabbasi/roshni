# Changelog

## 0.2.0

### Simplified Install

- **`pip install roshni` is all you need.** LLM (litellm), Google APIs, Telegram, and web search are now core dependencies. No more juggling `[bot]`, `[google]`, `[web]` extras.
- Optional extras remain only for heavy/specialized features: health/fitbit, journal RAG, financial models, LangChain.

### Vault — Persistent Memory That Doesn't Lose History

- **Append, don't overwrite.** Saving a person/project/idea that already exists now appends a dated bullet instead of destroying previous notes.
- **Partial name matching.** `get_person("alice")` finds `alice-smith.md`. Works for people, projects, and ideas.
- **New `get_idea` tool.** Read back ideas by name, not just list and search them.
- **Rich listings.** `list_people`, `list_projects`, `list_ideas` now show last-updated dates.
- **Timestamps on everything.** `created` and `updated` in frontmatter for all vault types.
- **Thread-safe writes.** File lock prevents corruption from concurrent saves.

### Trello — Daily Briefing and Urgency

- **`trello_today` daily briefing.** Pulls cards from "Today" and "This Week" lists, sorted by urgency: overdue first, then due today, then upcoming.
- **Due date urgency.** All card listings now show human-readable urgency ("OVERDUE (3d ago)", "TODAY", "tomorrow", "in 5d") instead of raw ISO timestamps.
- **Name-based lookup.** `trello_find_list` and `trello_find_card` let the agent find things by name instead of requiring IDs.
- **Checklist rendering.** Card details now show checklists with `[x]/[ ]` progress.

### Web Search — Real Results

- **Replaced DuckDuckGo Instant Answer API** (which returned encyclopedia entries) **with `duckduckgo-search`** for actual search results with titles, snippets, and URLs.
- **Auto-fetch first result.** First result page is fetched and appended for richer context.
- **Auto-append current year.** Queries without a year get the current year added for fresher results.

### Init Wizard

- Reworked for non-technical users with arrow-key selection, step-by-step setup guides for every provider and integration.

### Infrastructure

- Vault-backed brain with permission tiers (observe/interact/admin) and task management.
- LLM prompt caching, token management, response continuation, multi-provider fallback.
- Trello, Notion, and HealthKit integrations.
- Pre-commit lint and format hooks.
- MIT license.

## 0.1.0

Initial release. Core framework with config, secrets, storage, health collectors, journal RAG, financial calculators, and Google integrations.
