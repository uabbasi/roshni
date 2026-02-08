# Roshni

> *roshni* (روشنی) — "light" in Urdu

Personal data infrastructure framework for health tracking, financial planning, journal search, and AI agents.

## Install

```bash
pip install roshni                    # Core only (config, secrets, caching)
pip install roshni[health]            # + Health data collection framework
pip install roshni[fitbit]            # + Fitbit collector
pip install roshni[journal-faiss]     # + Journal RAG with FAISS
pip install roshni[financial-full]    # + Financial calculators + market data
pip install roshni[all]               # Everything
```

## Modules

| Module | What it does | Optional dep |
|--------|-------------|--------------|
| `core` | Config, secrets, caching, storage, LLM abstraction | *(included)* |
| `health` | Health data collection protocol + ETL base | `health` |
| `financial` | Mortgage, zakat, tax calculators, market data | `financial` / `financial-full` |
| `journal` | RAG-based journal search (chunking, embeddings, retrieval) | `journal-faiss` / `journal-chroma` |
| `agent` | AI agent framework (base agent, router, circuit breaker) | `agent` |
| `gateway` | Messaging gateway (Telegram bot framework) | `gateway-telegram` |
| `integrations` | Google Sheets, Drive, Gmail, Cloud Storage wrappers | `google` |

## Quick Start

### Configuration

```python
from roshni.core.config import Config

config = Config(
    config_file="config/config.yaml",
    env_prefix="MYAPP_",
    data_dir="~/.myapp-data",
)

print(config.get("llm.provider"))
```

### Secrets Management

```python
from roshni.core.secrets import SecretsManager, EnvProvider, YamlFileProvider

manager = SecretsManager(providers=[
    EnvProvider("MYAPP_"),
    YamlFileProvider("~/.myapp/secrets.yaml"),
])

api_key = manager.get("trello.api_key")
all_trello = manager.get_namespace("trello")
```

### Financial Calculators

```python
from roshni.financial.calculators import MortgageTerms, calculate_monthly_payment

terms = MortgageTerms(
    balance=500_000,
    current_rate=0.065,
    is_interest_only=False,
    remaining_term_years=30,
)

payment = calculate_monthly_payment(terms)
```

### Health Data Collection

```python
from roshni.health.collector import HealthCollector, BaseCollector
from roshni.health.models import DailyHealth

class MyCollector(BaseCollector):
    name = "my_tracker"

    def collect(self, start_date, end_date):
        # Fetch from your data source
        return [DailyHealth(date=start_date, steps=10000, sleep_hours=7.5)]
```

## Optional Dependencies

| Extra | Installs | Use case |
|-------|----------|----------|
| `health` | pandas, requests | Health data pipelines |
| `fitbit` | fitbit, requests-oauthlib | Fitbit API collector |
| `journal` | sentence-transformers, scikit-learn | Journal embeddings + search |
| `journal-faiss` | faiss-cpu | FAISS vector backend |
| `journal-chroma` | chromadb | ChromaDB vector backend |
| `financial` | pandas | Financial data analysis |
| `financial-full` | yfinance, duckdb, cvxpy | Market data + portfolio optimization |
| `llm` | litellm | Multi-provider LLM abstraction |
| `agent` | litellm | AI agent framework |
| `agent-langchain` | langchain, langchain-core | LangChain agent integration |
| `gateway-telegram` | python-telegram-bot, apscheduler | Telegram bot gateway |
| `google` | gspread, google-api-python-client, etc. | Google API integrations |
| `storage-gcs` | google-cloud-storage | GCS storage backend |

## Development

```bash
git clone https://github.com/uabbasi/roshni.git
cd roshni
uv sync --extra dev

# Run tests
uv run pytest tests/

# Lint + format
uv run ruff check . --fix && uv run ruff format .

# Type check
uv run mypy src/roshni/
```

## License

Apache 2.0
