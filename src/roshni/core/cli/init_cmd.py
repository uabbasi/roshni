"""roshni init — interactive setup wizard."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import click
import questionary
import yaml
from questionary import Choice
from rich.console import Console
from rich.panel import Panel

ROSHNI_DIR = Path.home() / ".roshni"
CONFIG_PATH = ROSHNI_DIR / "config.yaml"
SECRETS_PATH = ROSHNI_DIR / "secrets.yaml"

console = Console()

# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------

ALL_PROVIDERS = ["gemini", "openai", "anthropic", "deepseek", "xai", "groq", "local"]

PROVIDER_DISPLAY: dict[str, str] = {
    "gemini": "Google Gemini",
    "openai": "OpenAI (GPT)",
    "anthropic": "Anthropic (Claude)",
    "deepseek": "DeepSeek",
    "xai": "xAI (Grok)",
    "groq": "Groq",
    "local": "Ollama (local)",
}

PROVIDER_SETUP_GUIDES: dict[str, str] = {
    "gemini": (
        "1. Go to: https://aistudio.google.com/apikey\n"
        "2. Sign in with your Google account\n"
        '3. Click "Create API Key"\n'
        '4. Select or create a project, then click "Create"\n'
        "5. Copy the key and paste it below\n"
        "\n"
        "Free tier available — no credit card needed."
    ),
    "openai": (
        "1. Go to: https://platform.openai.com/api-keys\n"
        "2. Sign in or create an account\n"
        '3. Click "Create new secret key"\n'
        "4. Copy the key and paste it below\n"
        "\n"
        "Free trial credits available for new accounts."
    ),
    "anthropic": (
        "1. Go to: https://console.anthropic.com/settings/keys\n"
        "2. Sign in or create an account\n"
        '3. Click "Create Key"\n'
        "4. Copy the key and paste it below"
    ),
    "deepseek": (
        "1. Go to: https://platform.deepseek.com/api_keys\n"
        "2. Sign in or create an account\n"
        "3. Create an API key\n"
        "4. Copy the key and paste it below\n"
        "\n"
        "Very affordable pricing."
    ),
    "xai": (
        "1. Go to: https://console.x.ai/\n"
        "2. Sign in with your X account\n"
        "3. Navigate to API keys\n"
        "4. Create a key and paste it below"
    ),
    "groq": (
        "1. Go to: https://console.groq.com/keys\n"
        "2. Sign in or create an account\n"
        '3. Click "Create API Key"\n'
        "4. Copy the key and paste it below\n"
        "\n"
        "Free tier available — very fast inference."
    ),
    "local": (
        "1. Install Ollama: https://ollama.com/download\n"
        "2. Run: ollama pull deepseek-r1\n"
        "3. That's it — no API key needed!"
    ),
}

TONE_DESCRIPTIONS: dict[str, str] = {
    "friendly": "Casual and approachable — like a helpful friend",
    "professional": "Precise and efficient — like a great executive assistant",
    "warm": "Empathetic and thoughtful — calm and caring",
    "witty": "Sharp and clever — dry humor with substance",
}

SAFETY_LEVELS = {
    "balanced": {
        "label": "Balanced (recommended)",
        "description": (
            "Can read and search everything. Can create drafts, tasks, and notes. "
            "Asks for your OK before sending emails or deleting anything."
        ),
        "tier": "interact",
    },
    "readonly": {
        "label": "Read only",
        "description": "Can search and read your data, but can't create or change anything.",
        "tier": "observe",
    },
    "full": {
        "label": "Full access",
        "description": "Can do everything including send emails and delete items.",
        "tier": "full",
    },
}

PERMISSION_DOMAINS = ["gmail", "trello", "notion", "obsidian", "health", "tasks", "vault"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_existing_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_existing_secrets() -> dict:
    if SECRETS_PATH.exists():
        with open(SECRETS_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _mask_key(key: str) -> str:
    if len(key) <= 12:
        return key[:4] + "..." + key[-2:]
    return key[:4] + "..." + key[-4:]


def _validate_api_key(provider: str, api_key: str) -> bool | None:
    """Returns True if valid, False if invalid, None if skipped."""
    try:
        import litellm

        from roshni.core.llm.config import get_default_model

        model = get_default_model(provider)
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            max_tokens=10,
            api_key=api_key,
        )
        return True
    except ImportError:
        return None
    except Exception as e:
        click.echo(f"  API key validation failed: {e}")
        return False


def _show_guide(title: str, body: str) -> None:
    console.print()
    console.print(Panel(body, title=title, border_style="cyan"))
    console.print()


def _ask_text(message: str, default: str = "") -> str:
    result = questionary.text(message, default=default).ask()
    if result is None:
        raise click.Abort()
    return result


def _ask_select(message: str, choices: list[Choice], **kwargs) -> str:
    result = questionary.select(message, choices=choices, **kwargs).ask()
    if result is None:
        raise click.Abort()
    return result


def _ask_confirm(message: str, default: bool = True) -> bool:
    result = questionary.confirm(message, default=default).ask()
    if result is None:
        raise click.Abort()
    return result


def _ask_checkbox(message: str, choices: list[Choice]) -> list[str]:
    result = questionary.checkbox(message, choices=choices).ask()
    if result is None:
        raise click.Abort()
    return result


def _prompt_api_key(provider: str, existing: str = "") -> str:
    from roshni.core.llm.config import PROVIDER_ENV_MAP

    if existing:
        click.echo(f"  {PROVIDER_DISPLAY[provider]} key on file: {_mask_key(existing)}")
        if not _ask_confirm("  Change it?", default=False):
            return existing

    guide = PROVIDER_SETUP_GUIDES.get(provider, "")
    if guide:
        _show_guide(f"How to get your {PROVIDER_DISPLAY[provider]} API key", guide)

    env_var = PROVIDER_ENV_MAP.get(provider, "")
    hint = f" (or set {env_var})" if env_var else ""
    key = questionary.password(f"  Paste your {PROVIDER_DISPLAY[provider]} API key{hint}:").ask()
    if key is None:
        raise click.Abort()

    try:
        import litellm  # noqa: F401

        can_validate = True
    except ImportError:
        can_validate = False

    if can_validate:
        click.echo("  Validating...")
        validation = _validate_api_key(provider, key)
        if validation is True:
            console.print("  [green]Valid![/green]")
        elif validation is False:
            if not _ask_confirm("  Key validation failed. Use it anyway?", default=False):
                raise click.Abort()
    return key


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------


def _step_welcome(is_reconfig: bool) -> None:
    """Step 1: Welcome."""
    if is_reconfig:
        console.print(
            Panel(
                "Let's update your assistant's settings.\nExisting values are shown as defaults.",
                title="Roshni Setup",
            )
        )
    else:
        console.print(
            Panel(
                "Let's set up your personal AI assistant step by step.\n"
                "Use arrow keys to select options. Press Enter to confirm.",
                title="Welcome to Roshni",
            )
        )


def _step_identity(existing_config: dict) -> tuple[str, str, str]:
    """Step 2: Bot name, user name, personality."""
    console.print("\n[bold]Step 1 · Identity[/bold]")

    default_name = existing_config.get("bot", {}).get("name", "Roshni")
    bot_name = _ask_text("What should your bot be called?", default=default_name)

    default_user = existing_config.get("user", {}).get("name", "") or os.environ.get("USER", "")
    user_name = _ask_text("What's your name?", default=default_user)

    tones = list(TONE_DESCRIPTIONS.keys()) + ["custom"]
    default_tone = existing_config.get("bot", {}).get("tone", "friendly")

    tone_choices = [Choice(title=f"{t.capitalize()} — {TONE_DESCRIPTIONS[t]}", value=t) for t in TONE_DESCRIPTIONS]
    tone_choices.append(Choice(title="Custom — describe your own", value="custom"))

    tone = _ask_select(
        "What personality should your bot have?",
        tone_choices,
        default=default_tone if default_tone in tones else "friendly",
    )

    if tone == "custom":
        tone = _ask_text("Describe the personality you'd like (e.g., 'sarcastic but kind'):")

    return bot_name, user_name, tone


def _step_ai_provider(existing_config: dict, existing_secrets: dict) -> tuple[str, str, dict[str, str]]:
    """Step 3: AI provider + API keys. Returns (provider, fallback, {provider: key})."""
    console.print("\n[bold]Step 2 · AI Provider[/bold]")

    existing_llm = existing_config.get("llm", {})
    default_provider = existing_llm.get("default", "") or existing_llm.get("provider", "gemini")

    provider_choices = []
    for p in ALL_PROVIDERS:
        label = PROVIDER_DISPLAY[p]
        if p == "gemini":
            label += "  (recommended — free tier available)"
        provider_choices.append(Choice(title=label, value=p))

    provider = _ask_select(
        "Which AI provider do you want to use?",
        provider_choices,
        default=default_provider if default_provider in ALL_PROVIDERS else "gemini",
    )

    # Collect existing API keys
    existing_api_keys: dict = existing_secrets.get("llm", {}).get("api_keys", {}) or {}
    if not existing_api_keys:
        legacy_key = existing_secrets.get("llm", {}).get("api_key", "")
        legacy_provider = existing_llm.get("provider", "")
        if legacy_key and legacy_provider:
            existing_api_keys = {legacy_provider: legacy_key}

    configured: dict[str, str] = {}

    # Primary key
    if provider != "local":
        configured[provider] = _prompt_api_key(provider, existing_api_keys.get(provider, ""))
    else:
        _show_guide("Ollama Setup", PROVIDER_SETUP_GUIDES["local"])

    # Fallback
    existing_fallback = existing_llm.get("fallback", "")
    fallback_provider = ""
    if _ask_confirm("Add a fallback provider? (used when default is unavailable)", default=bool(existing_fallback)):
        fallback_options = [p for p in ALL_PROVIDERS if p != provider and p != "local"]
        fb_choices = [Choice(title=PROVIDER_DISPLAY[p], value=p) for p in fallback_options]
        fallback_provider = _ask_select(
            "Which fallback provider?",
            fb_choices,
            default=existing_fallback if existing_fallback in fallback_options else fallback_options[0],
        )
        configured[fallback_provider] = _prompt_api_key(fallback_provider, existing_api_keys.get(fallback_provider, ""))

    return provider, fallback_provider, configured


def _step_vault(existing_config: dict, bot_name: str) -> tuple[str, str]:
    """Step 4: Bot's Brain (vault). Returns (vault_path, agent_dir)."""
    console.print("\n[bold]Step 3 · Bot's Brain[/bold]")
    console.print(
        "  Your bot stores its memory, tasks, and knowledge in a folder.\n"
        "  If you use Obsidian, point this to your vault and your bot can search your notes too."
    )

    existing_vault = existing_config.get("vault", {}) or {}
    default_path = existing_vault.get("path", "") or "~/Obsidian"
    default_agent = existing_vault.get("agent_dir", "") or bot_name.lower().replace(" ", "-")

    vault_path = _ask_text("Folder path:", default=default_path)
    vault_path = str(Path(vault_path).expanduser())

    vault_p = Path(vault_path)
    if vault_p.is_dir():
        console.print(f"  [green]Found:[/green] {vault_path}")
    else:
        console.print("  [yellow]Folder doesn't exist yet — it will be created during setup.[/yellow]")

    agent_dir = _ask_text("Bot's subdirectory name:", default=default_agent)

    return vault_path, agent_dir


def _step_integrations(existing_config: dict, existing_secrets: dict) -> tuple[dict, dict]:
    """Step 5: Integrations. Returns (integrations_config, secrets_updates)."""
    console.print("\n[bold]Step 4 · Integrations[/bold]")
    console.print("  Connect the services you want your bot to help with. You can add more later.\n")

    integrations: dict = existing_config.get("integrations", {}) or {}
    secrets_updates: dict = {}

    # --- 5a: Google Services (Gmail + Workspace combined) ---
    google_cfg = integrations.get("google_workspace", {}) or {}
    gmail_cfg = integrations.get("gmail", {}) or {}
    google_default = google_cfg.get("enabled", False) or gmail_cfg.get("enabled", False)

    google_enabled = _ask_confirm("Connect Google services? (Gmail, Calendar, Docs, Sheets)", default=google_default)

    gmail_enabled = False
    gmail_mode = "draft"
    gmail_allow_send = False
    gmail_address = existing_secrets.get("gmail", {}).get("address", "")
    gmail_app_password = existing_secrets.get("gmail", {}).get("app_password", "")
    google_ws_enabled = False
    google_credentials_path = str(
        Path(google_cfg.get("credentials_path", "~/.roshni/google/client_secret.json")).expanduser()
    )
    google_token_path = str(Path(google_cfg.get("token_path", "~/.roshni/google/token.pickle")).expanduser())
    google_scopes: dict = google_cfg.get("scopes", {}) or {}
    selected_services: list[str] = []

    if google_enabled:
        _show_guide(
            "Google Cloud Setup (one-time)",
            "1. Go to: https://console.cloud.google.com/\n"
            "2. Create a new project (or pick an existing one)\n"
            "3. Enable the APIs you need:\n"
            "     → Gmail API, Google Calendar API, Google Docs API, Google Sheets API\n"
            "4. Go to APIs & Services → OAuth consent screen → set up for External\n"
            "5. Go to Credentials → Create OAuth client ID → Desktop app\n"
            "6. Download the JSON file and save it\n"
            "\n"
            "You only need to do this once for all Google services.",
        )

        google_credentials_path = str(
            Path(
                _ask_text("Path to your downloaded OAuth credentials JSON:", default=google_credentials_path)
            ).expanduser()
        )

        gmail_address = _ask_text("Your Gmail address:", default=gmail_address)

        service_choices = [
            Choice("Gmail", checked=gmail_cfg.get("enabled", True)),
            Choice("Calendar", checked=google_scopes.get("calendar_rw", True)),
            Choice("Docs", checked=google_scopes.get("docs_readonly", True)),
            Choice("Sheets", checked=google_scopes.get("sheets_readonly", True)),
        ]
        selected_services = _ask_checkbox("Which Google services?", service_choices)

        gmail_enabled = "Gmail" in selected_services
        google_ws_enabled = True

        if gmail_enabled:
            gmail_allow_send = _ask_confirm(
                "Allow your bot to send emails directly? (default is draft-only)", default=False
            )
            gmail_mode = "send" if gmail_allow_send else "draft"
            if gmail_allow_send:
                if not gmail_app_password or _ask_confirm("Set/change Gmail App Password?", default=True):
                    _show_guide(
                        "Gmail App Password",
                        "1. Go to myaccount.google.com → Security → App Passwords\n"
                        "2. Create a new app password for 'Mail'\n"
                        "3. Copy the 16-character password",
                    )
                    pw = questionary.password("  Gmail App Password:").ask()
                    if pw is None:
                        raise click.Abort()
                    gmail_app_password = pw

        secrets_updates["gmail"] = {"address": gmail_address}
        if gmail_allow_send and gmail_app_password:
            secrets_updates["gmail"]["app_password"] = gmail_app_password

    # --- 5b: Trello ---
    trello_cfg = integrations.get("trello", {}) or {}
    trello_enabled = _ask_confirm("Connect Trello? (boards, lists, cards)", default=trello_cfg.get("enabled", False))
    trello_api_key = existing_secrets.get("trello", {}).get("api_key", "")
    trello_token = existing_secrets.get("trello", {}).get("token", "")
    trello_disable_board_delete = bool(trello_cfg.get("disable_board_delete", False))

    if trello_enabled:
        if not trello_api_key or _ask_confirm("Set/change Trello credentials?", default=not bool(trello_api_key)):
            _show_guide(
                "Trello Setup",
                "1. Go to: https://trello.com/power-ups/admin\n"
                '2. Click "New" to create a Power-Up\n'
                '3. Fill in a name (e.g., "Roshni Bot") and your workspace\n'
                '4. After creation, click "Generate a new API key"\n'
                "5. Copy the API key, then click the Token link to authorize\n"
                "6. Copy the token value",
            )
            key = questionary.password("  Trello API key:").ask()
            if key is None:
                raise click.Abort()
            trello_api_key = key
            tok = questionary.password("  Trello token:").ask()
            if tok is None:
                raise click.Abort()
            trello_token = tok

        trello_disable_board_delete = _ask_confirm("Disable permanent board deletion? (recommended)", default=True)
        secrets_updates["trello"] = {"api_key": trello_api_key, "token": trello_token}

    # --- 5c: Notion ---
    notion_cfg = integrations.get("notion", {}) or {}
    notion_enabled = _ask_confirm(
        "Connect Notion? (search, create, update pages)", default=notion_cfg.get("enabled", False)
    )
    notion_database_id = notion_cfg.get("database_id", "")
    notion_title_property = notion_cfg.get("title_property", "Name")
    notion_token = existing_secrets.get("notion", {}).get("token", "")

    if notion_enabled:
        if not notion_token or _ask_confirm("Set/change Notion credentials?", default=not bool(notion_token)):
            _show_guide(
                "Notion Setup",
                "1. Go to: https://www.notion.so/my-integrations\n"
                '2. Click "New integration"\n'
                '3. Give it a name (e.g., "Roshni Bot") and select your workspace\n'
                "4. Copy the Internal Integration Token\n"
                "5. Go to the database you want to connect\n"
                "6. Click ••• → Add connections → select your integration",
            )
            tok = questionary.password("  Notion integration token:").ask()
            if tok is None:
                raise click.Abort()
            notion_token = tok

        notion_database_id = _ask_text("Notion database ID:", default=notion_database_id)
        notion_title_property = _ask_text("Title property name:", default=notion_title_property)
        secrets_updates["notion"] = {"token": notion_token}

    # --- 5d: Apple Health ---
    healthkit_cfg = integrations.get("healthkit", {}) or {}
    healthkit_enabled = _ask_confirm("Import Apple Health data?", default=healthkit_cfg.get("enabled", False))
    healthkit_export_path = str(
        Path(healthkit_cfg.get("export_path", "~/Downloads/apple_health_export/export.xml")).expanduser()
    )
    if healthkit_enabled:
        console.print(
            "  [dim]On your iPhone: Health app → profile picture → Export All Health Data → "
            "share the zip to your Mac → unzip it[/dim]"
        )
        healthkit_export_path = str(Path(_ask_text("Path to export.xml:", default=healthkit_export_path)).expanduser())

    # --- 5e: Fitbit ---
    fitbit_cfg = integrations.get("fitbit", {}) or {}
    fitbit_enabled = _ask_confirm(
        "Connect Fitbit? (steps, sleep, heart rate)", default=fitbit_cfg.get("enabled", False)
    )
    fitbit_client_id = existing_secrets.get("fitbit", {}).get("client_id", "")
    fitbit_client_secret = existing_secrets.get("fitbit", {}).get("client_secret", "")

    if fitbit_enabled:
        if not fitbit_client_id or _ask_confirm("Set/change Fitbit credentials?", default=not bool(fitbit_client_id)):
            _show_guide(
                "Fitbit API Setup",
                "1. Go to: https://dev.fitbit.com/apps/new\n"
                "2. Sign in with your Fitbit account\n"
                "3. Register a new app:\n"
                "     → Application Name: Roshni (or anything)\n"
                "     → OAuth 2.0 Application Type: Personal\n"
                "     → Callback URL: http://localhost:8080/\n"
                "4. After creation, copy the Client ID and Client Secret",
            )
            cid = _ask_text("  Fitbit Client ID:", default=fitbit_client_id)
            fitbit_client_id = cid
            cs = questionary.password("  Fitbit Client Secret:").ask()
            if cs is None:
                raise click.Abort()
            fitbit_client_secret = cs

        secrets_updates["fitbit"] = {
            "client_id": fitbit_client_id,
            "client_secret": fitbit_client_secret,
        }

    # Build integrations config
    integrations_cfg = {
        "gmail": {
            "enabled": gmail_enabled,
            "mode": gmail_mode,
            "allow_send": gmail_allow_send,
        },
        "obsidian": {
            "enabled": True,  # auto-enabled when vault is configured
            "vault_path": "",  # filled in by caller with vault_path
        },
        "builtins": {"enabled": True},
        "delighters": {"enabled": True},
        "google_workspace": {
            "enabled": google_ws_enabled,
            "credentials_path": google_credentials_path,
            "token_path": google_token_path,
            "scopes": {
                "gmail_drafts": gmail_enabled,
                "calendar_rw": "Calendar" in selected_services,
                "docs_readonly": "Docs" in selected_services,
                "sheets_readonly": "Sheets" in selected_services,
            },
        },
        "trello": {
            "enabled": trello_enabled,
            "disable_board_delete": trello_disable_board_delete,
        },
        "notion": {
            "enabled": notion_enabled,
            "database_id": notion_database_id,
            "title_property": notion_title_property,
        },
        "healthkit": {
            "enabled": healthkit_enabled,
            "export_path": healthkit_export_path,
        },
        "fitbit": {
            "enabled": fitbit_enabled,
        },
    }

    return integrations_cfg, secrets_updates


def _step_safety() -> str:
    """Step 6: Safety level. Returns the tier string."""
    console.print("\n[bold]Step 5 · Safety Level[/bold]")

    choices = [
        Choice(
            title=f"{info['label']}\n    {info['description']}",
            value=key,
        )
        for key, info in SAFETY_LEVELS.items()
    ]

    level = _ask_select("How much should your bot be able to do on its own?", choices, default="balanced")
    return SAFETY_LEVELS[level]["tier"]


def _step_platform(existing_config: dict, existing_secrets: dict) -> tuple[str, str, str]:
    """Step 7: Platform. Returns (platform, telegram_token, telegram_user_id)."""
    console.print("\n[bold]Step 6 · Platform[/bold]")

    default_platform = existing_config.get("platform", "telegram")
    platform = _ask_select(
        "Where will your bot live?",
        [
            Choice(title="Telegram", value="telegram"),
            Choice(title="Terminal only", value="terminal-only"),
        ],
        default=default_platform,
    )

    telegram_token = existing_secrets.get("telegram", {}).get("bot_token", "")
    tg_cfg = existing_config.get("telegram", {})
    tg_uids = tg_cfg.get("allowed_user_ids", [])
    telegram_user_id = tg_uids[0] if tg_uids else ""

    if platform == "telegram":
        if not telegram_token or _ask_confirm("Set up Telegram bot token?", default=not bool(telegram_token)):
            _show_guide(
                "Create a Telegram Bot",
                "1. Open Telegram and search for @BotFather\n"
                "2. Send /newbot and follow the prompts\n"
                "3. Copy the token BotFather gives you",
            )
            tok = questionary.password("Bot token:").ask()
            if tok is None:
                raise click.Abort()
            telegram_token = tok

        if not telegram_user_id or _ask_confirm("Set up your Telegram user ID?", default=not bool(telegram_user_id)):
            _show_guide(
                "Find Your Telegram User ID",
                "1. Open Telegram and search for @userinfobot\n"
                "2. Send it any message\n"
                "3. It will reply with your user ID (a number)",
            )
            while True:
                telegram_user_id = _ask_text("Your Telegram user ID:")
                if telegram_user_id.strip().isdigit():
                    telegram_user_id = telegram_user_id.strip()
                    break
                click.echo("Please enter numbers only (example: 123456789).")

        if not telegram_user_id:
            raise click.ClickException(
                "Telegram requires an allowed user ID for security. Re-run and provide your Telegram user ID."
            )

    return platform, telegram_token, telegram_user_id


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


@click.command()
def init() -> None:
    """Set up your personal AI assistant."""
    existing_config = _load_existing_config()
    existing_secrets = _load_existing_secrets()
    is_reconfig = bool(existing_config)

    # Step 1: Welcome
    _step_welcome(is_reconfig)

    # Step 2: Identity
    bot_name, user_name, tone = _step_identity(existing_config)

    # Step 3: AI Provider
    provider, fallback_provider, configured_providers = _step_ai_provider(existing_config, existing_secrets)

    # Step 4: Bot's Brain (vault)
    vault_path, agent_dir = _step_vault(existing_config, bot_name)

    # Step 5: Integrations
    integrations_cfg, integration_secrets = _step_integrations(existing_config, existing_secrets)
    # Set obsidian vault_path to same as brain vault_path
    integrations_cfg["obsidian"]["vault_path"] = vault_path

    # Step 6: Safety Level
    safety_tier = _step_safety()
    permissions_cfg = {domain: safety_tier for domain in PERMISSION_DOMAINS}

    # Step 7: Platform
    platform, telegram_token, telegram_user_id = _step_platform(existing_config, existing_secrets)

    # --- Step 8: Save & Summary ---
    console.print("\n[bold]Saving configuration...[/bold]")

    # Ensure directories
    ROSHNI_DIR.mkdir(exist_ok=True)
    (ROSHNI_DIR / "logs").mkdir(exist_ok=True)
    (ROSHNI_DIR / "drafts" / "email").mkdir(parents=True, exist_ok=True)
    if integrations_cfg["google_workspace"]["enabled"]:
        Path(integrations_cfg["google_workspace"]["token_path"]).parent.mkdir(parents=True, exist_ok=True)

    # Scaffold vault
    from roshni.agent.vault import VaultManager

    vault = VaultManager(vault_path, agent_dir)
    vault.scaffold()
    click.echo(f"  Vault:   {vault.base_dir}/")

    persona_dir = vault.persona_dir

    # Build LLM config
    from roshni.core.llm.config import get_default_model

    llm_config: dict = {"default": provider}
    if fallback_provider:
        llm_config["fallback"] = fallback_provider
    providers_cfg: dict = {}
    for prov in configured_providers:
        providers_cfg[prov] = {"model": get_default_model(prov)}
    if provider == "local":
        providers_cfg["local"] = {"model": get_default_model("local")}
    llm_config["providers"] = providers_cfg

    config_data = {
        "bot": {"name": bot_name, "tone": tone},
        "user": {"name": user_name},
        "llm": llm_config,
        "platform": platform,
        "vault": {"path": vault_path, "agent_dir": agent_dir},
        "permissions": permissions_cfg,
        "paths": {
            "data_dir": str(ROSHNI_DIR),
            "log_dir": str(ROSHNI_DIR / "logs"),
            "persona_dir": str(persona_dir),
            "email_drafts_dir": str(ROSHNI_DIR / "drafts" / "email"),
            "reminders_path": str(ROSHNI_DIR / "reminders.json"),
        },
        "security": {"require_write_approval": safety_tier != "full"},
    }

    if platform == "telegram":
        config_data["telegram"] = {
            "allowed_user_ids": [str(telegram_user_id)] if telegram_user_id else [],
        }

    config_data["integrations"] = integrations_cfg

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    click.echo(f"  Config:  {CONFIG_PATH}")

    # Secrets
    secrets_data: dict = {
        "llm": {"api_keys": configured_providers} if configured_providers else {},
    }
    if telegram_token:
        secrets_data["telegram"] = {"bot_token": telegram_token}
    secrets_data.update(integration_secrets)

    with open(SECRETS_PATH, "w") as f:
        yaml.dump(secrets_data, f, default_flow_style=False, sort_keys=False)
    os.chmod(SECRETS_PATH, stat.S_IRUSR | stat.S_IWUSR)
    click.echo(f"  Secrets: {SECRETS_PATH} (mode 600)")

    # Persona files
    try:
        from roshni.agent.templates import (
            get_agents_template,
            get_identity_template,
            get_soul_template,
            get_user_template,
        )

        replacements = {"{bot_name}": bot_name, "{user_name}": user_name}

        def _apply(content: str) -> str:
            for k, v in replacements.items():
                content = content.replace(k, v)
            return content

        # For custom tones, fall back to "friendly" template
        template_tone = tone if tone in TONE_DESCRIPTIONS else "friendly"
        (persona_dir / "IDENTITY.md").write_text(_apply(get_identity_template(template_tone)))
        (persona_dir / "SOUL.md").write_text(_apply(get_soul_template()))
        (persona_dir / "USER.md").write_text(_apply(get_user_template()))
        (persona_dir / "AGENTS.md").write_text(_apply(get_agents_template()))
        click.echo(f"  Persona: {persona_dir}/")
    except Exception as e:
        click.echo(f"  Warning: Could not write persona files: {e}")

    # Summary
    prov_summary = PROVIDER_DISPLAY[provider]
    if fallback_provider:
        prov_summary += f" → {PROVIDER_DISPLAY[fallback_provider]}"

    enabled_integrations = [
        name
        for name in ["gmail", "trello", "notion", "healthkit", "fitbit"]
        if integrations_cfg.get(name, {}).get("enabled")
    ]
    if integrations_cfg.get("google_workspace", {}).get("enabled"):
        enabled_integrations.append("google workspace")

    safety_label = next(
        (info["label"] for info in SAFETY_LEVELS.values() if info["tier"] == safety_tier),
        safety_tier,
    )

    console.print()
    console.print(
        Panel(
            f"[bold]{bot_name}[/bold] is ready!\n"
            f"\n"
            f"  AI:            {prov_summary}\n"
            f"  Personality:   {tone}\n"
            f"  Brain:         {vault.base_dir}\n"
            f"  Safety:        {safety_label}\n"
            f"  Integrations:  {', '.join(enabled_integrations) if enabled_integrations else 'none yet'}\n"
            f"  Platform:      {platform}",
            title="Setup Complete",
            border_style="green",
        )
    )

    if platform == "telegram":
        click.echo("\nRun your bot:  roshni run")
    click.echo("Try terminal:  roshni chat")
