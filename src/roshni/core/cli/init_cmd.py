"""roshni init — interactive setup wizard."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import click
import yaml

ROSHNI_DIR = Path.home() / ".roshni"
CONFIG_PATH = ROSHNI_DIR / "config.yaml"
SECRETS_PATH = ROSHNI_DIR / "secrets.yaml"
PERSONA_DIR = ROSHNI_DIR / "persona"


def _load_existing_config() -> dict:
    """Load existing config if present."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_existing_secrets() -> dict:
    """Load existing secrets if present."""
    if SECRETS_PATH.exists():
        with open(SECRETS_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _mask_key(key: str) -> str:
    """Show first 4 and last 4 chars of a key."""
    if len(key) <= 12:
        return key[:4] + "..." + key[-2:]
    return key[:4] + "..." + key[-4:]


def _validate_api_key(provider: str, api_key: str) -> bool | None:
    """Make a test LLM call to validate the API key.

    Returns True if valid, False if invalid, None if validation was skipped.
    """
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
        # litellm not installed — skip validation
        return None
    except Exception as e:
        click.echo(f"  API key validation failed: {e}")
        return False


def _prompt_choice(prompt: str, choices: list[str], default: str = "") -> str:
    """Prompt for a choice from a numbered list."""
    click.echo(f"\n{prompt}")
    for i, c in enumerate(choices, 1):
        marker = " (default)" if c == default else ""
        click.echo(f"  {i}. {c}{marker}")

    while True:
        raw = click.prompt("  Choose", default=str(choices.index(default) + 1) if default in choices else "1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            # Maybe they typed the name
            if raw in choices:
                return raw
        click.echo(f"  Please enter a number 1-{len(choices)}")


@click.command()
def init() -> None:
    """Set up your personal AI assistant."""
    try:
        from rich.console import Console
        from rich.panel import Panel
    except ImportError:
        raise click.ClickException("Install rich: pip install rich")

    console = Console()
    existing_config = _load_existing_config()
    existing_secrets = _load_existing_secrets()
    is_reconfig = bool(existing_config)

    # Welcome
    if is_reconfig:
        console.print(Panel("Reconfiguring your assistant. Existing values shown as defaults.", title="Roshni Setup"))
    else:
        console.print(
            Panel(
                "Let's set up your personal AI assistant.\nThis takes about 5 minutes.",
                title="Welcome to Roshni",
            )
        )

    # --- Bot name ---
    default_name = existing_config.get("bot", {}).get("name", "Roshni")
    bot_name = click.prompt("\nWhat should your bot be called?", default=default_name)

    # --- User name ---
    default_user = existing_config.get("user", {}).get("name", "")
    user_name = click.prompt("What's your name?", default=default_user or os.environ.get("USER", ""))

    # --- Tone ---
    tones = ["friendly", "professional", "warm", "witty"]
    tone_descriptions = {
        "friendly": "Casual and approachable — like a helpful friend",
        "professional": "Precise and efficient — like a great executive assistant",
        "warm": "Empathetic and thoughtful — calm and caring",
        "witty": "Sharp and clever — dry humor with substance",
    }
    default_tone = existing_config.get("bot", {}).get("tone", "friendly")
    click.echo("\nWhat personality should your bot have?")
    for i, t in enumerate(tones, 1):
        marker = " (default)" if t == default_tone else ""
        click.echo(f"  {i}. {t} — {tone_descriptions[t]}{marker}")

    tone_input = click.prompt("  Choose", default=str(tones.index(default_tone) + 1))
    try:
        tone = tones[int(tone_input) - 1]
    except (ValueError, IndexError):
        tone = tone_input if tone_input in tones else default_tone

    # --- Security ---
    security_cfg = existing_config.get("security", {}) or {}
    require_write_approval = click.confirm(
        "\nRequire approval before write/send actions? (recommended)",
        default=security_cfg.get("require_write_approval", True),
    )

    # --- LLM providers ---
    from roshni.core.llm.config import PROVIDER_ENV_MAP

    all_providers = ["anthropic", "openai", "gemini", "deepseek", "xai", "groq", "local"]
    provider_names = {
        "anthropic": "Anthropic (Claude)",
        "openai": "OpenAI (GPT)",
        "gemini": "Google (Gemini)",
        "deepseek": "DeepSeek",
        "xai": "xAI (Grok)",
        "groq": "Groq",
        "local": "Local (Ollama)",
    }

    # Read existing multi-provider config or legacy format
    existing_llm = existing_config.get("llm", {})
    default_provider = existing_llm.get("default", "") or existing_llm.get("provider", "anthropic")

    click.echo("\nWhich AI provider do you want as your default?")
    for i, p in enumerate(all_providers, 1):
        marker = " (default)" if p == default_provider else ""
        click.echo(f"  {i}. {provider_names[p]}{marker}")

    default_idx = str(all_providers.index(default_provider) + 1) if default_provider in all_providers else "1"
    provider_input = click.prompt("  Choose", default=default_idx)
    try:
        provider = all_providers[int(provider_input) - 1]
    except (ValueError, IndexError):
        provider = provider_input if provider_input in all_providers else default_provider

    # --- API keys (multi-provider) ---
    # Read existing keys from new or legacy format
    existing_api_keys: dict = existing_secrets.get("llm", {}).get("api_keys", {}) or {}
    if not existing_api_keys:
        legacy_key = existing_secrets.get("llm", {}).get("api_key", "")
        legacy_provider = existing_llm.get("provider", "")
        if legacy_key and legacy_provider:
            existing_api_keys = {legacy_provider: legacy_key}

    configured_providers: dict[str, str] = {}  # provider -> api_key

    def _prompt_api_key(prov: str, existing: str = "") -> str:
        """Prompt for an API key for a provider."""
        if existing:
            click.echo(f"\n  {provider_names[prov]} key on file: {_mask_key(existing)}")
            if not click.confirm("  Change it?", default=False):
                return existing

        env_var = PROVIDER_ENV_MAP.get(prov, "")
        hint = f" (or set {env_var})" if env_var else ""
        key = click.prompt(f"  Paste your {provider_names[prov]} API key{hint}", hide_input=True)

        click.echo("  Validating...")
        validation = _validate_api_key(prov, key)
        if validation is True:
            console.print("  [green]Valid![/green]")
        elif validation is None:
            click.echo("  Skipped (install roshni\\[llm] to validate)")
        else:
            if not click.confirm("  Key validation failed. Use it anyway?", default=False):
                raise click.Abort()
        return key

    # Default provider API key (skip for local)
    if provider != "local":
        existing_key = existing_api_keys.get(provider, "")
        configured_providers[provider] = _prompt_api_key(provider, existing_key)

    # --- Fallback provider ---
    existing_fallback = existing_llm.get("fallback", "")
    fallback_provider = ""
    fallback_choices = [p for p in all_providers if p != provider and p != "local"]

    if click.confirm("\nAdd a fallback provider? (used when default fails)", default=bool(existing_fallback)):
        click.echo("  Choose fallback provider:")
        for i, p in enumerate(fallback_choices, 1):
            marker = " (current)" if p == existing_fallback else ""
            click.echo(f"    {i}. {provider_names[p]}{marker}")

        fb_idx = fallback_choices.index(existing_fallback) + 1 if existing_fallback in fallback_choices else 1
        fb_default = str(fb_idx)
        fb_input = click.prompt("    Choose", default=fb_default)
        try:
            fallback_provider = fallback_choices[int(fb_input) - 1]
        except (ValueError, IndexError):
            fallback_provider = fb_input if fb_input in fallback_choices else ""

        if fallback_provider:
            existing_fb_key = existing_api_keys.get(fallback_provider, "")
            configured_providers[fallback_provider] = _prompt_api_key(fallback_provider, existing_fb_key)

    # --- Additional providers ---
    remaining = [p for p in all_providers if p not in configured_providers and p != "local"]
    while remaining and click.confirm("\nAdd another provider?", default=False):
        click.echo("  Available providers:")
        for i, p in enumerate(remaining, 1):
            click.echo(f"    {i}. {provider_names[p]}")

        extra_input = click.prompt("    Choose", default="1")
        try:
            extra_provider = remaining[int(extra_input) - 1]
        except (ValueError, IndexError):
            continue

        existing_extra_key = existing_api_keys.get(extra_provider, "")
        configured_providers[extra_provider] = _prompt_api_key(extra_provider, existing_extra_key)
        remaining = [p for p in remaining if p != extra_provider]

    # --- Platform ---
    default_platform = existing_config.get("platform", "telegram")
    platform = _prompt_choice(
        "Where will your bot live?",
        ["telegram", "terminal-only"],
        default=default_platform,
    )

    # --- Telegram setup ---
    telegram_token = existing_secrets.get("telegram", {}).get("bot_token", "")
    tg_cfg = existing_config.get("telegram", {})
    tg_uids = tg_cfg.get("allowed_user_ids", [])
    telegram_user_id = tg_uids[0] if tg_uids else ""

    if platform == "telegram":
        if not telegram_token or click.confirm("\nSet up Telegram bot token?", default=not bool(telegram_token)):
            console.print(
                Panel(
                    "1. Open Telegram and search for @BotFather\n"
                    "2. Send /newbot and follow the prompts\n"
                    "3. Copy the token BotFather gives you",
                    title="Create a Telegram Bot",
                )
            )
            telegram_token = click.prompt("Paste your bot token")

        if not telegram_user_id or click.confirm("Set up your Telegram user ID?", default=not bool(telegram_user_id)):
            console.print(
                Panel(
                    "1. Open Telegram and search for @userinfobot\n"
                    "2. Send it any message\n"
                    "3. It will reply with your user ID (a number)",
                    title="Find Your Telegram User ID",
                )
            )
            while True:
                telegram_user_id = click.prompt("Your Telegram user ID").strip()
                if telegram_user_id.isdigit():
                    break
                click.echo("Please enter numbers only (example: 123456789).")

        if not telegram_user_id:
            raise click.ClickException(
                "Telegram requires an allowed user ID for security. Re-run and provide your Telegram user ID."
            )

    # --- Vault setup ---
    existing_vault = existing_config.get("vault", {}) or {}
    existing_vault_path = existing_vault.get("path", "")
    existing_agent_dir = existing_vault.get("agent_dir", "")

    click.echo("\nVault configuration (the agent's brain lives in your Obsidian vault):")
    vault_path = click.prompt(
        "  Path to your Obsidian vault",
        default=existing_vault_path or "~/Obsidian",
    )
    vault_path = str(Path(vault_path).expanduser())

    agent_dir = click.prompt(
        "  Agent directory name in vault",
        default=existing_agent_dir or bot_name.lower().replace(" ", "-"),
    )

    # --- Integrations ---
    click.echo("\nOptional integrations (you can add these later):")
    integrations: dict = existing_config.get("integrations", {}) or {}

    # Gmail
    gmail_cfg = integrations.get("gmail", {}) or {}
    gmail_enabled = click.confirm(
        "  Enable Gmail assistant? (draft mode by default)",
        default=gmail_cfg.get("enabled", False),
    )
    gmail_mode = "draft"
    gmail_allow_send = False
    gmail_address = existing_secrets.get("gmail", {}).get("address", "")
    gmail_app_password = existing_secrets.get("gmail", {}).get("app_password", "")
    if gmail_enabled:
        click.echo("    Gmail default is draft-only (no sending).")
        gmail_address = click.prompt("    Gmail address (for drafting context)", default=gmail_address)
        gmail_allow_send = click.confirm(
            "    Allow direct sending too? (higher risk)",
            default=gmail_cfg.get("allow_send", False),
        )
        if gmail_allow_send:
            gmail_mode = "send"
            change_pw = not gmail_app_password or click.confirm(
                "    Set/change Gmail App Password?",
                default=not bool(gmail_app_password),
            )
            if change_pw:
                console.print(
                    Panel(
                        "1. Go to myaccount.google.com → Security → App Passwords\n"
                        "2. Create a new app password for 'Mail'\n"
                        "3. Copy the 16-character password",
                        title="Gmail App Password",
                    )
                )
                gmail_app_password = click.prompt("    App Password", hide_input=True)

    # Obsidian
    obsidian_enabled = integrations.get("obsidian", {}).get("enabled", False)
    obsidian_path = integrations.get("obsidian", {}).get("vault_path", "")
    obsidian_enabled = click.confirm("  Enable Obsidian vault search?", default=obsidian_enabled)
    if obsidian_enabled:
        obsidian_path = click.prompt("    Path to Obsidian vault", default=obsidian_path)
        obsidian_path = str(Path(obsidian_path).expanduser())

    # Trello
    trello_cfg = integrations.get("trello", {}) or {}
    trello_enabled = click.confirm(
        "  Enable Trello project tools? (boards/lists/cards/labels/comments)",
        default=trello_cfg.get("enabled", False),
    )
    trello_disable_board_delete = bool(trello_cfg.get("disable_board_delete", False))
    trello_api_key = existing_secrets.get("trello", {}).get("api_key", "")
    trello_token = existing_secrets.get("trello", {}).get("token", "")
    if trello_enabled:
        if not trello_api_key or click.confirm("    Set/change Trello API key?", default=not bool(trello_api_key)):
            console.print(
                Panel(
                    "1. Visit https://trello.com/power-ups/admin\n"
                    "2. Open your Power-Up / API page and copy your API key",
                    title="Trello API Key",
                )
            )
            trello_api_key = click.prompt("    Trello API key", hide_input=True)
        if not trello_token or click.confirm("    Set/change Trello token?", default=not bool(trello_token)):
            console.print(
                Panel(
                    "1. Open https://trello.com/1/authorize with your key\n"
                    "2. Generate a token with read/write scopes\n"
                    "3. Copy the token value",
                    title="Trello Token",
                )
            )
            trello_token = click.prompt("    Trello token", hide_input=True)
        trello_disable_board_delete = click.confirm(
            "    Disable permanent board deletion tool? (recommended)",
            default=trello_disable_board_delete,
        )

    # Notion
    notion_cfg = integrations.get("notion", {}) or {}
    notion_enabled = click.confirm(
        "  Enable Notion knowledge tools? (search/create/update pages)",
        default=notion_cfg.get("enabled", False),
    )
    notion_database_id = notion_cfg.get("database_id", "")
    notion_title_property = notion_cfg.get("title_property", "Name")
    notion_token = existing_secrets.get("notion", {}).get("token", "")
    if notion_enabled:
        notion_database_id = click.prompt("    Notion database ID", default=notion_database_id)
        notion_title_property = click.prompt("    Notion title property name", default=notion_title_property)
        if not notion_token or click.confirm(
            "    Set/change Notion integration token?",
            default=not bool(notion_token),
        ):
            console.print(
                Panel(
                    "1. Create a Notion internal integration at https://www.notion.so/my-integrations\n"
                    "2. Copy the integration token\n"
                    "3. Share your target database with that integration",
                    title="Notion Token",
                )
            )
            notion_token = click.prompt("    Notion token", hide_input=True)

    # HealthKit (Apple Health export)
    healthkit_cfg = integrations.get("healthkit", {}) or {}
    healthkit_enabled = click.confirm(
        "  Enable HealthKit import via Apple Health export.xml?",
        default=healthkit_cfg.get("enabled", False),
    )
    healthkit_export_path = str(
        Path(healthkit_cfg.get("export_path", "~/Downloads/apple_health_export/export.xml")).expanduser()
    )
    if healthkit_enabled:
        healthkit_export_path = str(
            Path(
                click.prompt(
                    "    Path to Apple Health export.xml",
                    default=healthkit_export_path,
                )
            ).expanduser()
        )

    # Builtins + delighters
    builtins_cfg = integrations.get("builtins", {}) or {}
    delighters_cfg = integrations.get("delighters", {}) or {}
    builtins_enabled = click.confirm(
        "  Enable builtins (weather, web search/fetch)?",
        default=builtins_cfg.get("enabled", True),
    )
    delighters_enabled = click.confirm(
        "  Enable delighters (morning brief, plans, reminders)?",
        default=delighters_cfg.get("enabled", True),
    )

    # Google Workspace profile (least-privilege defaults)
    google_cfg = integrations.get("google_workspace", {}) or {}
    google_enabled = click.confirm(
        "  Enable Google Workspace profile? (Gmail drafts, Calendar, Docs, Sheets)",
        default=google_cfg.get("enabled", False),
    )
    google_credentials_path = str(
        Path(google_cfg.get("credentials_path", "~/.roshni/google/client_secret.json")).expanduser()
    )
    google_token_path = str(Path(google_cfg.get("token_path", "~/.roshni/google/token.pickle")).expanduser())
    google_scopes = google_cfg.get("scopes", {}) or {}
    google_gmail_drafts = bool(google_scopes.get("gmail_drafts", True))
    google_calendar_rw = bool(google_scopes.get("calendar_rw", True))
    google_docs_ro = bool(google_scopes.get("docs_readonly", True))
    google_sheets_ro = bool(google_scopes.get("sheets_readonly", True))
    if google_enabled:
        click.echo("    Least-privilege defaults are preselected.")
        google_credentials_path = str(
            Path(
                click.prompt(
                    "    OAuth client credentials JSON path",
                    default=google_credentials_path,
                )
            ).expanduser()
        )
        google_token_path = str(
            Path(
                click.prompt(
                    "    OAuth token cache path",
                    default=google_token_path,
                )
            ).expanduser()
        )
        google_gmail_drafts = click.confirm("    Gmail draft access", default=google_gmail_drafts)
        google_calendar_rw = click.confirm("    Calendar read/write access", default=google_calendar_rw)
        google_docs_ro = click.confirm("    Docs read-only access", default=google_docs_ro)
        google_sheets_ro = click.confirm("    Sheets read-only access", default=google_sheets_ro)

    # --- Permission tiers ---
    existing_perms = existing_config.get("permissions", {}) or {}
    tier_choices = ["observe", "interact", "full"]

    click.echo("\nPermission levels for enabled integrations:")
    click.echo("  (observe=read-only, interact=read+writes, full=everything)")

    permissions_cfg: dict[str, str] = {}
    domain_defaults = {
        "gmail": "interact",
        "obsidian": "interact",
        "trello": "interact",
        "notion": "interact",
        "health": "observe",
        "tasks": "interact",
        "vault": "interact",
    }
    enabled_domains: list[tuple[str, bool]] = [
        ("gmail", gmail_enabled),
        ("trello", trello_enabled),
        ("notion", notion_enabled),
        ("obsidian", obsidian_enabled),
        ("health", healthkit_enabled),
    ]
    # Tasks and vault are always-on when vault is configured
    enabled_domains.append(("tasks", True))
    enabled_domains.append(("vault", True))

    for domain, enabled in enabled_domains:
        if not enabled:
            permissions_cfg[domain] = "none"
            continue
        default_tier = existing_perms.get(domain, domain_defaults.get(domain, "interact"))
        tier = _prompt_choice(
            f"  {domain.title()} permission level?",
            tier_choices,
            default=default_tier,
        )
        permissions_cfg[domain] = tier

    # --- Generate files ---
    click.echo("\nSaving configuration...")

    # Ensure directories
    ROSHNI_DIR.mkdir(exist_ok=True)
    (ROSHNI_DIR / "logs").mkdir(exist_ok=True)
    (ROSHNI_DIR / "drafts" / "email").mkdir(parents=True, exist_ok=True)
    if google_enabled:
        Path(google_token_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    # Scaffold vault
    from roshni.agent.vault import VaultManager

    vault = VaultManager(vault_path, agent_dir)
    vault.scaffold()
    click.echo(f"  Vault:   {vault.base_dir}/")

    # Persona dir points into vault now
    persona_dir = vault.persona_dir

    # Config — new multi-provider format
    from roshni.core.llm.config import get_default_model

    llm_config: dict = {
        "default": provider,
    }
    if fallback_provider:
        llm_config["fallback"] = fallback_provider

    # Build providers section with default models
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
        "vault": {
            "path": vault_path,
            "agent_dir": agent_dir,
        },
        "permissions": permissions_cfg,
        "paths": {
            "data_dir": str(ROSHNI_DIR),
            "log_dir": str(ROSHNI_DIR / "logs"),
            "persona_dir": str(persona_dir),
            "email_drafts_dir": str(ROSHNI_DIR / "drafts" / "email"),
            "reminders_path": str(ROSHNI_DIR / "reminders.json"),
        },
        "security": {"require_write_approval": require_write_approval},
    }

    if platform == "telegram":
        config_data["telegram"] = {
            "allowed_user_ids": [str(telegram_user_id)] if telegram_user_id else [],
        }

    config_data["integrations"] = {
        "gmail": {
            "enabled": gmail_enabled,
            "mode": gmail_mode,
            "allow_send": gmail_allow_send,
        },
        "obsidian": {
            "enabled": obsidian_enabled,
            "vault_path": obsidian_path,
        },
        "builtins": {"enabled": builtins_enabled},
        "delighters": {"enabled": delighters_enabled},
        "google_workspace": {
            "enabled": google_enabled,
            "credentials_path": google_credentials_path,
            "token_path": google_token_path,
            "scopes": {
                "gmail_drafts": google_gmail_drafts,
                "calendar_rw": google_calendar_rw,
                "docs_readonly": google_docs_ro,
                "sheets_readonly": google_sheets_ro,
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
    }

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    click.echo(f"  Config:  {CONFIG_PATH}")

    # Secrets — new multi-provider format
    secrets_data: dict = {
        "llm": {"api_keys": configured_providers} if configured_providers else {},
    }
    if telegram_token:
        secrets_data["telegram"] = {"bot_token": telegram_token}
    if gmail_enabled:
        secrets_data["gmail"] = {
            "address": gmail_address,
        }
        if gmail_allow_send and gmail_app_password:
            secrets_data["gmail"]["app_password"] = gmail_app_password
    if trello_enabled:
        secrets_data["trello"] = {
            "api_key": trello_api_key,
            "token": trello_token,
        }
    if notion_enabled:
        secrets_data["notion"] = {
            "token": notion_token,
        }

    with open(SECRETS_PATH, "w") as f:
        yaml.dump(secrets_data, f, default_flow_style=False, sort_keys=False)
    os.chmod(SECRETS_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 600
    click.echo(f"  Secrets: {SECRETS_PATH} (mode 600)")

    # Persona files — written into vault
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

        (persona_dir / "IDENTITY.md").write_text(_apply(get_identity_template(tone)))
        (persona_dir / "SOUL.md").write_text(_apply(get_soul_template()))
        (persona_dir / "USER.md").write_text(_apply(get_user_template()))
        (persona_dir / "AGENTS.md").write_text(_apply(get_agents_template()))
        click.echo(f"  Persona: {persona_dir}/")
    except Exception as e:
        click.echo(f"  Warning: Could not write persona files: {e}")

    # Summary
    click.echo("")
    providers_summary = provider_names[provider]
    if fallback_provider:
        providers_summary += f" (fallback: {provider_names[fallback_provider]})"
    extra = [p for p in configured_providers if p != provider and p != fallback_provider]
    if extra:
        providers_summary += f" + {', '.join(provider_names[p] for p in extra)}"

    if gmail_enabled and not gmail_allow_send:
        gmail_status = "draft-only"
    elif gmail_enabled:
        gmail_status = "send+draft"
    else:
        gmail_status = "disabled"

    # Build permission summary
    perm_summary = ", ".join(f"{d}={t}" for d, t in permissions_cfg.items() if t != "none")

    console.print(
        Panel(
            f"Bot name: {bot_name}\n"
            f"Tone: {tone}\n"
            f"Provider: {providers_summary}\n"
            f"Platform: {platform}\n"
            f"Vault: {vault.base_dir}\n"
            f"Permissions: {perm_summary}\n"
            f"Write approvals: {'enabled' if require_write_approval else 'disabled'}\n"
            f"Gmail: {gmail_status}\n"
            f"Obsidian: {'enabled' if obsidian_enabled else 'disabled'}\n"
            f"Trello: {'enabled' if trello_enabled else 'disabled'}\n"
            f"Notion: {'enabled' if notion_enabled else 'disabled'}\n"
            f"HealthKit: {'enabled' if healthkit_enabled else 'disabled'}\n"
            f"Builtins: {'enabled' if builtins_enabled else 'disabled'}\n"
            f"Delighters: {'enabled' if delighters_enabled else 'disabled'}\n"
            f"Google profile: {'enabled' if google_enabled else 'disabled'}",
            title="Setup Complete",
        )
    )

    if platform == "telegram":
        click.echo("\nRun your bot:  roshni run")
    click.echo("Try terminal:  roshni chat")
