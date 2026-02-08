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
                "Let's set up your personal AI assistant.\nThis takes about 2 minutes.",
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
            telegram_user_id = click.prompt("Your Telegram user ID")

    # --- Integrations ---
    click.echo("\nOptional integrations (you can add these later):")
    integrations: dict = existing_config.get("integrations", {}) or {}

    # Gmail
    gmail_enabled = integrations.get("gmail", {}).get("enabled", False)
    gmail_enabled = click.confirm("  Enable Gmail (send emails)?", default=gmail_enabled)
    gmail_address = existing_secrets.get("gmail", {}).get("address", "")
    gmail_app_password = existing_secrets.get("gmail", {}).get("app_password", "")
    if gmail_enabled:
        gmail_address = click.prompt("    Gmail address", default=gmail_address)
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

    # --- Generate files ---
    click.echo("\nSaving configuration...")

    # Ensure directories
    ROSHNI_DIR.mkdir(exist_ok=True)
    PERSONA_DIR.mkdir(exist_ok=True)
    (ROSHNI_DIR / "notes").mkdir(exist_ok=True)
    (ROSHNI_DIR / "logs").mkdir(exist_ok=True)

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
        "paths": {
            "data_dir": str(ROSHNI_DIR),
            "notes_dir": str(ROSHNI_DIR / "notes"),
            "log_dir": str(ROSHNI_DIR / "logs"),
            "persona_dir": str(PERSONA_DIR),
        },
    }

    if platform == "telegram":
        config_data["telegram"] = {
            "allowed_user_ids": [str(telegram_user_id)] if telegram_user_id else [],
        }

    config_data["integrations"] = {
        "gmail": {"enabled": gmail_enabled},
        "obsidian": {
            "enabled": obsidian_enabled,
            "vault_path": obsidian_path,
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
    if gmail_enabled and gmail_app_password:
        secrets_data["gmail"] = {
            "address": gmail_address,
            "app_password": gmail_app_password,
        }

    with open(SECRETS_PATH, "w") as f:
        yaml.dump(secrets_data, f, default_flow_style=False, sort_keys=False)
    os.chmod(SECRETS_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 600
    click.echo(f"  Secrets: {SECRETS_PATH} (mode 600)")

    # Persona files
    try:
        from roshni.agent.templates import get_identity_template, get_soul_template, get_user_template

        identity_content = get_identity_template(tone).replace("{bot_name}", bot_name).replace("{user_name}", user_name)
        soul_content = get_soul_template().replace("{bot_name}", bot_name).replace("{user_name}", user_name)
        user_content = get_user_template().replace("{bot_name}", bot_name).replace("{user_name}", user_name)

        (PERSONA_DIR / "IDENTITY.md").write_text(identity_content)
        (PERSONA_DIR / "SOUL.md").write_text(soul_content)
        (PERSONA_DIR / "USER.md").write_text(user_content)
        click.echo(f"  Persona: {PERSONA_DIR}/")
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

    console.print(
        Panel(
            f"Bot name: {bot_name}\n"
            f"Tone: {tone}\n"
            f"Provider: {providers_summary}\n"
            f"Platform: {platform}\n"
            f"Gmail: {'enabled' if gmail_enabled else 'disabled'}\n"
            f"Obsidian: {'enabled' if obsidian_enabled else 'disabled'}",
            title="Setup Complete",
        )
    )

    if platform == "telegram":
        click.echo("\nRun your bot:  roshni run")
    click.echo("Try terminal:  roshni chat")
