"""Generic Fitbit client builder for roshni-lib integrations.

This is a reusable skeleton for app/tool layers that want a standard Fitbit
client construction path. Apps can still provide their own refresh callbacks
and secret namespaces while migrating.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from roshni.core.secrets import SecretsManager


def build_fitbit_client(
    *,
    secrets: SecretsManager | None = None,
    namespace: str = "fitbit",
    secret_getter: Callable[[str], str | None] | None = None,
    refresh_cb: Callable[[dict[str, Any]], None] | None = None,
    expires_at: int = 0,
):
    """Build a Fitbit client from env vars and/or a SecretsManager.

    Environment variable names are intentionally compatible with existing app usage.
    Secret keys use dot notation under ``namespace`` (e.g. ``fitbit.client_id``).
    """
    from fitbit import Fitbit

    def _secret(key: str) -> str | None:
        if secret_getter is not None:
            return secret_getter(key)
        return secrets.get(f"{namespace}.{key}") if secrets else None

    client_id = os.environ.get("FITBIT_CLIENT_ID") or _secret("client_id")
    client_secret = os.environ.get("FITBIT_CLIENT_SECRET") or _secret("client_secret")
    access_token = os.environ.get("FITBIT_ACCESS_TOKEN") or _secret("access_token")
    refresh_token = os.environ.get("FITBIT_REFRESH_TOKEN") or _secret("refresh_token")

    if not all([client_id, client_secret, access_token, refresh_token]):
        raise ValueError(
            "Fitbit credentials required. Set FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET, "
            "FITBIT_ACCESS_TOKEN, FITBIT_REFRESH_TOKEN or configure secrets."
        )

    kwargs: dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
    }
    if refresh_cb is not None:
        kwargs["refresh_cb"] = refresh_cb

    return Fitbit(client_id, client_secret, **kwargs)


__all__ = ["build_fitbit_client"]
