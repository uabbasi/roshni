"""Reusable Fitbit day-fetch + normalization logic.

This module is provider-agnostic apart from the Fitbit client method shape.
It expects a client object that exposes:
  - activities(date=...)
  - time_series(resource=..., base_date=..., period=...)
  - make_request(url)

App layers (e.g. roshni gateway) should own credential lookup, token persistence,
and any app-specific config sourcing, then call ``fetch_fitbit_day_data``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date as date_type
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

API_ENDPOINT = "https://api.fitbit.com"
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


def resolve_target_date(date: str = "today", *, now_tz: ZoneInfo = PACIFIC_TZ) -> tuple[date_type, bool]:
    """Resolve a date selector into a concrete date and whether it is today."""
    now_date = datetime.now(now_tz).date()
    if date in ("today", ""):
        target_date = now_date
    elif date == "yesterday":
        target_date = now_date - timedelta(days=1)
    else:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date!r}. Use 'today', 'yesterday', or YYYY-MM-DD.") from e
    return target_date, target_date == now_date


def compute_step_pace(
    current_steps: int,
    config: dict[str, Any] | None = None,
    *,
    now_tz: ZoneInfo = PACIFIC_TZ,
) -> dict[str, Any]:
    """Compute time-based step pace assessment."""
    cfg = config or {}
    now = datetime.now(now_tz)
    step_target = cfg.get("step_target", 4500)
    wake_hour = cfg.get("pace_wake_hour", 7)
    ramp_start = cfg.get("pace_ramp_start", 8)
    end_hour = cfg.get("pace_end_hour", 21)

    current_hour = now.hour + now.minute / 60.0

    if current_hour < wake_hour:
        expected = 0
    elif current_hour < ramp_start:
        frac = (current_hour - wake_hour) / (ramp_start - wake_hour)
        expected = int(step_target * 0.05 * frac)
    elif current_hour >= end_hour:
        expected = step_target
    else:
        frac = (current_hour - ramp_start) / (end_hour - ramp_start)
        expected = int(step_target * (0.05 + 0.95 * frac))

    if expected <= 0:
        pace_pct = 100 if current_steps == 0 else 999
    else:
        pace_pct = int(current_steps / expected * 100)

    if pace_pct >= 110:
        status = "ahead"
    elif pace_pct >= 75:
        status = "on_track"
    elif pace_pct >= 50:
        status = "behind"
    else:
        status = "critical"

    return {
        "current_steps": current_steps,
        "expected_steps_now": expected,
        "step_target": step_target,
        "pace_percentage": pace_pct,
        "pace_status": status,
        "as_of": now.strftime("%I:%M %p"),
    }


def fetch_fitbit_day_data(
    *,
    client: Any,
    date: str = "today",
    pace_config: dict[str, Any] | None = None,
    safe_fetch: Callable[[str, Callable[[], Any], list[str] | None], Any | None],
    logger: Any | None = None,
) -> dict[str, Any]:
    """Fetch and normalize Fitbit metrics for a given date selector.

    ``safe_fetch`` is injected by the app layer so retry/error semantics can remain
    app-specific during migration.
    """
    target_date, is_today = resolve_target_date(date)
    date_str = target_date.strftime("%Y-%m-%d")
    result: dict[str, Any] = {"date": date_str}
    failures: list[str] = []

    activities_raw = safe_fetch("activities", lambda: client.activities(date=target_date), failures)
    if activities_raw and "summary" in activities_raw:
        summary = activities_raw["summary"]
        fairly = summary.get("fairlyActiveMinutes", 0)
        very = summary.get("veryActiveMinutes", 0)
        distances = [
            {"activity": d.get("activity", ""), "distance": d.get("distance", 0)} for d in summary.get("distances", [])
        ]
        result["activity"] = {
            "steps": summary.get("steps"),
            "active_minutes": fairly + very,
            "calories_out": summary.get("caloriesOut"),
            "distances": distances,
            "resting_hr": summary.get("restingHeartRate"),
        }

    hr_raw = safe_fetch(
        "heart_rate_zones",
        lambda: client.time_series(resource="activities/heart", base_date=date_str, period="1d"),
        failures,
    )
    if hr_raw and hr_raw.get("activities-heart"):
        hr_entry = hr_raw["activities-heart"][0].get("value", {})
        hr_zones = [
            {"name": z.get("name", ""), "minutes": z.get("minutes", 0)} for z in hr_entry.get("heartRateZones", [])
        ]
        if hr_zones:
            result.setdefault("activity", {})["heart_rate_zones"] = hr_zones
        if hr_entry.get("restingHeartRate") and not result.get("activity", {}).get("resting_hr"):
            result.setdefault("activity", {})["resting_hr"] = hr_entry["restingHeartRate"]

    sleep_raw = safe_fetch(
        "sleep",
        lambda: client.make_request(f"{API_ENDPOINT}/1.2/user/-/sleep/date/{date_str}.json"),
        failures,
    )
    if sleep_raw and sleep_raw.get("sleep"):
        sleep_logs = sleep_raw["sleep"]
        main_sleep = next((s for s in sleep_logs if s.get("isMainSleep")), None)
        if not main_sleep:
            main_sleep = max(sleep_logs, key=lambda x: x.get("duration", 0))

        sleep_data: dict[str, Any] = {
            "hours": round(main_sleep.get("minutesAsleep", 0) / 60, 1),
            "efficiency": main_sleep.get("efficiency"),
        }
        if "levels" in main_sleep and "summary" in main_sleep["levels"]:
            s = main_sleep["levels"]["summary"]
            sleep_data["stages"] = {
                "deep": s.get("deep", {}).get("minutes", 0),
                "light": s.get("light", {}).get("minutes", 0),
                "rem": s.get("rem", {}).get("minutes", 0),
                "wake": s.get("wake", {}).get("minutes", 0),
            }
        result["sleep"] = sleep_data

    hrv_raw = safe_fetch(
        "HRV",
        lambda: client.make_request(f"{API_ENDPOINT}/1/user/-/hrv/date/{date_str}.json"),
        failures,
    )
    if hrv_raw and hrv_raw.get("hrv"):
        hrv_list = hrv_raw["hrv"]
        if hrv_list and "value" in hrv_list[0] and "dailyRmssd" in hrv_list[0]["value"]:
            result["hrv_rmssd"] = hrv_list[0]["value"]["dailyRmssd"]

    spo2_raw = safe_fetch(
        "SpO2",
        lambda: client.make_request(f"{API_ENDPOINT}/1/user/-/spo2/date/{date_str}.json"),
        failures,
    )
    if spo2_raw:
        spo2_val = spo2_raw.get("value") or spo2_raw
        if isinstance(spo2_val, dict) and "avg" in spo2_val:
            result["spo2"] = {
                "avg": spo2_val.get("avg"),
                "min": spo2_val.get("min"),
                "max": spo2_val.get("max"),
            }

    br_raw = safe_fetch(
        "breathing_rate",
        lambda: client.make_request(f"{API_ENDPOINT}/1/user/-/br/date/{date_str}.json"),
        failures,
    )
    if br_raw and br_raw.get("br"):
        br_list = br_raw["br"]
        if br_list and "value" in br_list[0]:
            result["breathing_rate"] = br_list[0]["value"].get("breathingRate")

    steps = result.get("activity", {}).get("steps")
    if steps is not None and is_today:
        result["step_pace"] = compute_step_pace(steps, pace_config or {})

    total_endpoints = 6
    if failures:
        if len(failures) >= total_endpoints:
            result["error"] = "Fitbit data fetch failed (all endpoints failed in this tool call)."
            result["error_type"] = "fitbit_tool_or_api_unavailable"
            result["failed_endpoints"] = failures
            result["diagnostic_note"] = (
                "Cause is not determined by this response alone (could be API, token, network, or tool issue). "
                "Do not assume re-authentication is required without a credential-specific error."
            )
            if logger:
                logger.error(f"Fitbit: ALL {len(failures)} endpoints failed: {failures}")
        else:
            result["warnings"] = f"Some Fitbit endpoints failed: {', '.join(failures)}"
            result["failed_endpoints"] = failures
            if logger:
                logger.warning(f"Fitbit: {len(failures)}/{total_endpoints} endpoints failed: {failures}")

    if logger:
        logger.info(
            f"Fitbit day: steps={result.get('activity', {}).get('steps')}, "
            f"sleep={result.get('sleep', {}).get('hours')}h, "
            f"hrv={result.get('hrv_rmssd')}"
        )
    return result


__all__ = ["API_ENDPOINT", "PACIFIC_TZ", "compute_step_pace", "fetch_fitbit_day_data", "resolve_target_date"]
