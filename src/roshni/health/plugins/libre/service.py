"""Reusable Libre/CGM fetch + normalization logic."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")

TREND_MAP = {
    1: "falling_fast",
    2: "falling",
    3: "stable",
    4: "rising",
    5: "rising_fast",
}


def fetch_glucose_now_data(
    *,
    get_auth_with_retry: Callable[[], tuple[str, str]],
    get_patient_connections: Callable[[str, str], dict[str, Any]],
    get_cgm_data: Callable[[str, str, str], dict[str, Any]],
    logger: Any | None = None,
    now_tz: ZoneInfo = PACIFIC_TZ,
) -> dict[str, Any]:
    """Fetch current glucose + recent stats using injected provider callbacks."""
    now = datetime.now(now_tz)
    result: dict[str, Any] = {}

    token, account_id = get_auth_with_retry()

    connections = get_patient_connections(token, account_id)
    patients = connections.get("data", [])
    if not patients:
        raise ValueError("No Libre patient connections found")

    patient = patients[0]
    patient_id = patient.get("patientId")

    glucose_item = patient.get("glucoseMeasurement", {})
    if glucose_item:
        value = glucose_item.get("ValueInMgPerDl") or glucose_item.get("Value")
        trend_arrow = glucose_item.get("TrendArrow")
        timestamp_str = glucose_item.get("Timestamp", "")
        result["current"] = {
            "value": value,
            "trend": TREND_MAP.get(trend_arrow, "unknown"),
            "timestamp": timestamp_str,
        }
    else:
        result["status"] = "no_current_glucose_reading"
        result["warning"] = (
            "Libre connection succeeded, but no current glucose reading is available. "
            "Common causes include sensor not worn, warm-up, or sync delay (not necessarily credentials)."
        )

    try:
        cgm_response = get_cgm_data(token, account_id, patient_id)
        graph_data = cgm_response.get("data", {}).get("graphData", [])
        if graph_data:
            values = [r.get("ValueInMgPerDl") or r.get("Value") for r in graph_data]
            values = [v for v in values if v is not None]
            if values:
                in_range = sum(1 for v in values if 70 <= v <= 180)
                result["recent_hours"] = {
                    "readings": len(values),
                    "mean": round(sum(values) / len(values), 1),
                    "min": min(values),
                    "max": max(values),
                    "time_in_range_pct": round(in_range / len(values) * 100),
                }
    except Exception as e:
        if logger:
            logger.warning(f"Libre graph data failed (returning current only): {e}")

    result["as_of"] = now.strftime("%I:%M %p")

    if logger:
        logger.info(
            f"Glucose: {result.get('current', {}).get('value')} mg/dL, trend={result.get('current', {}).get('trend')}"
        )
    return result


__all__ = ["PACIFIC_TZ", "TREND_MAP", "fetch_glucose_now_data"]
