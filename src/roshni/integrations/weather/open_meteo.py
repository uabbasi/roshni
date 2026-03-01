"""Open-Meteo weather provider helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "Clear",
    1: "Mostly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Foggy",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    56: "Freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Heavy rain showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with hail",
}


def assess_outdoor(temp: float, precip_prob: int, uv_index: float, wind_speed: float) -> str:
    """Generate outdoor activity assessment from conditions."""
    issues: list[str] = []
    if temp < 40:
        issues.append("very cold")
    elif temp < 50:
        issues.append("chilly")
    elif temp > 95:
        issues.append("extreme heat")
    elif temp > 85:
        issues.append("hot")

    if precip_prob > 60:
        issues.append("likely rain")
    elif precip_prob > 30:
        issues.append("chance of rain")

    if uv_index >= 8:
        issues.append("very high UV — sunscreen essential")
    elif uv_index >= 6:
        issues.append("high UV — wear sunscreen")

    if wind_speed > 25:
        issues.append("very windy")
    elif wind_speed > 15:
        issues.append("breezy")

    if not issues:
        return "Good conditions for a walk — mild, low chance of rain"
    if len(issues) == 1 and issues[0] in ("breezy", "chance of rain", "chilly"):
        return f"Decent for outdoor activity — {issues[0]}"
    return f"Caution outdoors — {', '.join(issues)}"


def _format_time(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%I:%M %p")
    except Exception:
        return iso_str


def fetch_open_meteo_weather(
    *,
    latitude: float,
    longitude: float,
    requests_get,
    logger: Any | None = None,
    now_tz: ZoneInfo = PACIFIC_TZ,
) -> dict[str, Any]:
    """Fetch current weather and hourly forecast from Open-Meteo."""
    now = datetime.now(now_tz)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles",
        "current": ["temperature_2m", "weather_code", "wind_speed_10m", "relative_humidity_2m", "uv_index"],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "sunrise",
            "sunset",
        ],
        "hourly": ["temperature_2m", "precipitation_probability", "uv_index"],
        "forecast_days": 1,
    }
    response = requests_get(WEATHER_API_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    current = data.get("current", {})
    temp_current = current.get("temperature_2m", 0)
    weather_code = current.get("weather_code", 0)
    wind_speed = current.get("wind_speed_10m", 0)
    humidity = current.get("relative_humidity_2m", 0)
    uv_now = current.get("uv_index", 0)

    daily = data.get("daily", {})
    temp_high = daily.get("temperature_2m_max", [0])[0]
    temp_low = daily.get("temperature_2m_min", [0])[0]
    precip_prob = daily.get("precipitation_probability_max", [0])[0] or 0
    sunrise_raw = daily.get("sunrise", [""])[0]
    sunset_raw = daily.get("sunset", [""])[0]

    hourly = data.get("hourly", {})
    hourly_times = hourly.get("time", [])
    hourly_temps = hourly.get("temperature_2m", [])
    hourly_precip = hourly.get("precipitation_probability", [])
    hourly_uv = hourly.get("uv_index", [])

    current_hour = now.hour
    hourly_forecast: list[dict[str, Any]] = []
    for i, t in enumerate(hourly_times):
        try:
            hour_dt = datetime.fromisoformat(t)
            if hour_dt.hour > current_hour and len(hourly_forecast) < 6:
                hourly_forecast.append(
                    {
                        "hour": hour_dt.strftime("%I %p").lstrip("0"),
                        "temp": round(hourly_temps[i], 1) if i < len(hourly_temps) else None,
                        "precip_prob": hourly_precip[i] if i < len(hourly_precip) else None,
                        "uv_index": round(hourly_uv[i], 1) if i < len(hourly_uv) else None,
                    }
                )
        except Exception:
            continue

    conditions = WEATHER_CODES.get(weather_code, "Unknown")
    result = {
        "current": {
            "temperature": round(temp_current, 1),
            "conditions": conditions,
            "wind_speed": round(wind_speed, 1),
            "humidity": int(humidity),
        },
        "today": {
            "high": round(temp_high, 1),
            "low": round(temp_low, 1),
            "precipitation_prob": int(precip_prob),
            "sunrise": _format_time(sunrise_raw),
            "sunset": _format_time(sunset_raw),
        },
        "hourly_forecast": hourly_forecast,
        "outdoor_assessment": assess_outdoor(temp_current, precip_prob, uv_now, wind_speed),
        "as_of": now.strftime("%I:%M %p"),
    }
    if logger:
        logger.info(
            f"Weather: {conditions}, {temp_current}°F, high {temp_high}°F/low {temp_low}°F, {precip_prob}% precip"
        )
    return result


__all__ = ["PACIFIC_TZ", "WEATHER_API_URL", "WEATHER_CODES", "assess_outdoor", "fetch_open_meteo_weather"]
