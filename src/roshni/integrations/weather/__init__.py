"""Weather integration providers."""

from .open_meteo import fetch_open_meteo_weather

__all__ = ["fetch_open_meteo_weather"]
