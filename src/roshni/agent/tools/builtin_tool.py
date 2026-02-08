"""Built-in utility tools: weather + web search/fetch."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from roshni.agent.tools import ToolDefinition


class _TextExtractor(HTMLParser):
    """Minimal HTML to text extractor."""

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.parts.append(text)


def _http_get(url: str, timeout: int = 10) -> bytes:
    req = urllib.request.Request(
        url=url,
        headers={"User-Agent": "RoshniBot/0.1 (https://github.com/uabbasi/roshni)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _weather(location: str) -> str:
    loc = urllib.parse.quote(location.strip())
    if not loc:
        return "Please provide a location."
    url = f"https://wttr.in/{loc}?format=j1"
    try:
        data = json.loads(_http_get(url).decode("utf-8", errors="ignore"))
    except Exception as e:
        return f"Failed to fetch weather: {e}"

    current = (data.get("current_condition") or [{}])[0]
    weather_desc = (current.get("weatherDesc") or [{"value": "Unknown"}])[0].get("value", "Unknown")
    temp_c = current.get("temp_C", "?")
    feels_c = current.get("FeelsLikeC", "?")
    humidity = current.get("humidity", "?")
    wind_kmph = current.get("windspeedKmph", "?")
    return (
        f"Weather for {location}:\n"
        f"- Condition: {weather_desc}\n"
        f"- Temp: {temp_c}C (feels like {feels_c}C)\n"
        f"- Humidity: {humidity}%\n"
        f"- Wind: {wind_kmph} km/h"
    )


def _web_search(query: str, limit: int = 5) -> str:
    q = query.strip()
    if not q:
        return "Please provide a search query."

    url = (
        "https://api.duckduckgo.com/?"
        + urllib.parse.urlencode({"q": q, "format": "json", "no_redirect": "1", "no_html": "1", "skip_disambig": "1"})
    )
    try:
        data = json.loads(_http_get(url).decode("utf-8", errors="ignore"))
    except Exception as e:
        return f"Web search failed: {e}"

    lines: list[str] = []
    abstract = (data.get("AbstractText") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()
    if abstract:
        if abstract_url:
            lines.append(f"1. {abstract}\n   Source: {abstract_url}")
        else:
            lines.append(f"1. {abstract}")

    topics = data.get("RelatedTopics") or []
    count = len(lines)
    for topic in topics:
        if count >= limit:
            break
        if isinstance(topic, dict) and "Topics" in topic:
            for nested in topic.get("Topics", []):
                if count >= limit:
                    break
                text = (nested.get("Text") or "").strip()
                first_url = (nested.get("FirstURL") or "").strip()
                if text:
                    count += 1
                    lines.append(f"{count}. {text}\n   Source: {first_url}")
        else:
            text = (topic.get("Text") or "").strip() if isinstance(topic, dict) else ""
            first_url = (topic.get("FirstURL") or "").strip() if isinstance(topic, dict) else ""
            if text:
                count += 1
                lines.append(f"{count}. {text}\n   Source: {first_url}")

    if not lines:
        return f"No quick web results found for '{query}'."
    return "\n".join(lines)


def _fetch_webpage(url: str, max_chars: int = 2500) -> str:
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"

    try:
        html = _http_get(u).decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Failed to fetch URL: {e}"

    parser = _TextExtractor()
    parser.feed(html)
    text = " ".join(parser.parts)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return f"No readable text found at {u}"
    clipped = text[:max_chars]
    if len(text) > max_chars:
        clipped += "..."
    return f"Fetched: {u}\n\n{clipped}"


def create_builtin_tools() -> list[ToolDefinition]:
    """Create safe read-only builtins."""
    return [
        ToolDefinition(
            name="get_weather",
            description="Get current weather for a location. Read-only.",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City, state/country"}},
                "required": ["location"],
            },
            function=lambda location: _weather(location),
            permission="read",
        ),
        ToolDefinition(
            name="search_web",
            description="Search the web and return quick results with source links. Read-only.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Number of results (1-10)"},
                },
                "required": ["query"],
            },
            function=lambda query, limit=5: _web_search(query, max(1, min(int(limit), 10))),
            permission="read",
        ),
        ToolDefinition(
            name="fetch_webpage",
            description="Fetch and summarize readable text from a URL. Read-only.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s) URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Maximum returned characters"},
                },
                "required": ["url"],
            },
            function=lambda url, max_chars=2500: _fetch_webpage(url, max(500, min(int(max_chars), 8000))),
            permission="read",
        ),
    ]
