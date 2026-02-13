"""Built-in utility tools: weather + web search/fetch."""

from __future__ import annotations

import datetime
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from ddgs import DDGS

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


def _web_search(query: str, limit: int = 5, auto_fetch: bool = True) -> str:
    q = query.strip()
    if not q:
        return "Please provide a search query."

    # Auto-append current year if query doesn't already contain one
    if not re.search(r"\b20\d{2}\b", q):
        q = f"{q} {datetime.datetime.now(datetime.UTC).year}"

    try:
        results = DDGS().text(q, max_results=limit)
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return f"No web results found for '{query}'."

    lines: list[str] = []
    first_url: str | None = None
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        if i == 1:
            first_url = href
        line = f"{i}. {title}"
        if body:
            line += f"\n   {body}"
        if href:
            line += f"\n   Source: {href}"
        lines.append(line)

    output = "\n".join(lines)

    # Auto-fetch first result page for richer context
    if auto_fetch and first_url:
        try:
            fetched = _fetch_webpage(first_url)
            output += f"\n\n---\n{fetched}"
        except Exception:
            pass  # silently skip on fetch failure

    return output


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
            description=(
                "Search the web and return results with source links."
                " Automatically fetches first result for richer context. Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Number of results (1-10)"},
                    "auto_fetch": {
                        "type": "boolean",
                        "description": "Fetch and append first result page content (default true)",
                    },
                },
                "required": ["query"],
            },
            function=lambda query, limit=5, auto_fetch=True: _web_search(
                query, max(1, min(int(limit), 10)), auto_fetch=auto_fetch
            ),
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
