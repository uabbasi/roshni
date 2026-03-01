"""Google Workspace ToolDefinition factory (registration/schema layer).

Covers Calendar, Docs, Sheets, and Drive tools.  Gmail tools are in
``gmail_tool.py`` — they share the same OAuth credentials but have a
separate factory with IMAP/SMTP fallback logic.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from roshni.agent.tools import ToolDefinition


class GetCalendarEventsInput(BaseModel):
    days: int = Field(default=7, description="Number of days ahead to fetch events")
    calendar_id: str = Field(default="primary", description="Calendar ID (default: primary)")


class CreateCalendarEventInput(BaseModel):
    title: str = Field(description="Event title/summary")
    start_time: str = Field(description="Start time in ISO 8601 format (e.g., 2026-02-05T10:00:00)")
    end_time: str = Field(description="End time in ISO 8601 format (e.g., 2026-02-05T11:00:00)")
    description: str = Field(default="", description="Event description")
    location: str = Field(default="", description="Event location")
    attendees: str = Field(default="", description="Comma-separated email addresses of attendees")
    confirmed: bool = Field(
        default=False,
        description="Set to true ONLY after the user has explicitly confirmed. First call with false to preview.",
    )


class ReadGoogleDocInput(BaseModel):
    doc_id: str = Field(description="Google Doc ID (from URL: docs.google.com/document/d/{DOC_ID}/edit)")


class ReadGoogleSheetInput(BaseModel):
    spreadsheet_id: str = Field(
        description="Spreadsheet ID (from URL: docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit)",
    )
    range: str = Field(
        default="",
        description="A1 range notation (e.g. 'Sheet1!A1:Z100'). Leave empty to read the first sheet.",
    )


class SearchGoogleDriveInput(BaseModel):
    query: str = Field(description="Search query for file names and content")
    file_type: str = Field(default="", description="Filter by MIME type: 'doc', 'sheet', 'slide', 'pdf', or '' for all")
    max_results: int = Field(default=10, description="Maximum files to return")


def create_google_workspace_tools(functions: dict[str, Callable[..., str]]) -> list[ToolDefinition]:
    """Create Google Workspace tools from injected implementations.

    Required keys:
      get_calendar_events, create_calendar_event, read_google_doc,
      read_google_sheet, search_google_drive
    """
    f = functions
    return [
        ToolDefinition.from_function(
            func=f["get_calendar_events"],
            name="get_calendar_events",
            description=(
                "Get upcoming calendar events for the next N days. "
                "Returns event titles, times, locations, and attendees. "
                "Use for: 'what's on my calendar', 'any meetings today', 'schedule this week'"
            ),
            args_schema=GetCalendarEventsInput,
        ),
        ToolDefinition.from_function(
            func=f["create_calendar_event"],
            name="create_calendar_event",
            description=(
                "Create a new Google Calendar event. Requires title, start time, and end time. "
                "Times must be ISO 8601 format with timezone (e.g., 2026-02-05T10:00:00). "
                "REQUIRES USER CONFIRMATION before calling."
            ),
            args_schema=CreateCalendarEventInput,
            permission="write",
        ),
        ToolDefinition.from_function(
            func=f["read_google_doc"],
            name="read_google_doc",
            description=(
                "Read the text content of a Google Doc by its ID. "
                "Extract the doc ID from a Google Docs URL: docs.google.com/document/d/{DOC_ID}/edit"
            ),
            args_schema=ReadGoogleDocInput,
        ),
        ToolDefinition.from_function(
            func=f["read_google_sheet"],
            name="read_google_sheet",
            description=(
                "Read data from a Google Sheet by its spreadsheet ID. "
                "Extract the ID from a Sheets URL: docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit. "
                "Optionally specify an A1 range (e.g. 'Sheet1!A1:Z100'). "
                "Returns headers and rows as key-value pairs."
            ),
            args_schema=ReadGoogleSheetInput,
        ),
        ToolDefinition.from_function(
            func=f["search_google_drive"],
            name="search_google_drive",
            description=(
                "Search Google Drive for files by name. "
                "Optionally filter by type: 'doc', 'sheet', 'slide', 'pdf'. "
                "Returns file names, links, and modification dates."
            ),
            args_schema=SearchGoogleDriveInput,
        ),
    ]


__all__ = ["create_google_workspace_tools"]
