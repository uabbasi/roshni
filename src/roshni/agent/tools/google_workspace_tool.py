"""Google Workspace ToolDefinition factory (registration/schema layer).

Incremental split: app layers can inject concrete implementations while reusing
the canonical tool schemas and descriptions here.
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


class SearchGmailInput(BaseModel):
    query: str = Field(description="Gmail search query (same syntax as Gmail search box)")
    max_results: int = Field(default=10, description="Maximum number of messages to return")


class CreateGmailDraftInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body (plain text)")


class CreateGmailReplyDraftInput(BaseModel):
    message_id: str = Field(description="Gmail message ID to reply to")
    body: str = Field(description="Reply body (plain text)")
    subject_override: str = Field(
        default="",
        description="Optional subject override. Leave empty to auto-prefix original subject with Re: if needed.",
    )


class GetGmailSummaryInput(BaseModel):
    pass


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
      get_calendar_events, create_calendar_event, search_gmail, create_gmail_draft,
      create_gmail_reply_draft, get_gmail_summary, read_google_doc,
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
            func=f["search_gmail"],
            name="search_gmail",
            description=(
                "Search Gmail messages using Gmail search syntax. "
                "Examples: 'from:boss@company.com', 'subject:invoice', 'is:unread after:2026/01/01'. "
                "Returns subject, sender, date, and snippet for each message."
            ),
            args_schema=SearchGmailInput,
        ),
        ToolDefinition.from_function(
            func=f["create_gmail_draft"],
            name="create_gmail_draft",
            description=(
                "Create a NEW Gmail draft (starts a new thread). "
                "Use this when composing a brand-new email. "
                "For replies to an existing message, use create_gmail_reply_draft instead."
            ),
            args_schema=CreateGmailDraftInput,
            permission="send",
            requires_approval=False,
        ),
        ToolDefinition.from_function(
            func=f["create_gmail_reply_draft"],
            name="create_gmail_reply_draft",
            description=(
                "Create a Gmail REPLY draft in an existing thread using a specific message_id. "
                "Use this when the user asks to reply/follow up on an existing email."
            ),
            args_schema=CreateGmailReplyDraftInput,
            permission="send",
            requires_approval=False,
        ),
        ToolDefinition.from_function(
            func=f["get_gmail_summary"],
            name="get_gmail_summary",
            description=(
                "Get Gmail inbox summary with two buckets: "
                "starred_unread (always surface to user) and important_unread (triage for urgency — "
                "only surface if time-sensitive, action-required, or financial/medical/legal). "
                "Use for: 'any new email', 'inbox summary', 'urgent messages'"
            ),
            args_schema=GetGmailSummaryInput,
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
