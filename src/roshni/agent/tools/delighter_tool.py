"""Delighter tools for daily assistant workflows."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from roshni.agent.tools import ToolDefinition


def _load_reminders(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_reminders(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _next_id(items: list[dict]) -> int:
    highest = 0
    for item in items:
        try:
            highest = max(highest, int(item.get("id", 0)))
        except Exception:
            continue
    return highest + 1


def _get_task_summary(tasks_dir: str) -> str:
    """Read task status from TaskStore if available. Returns summary lines or empty string."""
    if not tasks_dir:
        return ""
    try:
        from roshni.agent.task_store import TaskStore

        store = TaskStore(tasks_dir)
        now = datetime.now()
        today_date = date.today()
        open_tasks = store.list_tasks(status="open")
        in_progress = store.list_tasks(status="in_progress")

        overdue = [t for t in open_tasks + in_progress if t.due and t.due < now]
        due_today = [t for t in open_tasks + in_progress if t.due and t.due.date() == today_date]

        lines: list[str] = []
        if overdue:
            lines.append(f"- Overdue tasks ({len(overdue)}):")
            for t in overdue[:5]:
                lines.append(f"  - {t.title} (due: {t.due.strftime('%Y-%m-%d') if t.due else '?'})")
        if due_today:
            lines.append(f"- Due today ({len(due_today)}):")
            for t in due_today[:5]:
                lines.append(f"  - {t.title}")
        if in_progress:
            lines.append(f"- In progress ({len(in_progress)}):")
            for t in in_progress[:3]:
                lines.append(f"  - {t.title}")
        if not lines and open_tasks:
            lines.append(f"- Open tasks: {len(open_tasks)}")
        return "\n".join(lines)
    except Exception:
        return ""


def _morning_brief(name: str = "", location: str = "", top_focus: str = "", tasks_dir: str = "") -> str:
    today = datetime.now().strftime("%A, %B %d")
    user = name or "there"
    focus = top_focus or "your most important task"
    if location:
        weather_hint = f"\n- Weather check: run get_weather for {location}"
    else:
        weather_hint = "\n- Weather check: add your location"

    task_section = _get_task_summary(tasks_dir)
    if task_section:
        task_section = f"\n\nTask Status:\n{task_section}"

    return (
        f"Good morning, {user}. Here is your brief for {today}.\n"
        f"- Top focus: {focus}\n"
        "- Calendar: review today's meetings and travel time\n"
        "- Inbox: triage urgent items and draft responses\n"
        "- Personal: hydration, movement, and one short break plan"
        f"{weather_hint}\n"
        "- End-of-day: capture wins and carry-overs"
        f"{task_section}"
    )


def _daily_plan(priorities: str, meetings: str = "", constraints: str = "") -> str:
    now = datetime.now().strftime("%Y-%m-%d")
    meetings_block = meetings.strip() or "(none provided)"
    constraints_block = constraints.strip() or "(none provided)"
    return (
        f"Daily Plan - {now}\n\n"
        "Top Priorities:\n"
        f"{priorities}\n\n"
        "Meetings:\n"
        f"{meetings_block}\n\n"
        "Constraints:\n"
        f"{constraints_block}\n\n"
        "Suggested blocks:\n"
        "1. Deep work block (90 min)\n"
        "2. Communications block (30 min)\n"
        "3. Admin + follow-ups (30 min)\n"
        "4. End-of-day review (15 min)"
    )


def _weekly_review(wins: str = "", challenges: str = "", next_week_focus: str = "", tasks_dir: str = "") -> str:
    week = datetime.now().strftime("Week of %Y-%m-%d")

    completed_section = ""
    if tasks_dir:
        try:
            from roshni.agent.task_store import TaskStore

            store = TaskStore(tasks_dir)
            done = store.list_tasks(status="done", limit=50)
            if done:
                completed_section = "\n\nCompleted Tasks:\n"
                for t in done[:20]:
                    completed_section += f"- {t.title}"
                    if t.project:
                        completed_section += f" ({t.project})"
                    completed_section += "\n"
            # Trigger memory decay — archive old completed tasks
            archive_result = store.summarize_completed(older_than_days=30)
            if "Archived" in archive_result:
                completed_section += f"\n{archive_result}\n"
        except Exception:
            pass

    return (
        f"Weekly Review - {week}\n\n"
        "Wins:\n"
        f"{wins or '- Add your wins'}\n\n"
        "Challenges:\n"
        f"{challenges or '- Add your challenges'}\n\n"
        "Improvements:\n"
        "- What should be repeated?\n"
        "- What should be removed?\n"
        "- What needs support?\n\n"
        "Next Week Focus:\n"
        f"{next_week_focus or '- Define top 3 outcomes'}"
        f"{completed_section}"
    )


def _inbox_triage_bundle(items: str, tone: str = "professional") -> str:
    return (
        "Inbox Triage Bundle\n\n"
        "Input emails:\n"
        f"{items}\n\n"
        "Triage framework:\n"
        "1. Reply now (<2 min)\n"
        "2. Draft response (needs thought)\n"
        "3. Delegate / forward\n"
        "4. Archive / no response\n\n"
        f"Suggested reply tone: {tone}\n"
        "For each item, produce: action, priority, and one draft reply."
    )


def _save_reminder(path: Path, text: str, due: str = "", category: str = "general") -> str:
    items = _load_reminders(path)
    item_id = _next_id(items)
    items.append(
        {
            "id": item_id,
            "text": text.strip(),
            "due": due.strip(),
            "category": category.strip() or "general",
            "status": "open",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save_reminders(path, items)
    return f"Reminder saved (id={item_id}): {text}"


def _list_reminders(path: Path, status: str = "open") -> str:
    items = _load_reminders(path)
    status = (status or "open").strip().lower()
    filtered = [x for x in items if x.get("status", "open").lower() == status]
    if not filtered:
        return f"No {status} reminders."

    lines = [f"{status.title()} reminders:"]
    for item in filtered[:20]:
        due = item.get("due") or "no due date"
        lines.append(f"- [{item.get('id')}] {item.get('text')} (due: {due}, category: {item.get('category')})")
    return "\n".join(lines)


def _complete_reminder(path: Path, reminder_id: int) -> str:
    items = _load_reminders(path)
    for item in items:
        if int(item.get("id", -1)) == int(reminder_id):
            item["status"] = "done"
            item["completed_at"] = datetime.now().isoformat(timespec="seconds")
            _save_reminders(path, items)
            return f"Marked reminder {reminder_id} as done."
    return f"Reminder {reminder_id} not found."


def create_delighter_tools(reminders_path: str, tasks_dir: str = "") -> list[ToolDefinition]:
    """Create delight-oriented planning/reminder tools."""
    path = Path(reminders_path).expanduser()
    return [
        ToolDefinition(
            name="morning_brief",
            description="Generate a concise morning brief with focus prompts, routines, and task status.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User name"},
                    "location": {"type": "string", "description": "User location for weather follow-up"},
                    "top_focus": {"type": "string", "description": "Most important focus today"},
                },
                "required": [],
            },
            function=lambda name="", location="", top_focus="": _morning_brief(
                name=name,
                location=location,
                top_focus=top_focus,
                tasks_dir=tasks_dir,
            ),
            permission="read",
        ),
        ToolDefinition(
            name="daily_plan",
            description="Build a structured daily execution plan from priorities and meetings.",
            parameters={
                "type": "object",
                "properties": {
                    "priorities": {"type": "string", "description": "Top priorities for today"},
                    "meetings": {"type": "string", "description": "Meetings and fixed events"},
                    "constraints": {"type": "string", "description": "Energy/time constraints"},
                },
                "required": ["priorities"],
            },
            function=lambda priorities, meetings="", constraints="": _daily_plan(priorities, meetings, constraints),
            permission="read",
        ),
        ToolDefinition(
            name="weekly_review",
            description="Generate a weekly reflection and next-week planning template.",
            parameters={
                "type": "object",
                "properties": {
                    "wins": {"type": "string", "description": "Wins from this week"},
                    "challenges": {"type": "string", "description": "Challenges from this week"},
                    "next_week_focus": {"type": "string", "description": "Top outcomes for next week"},
                },
                "required": [],
            },
            function=lambda wins="", challenges="", next_week_focus="": _weekly_review(
                wins,
                challenges,
                next_week_focus,
                tasks_dir=tasks_dir,
            ),
            permission="read",
        ),
        ToolDefinition(
            name="inbox_triage_bundle",
            description="Create a triage and draft-response bundle for a set of emails.",
            parameters={
                "type": "object",
                "properties": {
                    "items": {"type": "string", "description": "List or paste of email summaries"},
                    "tone": {"type": "string", "description": "Draft tone"},
                },
                "required": ["items"],
            },
            function=lambda items, tone="professional": _inbox_triage_bundle(items, tone),
            permission="read",
        ),
        ToolDefinition(
            name="save_reminder",
            description="Save a reminder for proactive follow-up.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Reminder text"},
                    "due": {"type": "string", "description": "Optional due date/time"},
                    "category": {"type": "string", "description": "Optional category"},
                },
                "required": ["text"],
            },
            function=lambda text, due="", category="general": _save_reminder(path, text, due, category),
            permission="write",
        ),
        ToolDefinition(
            name="list_reminders",
            description="List reminders by status (open/done).",
            parameters={
                "type": "object",
                "properties": {"status": {"type": "string", "description": "Reminder status to list"}},
                "required": [],
            },
            function=lambda status="open": _list_reminders(path, status),
            permission="read",
        ),
        ToolDefinition(
            name="complete_reminder",
            description="Mark a reminder as done.",
            parameters={
                "type": "object",
                "properties": {"reminder_id": {"type": "integer", "description": "Reminder id"}},
                "required": ["reminder_id"],
            },
            function=lambda reminder_id: _complete_reminder(path, reminder_id),
            permission="write",
        ),
    ]
