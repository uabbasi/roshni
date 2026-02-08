"""Tests for delighter tools."""

from __future__ import annotations

import json
import os

from roshni.agent.tools.delighter_tool import create_delighter_tools


def test_reminder_write_and_read(tmp_dir):
    reminders_path = os.path.join(tmp_dir, "reminders.json")
    tools = {t.name: t for t in create_delighter_tools(reminders_path)}

    save_result = tools["save_reminder"].execute({"text": "Pay rent", "due": "2026-02-10"})
    assert "Reminder saved" in save_result

    listed = tools["list_reminders"].execute({"status": "open"})
    assert "Pay rent" in listed

    done = tools["complete_reminder"].execute({"reminder_id": 1})
    assert "Marked reminder 1 as done" in done

    data = json.loads(open(reminders_path).read())
    assert data[0]["status"] == "done"


def test_planning_tools_exist(tmp_dir):
    reminders_path = os.path.join(tmp_dir, "reminders.json")
    names = {t.name for t in create_delighter_tools(reminders_path)}
    assert "morning_brief" in names
    assert "daily_plan" in names
    assert "weekly_review" in names
    assert "inbox_triage_bundle" in names
