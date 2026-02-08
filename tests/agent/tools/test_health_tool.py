"""Tests for health summary tool."""

from __future__ import annotations

from pathlib import Path

from roshni.agent.tools.health_tool import create_health_tools


def _write_export(path: Path) -> None:
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<HealthData>
  <Record
    type=\"HKQuantityTypeIdentifierStepCount\"
    unit=\"count\"
    value=\"1200\"
    startDate=\"2026-01-02 09:00:00 -0800\"
    endDate=\"2026-01-02 09:05:00 -0800\"
  />
</HealthData>
"""
    path.write_text(xml, encoding="utf-8")


def test_get_health_summary_tool(tmp_dir):
    export_path = Path(tmp_dir) / "export.xml"
    _write_export(export_path)

    tools = create_health_tools(str(export_path))
    assert len(tools) == 1
    tool = tools[0]
    out = tool.execute({"start_date": "2026-01-02", "end_date": "2026-01-02"})

    assert "Health summary" in out
    assert "Total steps: 1200" in out


def test_get_health_summary_missing_file(tmp_dir):
    tools = create_health_tools(str(Path(tmp_dir) / "missing.xml"))
    out = tools[0].execute({"start_date": "2026-01-02"})
    assert "not found" in out.lower() or "invalid" in out.lower()
