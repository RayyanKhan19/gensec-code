"""Regression tests for MCP tool event tracking in the Study Planner agent."""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import ToolEventPrinter


def test_tool_end_tracks_name_from_start_when_name_keyword_is_absent(capsys):
    """A successful end callback records the name associated with its run ID."""
    events = ToolEventPrinter()
    run_id = uuid4()

    events.on_tool_start(
        {"name": "read_text_file"},
        '{"path": "assignments.json"}',
        run_id=run_id,
    )
    events.on_tool_end("assignments loaded", run_id=run_id)

    assert events.completed_tools == ["read_text_file"]
    assert run_id not in events._tool_names_by_run_id
    assert "[result] assignments loaded" in capsys.readouterr().out


def test_tool_error_does_not_mark_run_complete(capsys):
    """An errored tool run is removed from tracking and stays incomplete."""
    events = ToolEventPrinter()
    run_id = uuid4()
    events.on_tool_start({"name": "write_file"}, "{}", run_id=run_id)

    events.on_tool_error(RuntimeError("write failed"), run_id=run_id)
    events.on_tool_end("late result", run_id=run_id)

    assert events.completed_tools == []
    assert run_id not in events._tool_names_by_run_id
    assert "[tool error] RuntimeError: write failed" in capsys.readouterr().out
