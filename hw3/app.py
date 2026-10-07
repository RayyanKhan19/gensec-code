"""Homework 3: Study Planner Agent for khanr2024.

The planner stores assignments locally and uses deterministic Python code to
build schedules. The language model selects and explains the available tools.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator


APP_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_FILE = Path(
    os.environ.get("STUDY_PLANNER_DATA", APP_DIR / "assignments.json")
)
TERMINAL_TIMEOUT_SECONDS = 10
TERMINAL_OUTPUT_LIMIT = 4_000


class AddAssignmentInput(BaseModel):
    """Validated input for adding one assignment to the local planner."""

    title: str = Field(min_length=1, max_length=200, description="Assignment title")
    due_date: str = Field(description="Deadline in YYYY-MM-DD format")
    estimated_hours: float = Field(
        gt=0, allow_inf_nan=False, description="Positive finite study hours required"
    )

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        """Reject titles that contain only whitespace."""
        if not value.strip():
            raise ValueError("title must contain at least one non-space character")
        return value.strip()

    @field_validator("due_date")
    @classmethod
    def due_date_must_be_iso_date(cls, value: str) -> str:
        """Require a real calendar date written as YYYY-MM-DD."""
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("due_date must be a valid YYYY-MM-DD date") from exc
        if parsed.isoformat() != value:
            raise ValueError("due_date must use YYYY-MM-DD format")
        return value


class ListAssignmentsInput(BaseModel):
    """The assignment list tool has no arguments."""


class StudyPlanInput(BaseModel):
    """Validated inputs for deterministic study-time allocation."""

    daily_hours_budget: float = Field(
        gt=0, allow_inf_nan=False, description="Available study hours per day"
    )
    start_date: str | None = Field(
        default=None,
        description="First planning day in YYYY-MM-DD format; defaults to today",
    )

    @field_validator("start_date")
    @classmethod
    def start_date_must_be_iso_date(cls, value: str | None) -> str | None:
        """Validate an optional start date using the same strict date format."""
        if value is None:
            return value
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("start_date must be a valid YYYY-MM-DD date") from exc
        if parsed.isoformat() != value:
            raise ValueError("start_date must use YYYY-MM-DD format")
        return value


class TerminalInput(BaseModel):
    """Input for the human-approved terminal helper."""

    command: str = Field(min_length=1, max_length=2_000, description="Shell command")

    @field_validator("command")
    @classmethod
    def command_must_not_be_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only commands."""
        if not value.strip():
            raise ValueError("command must not be blank")
        return value.strip()


def _load_assignments() -> list[dict[str, Any]]:
    """Read assignments from JSON, returning an empty list before first use."""
    try:
        with ASSIGNMENTS_FILE.open("r", encoding="utf-8") as assignments_file:
            data = json.load(assignments_file)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Could not read {ASSIGNMENTS_FILE.name}: {exc}") from exc
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise RuntimeError(f"{ASSIGNMENTS_FILE.name} must contain a JSON list of assignments")
    return data


def _save_assignments(assignments: list[dict[str, Any]]) -> None:
    """Atomically save assignments to the ignored local JSON file."""
    ASSIGNMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = ASSIGNMENTS_FILE.with_suffix(ASSIGNMENTS_FILE.suffix + ".tmp")
    try:
        temporary_file.write_text(
            json.dumps(assignments, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_file.replace(ASSIGNMENTS_FILE)
    except OSError as exc:
        temporary_file.unlink(missing_ok=True)
        raise RuntimeError(f"Could not save {ASSIGNMENTS_FILE.name}: {exc}") from exc


def _sort_assignments(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by deadline, then by saved numeric ID for stable ties."""
    return sorted(assignments, key=lambda item: (item["due_date"], item["id"]))


@tool(args_schema=AddAssignmentInput)
def add_assignment(title: str, due_date: str, estimated_hours: float) -> str:
    """Save an assignment with its title, due date, and estimated study hours."""
    assignments = _load_assignments()
    assignment = {
        "id": max((item.get("id", 0) for item in assignments), default=0) + 1,
        "title": title,
        "due_date": due_date,
        "estimated_hours": estimated_hours,
    }
    assignments.append(assignment)
    _save_assignments(assignments)
    return (
        f"Saved assignment #{assignment['id']}: {title} — due {due_date}, "
        f"{estimated_hours:g} study hours."
    )


@tool(args_schema=ListAssignmentsInput)
def list_assignments() -> str:
    """Show saved assignments ordered by due date, with stable tie ordering."""
    assignments = _sort_assignments(_load_assignments())
    if not assignments:
        return "No assignments saved yet."
    return "\n".join(
        f"{item['due_date']} | {item['title']} | {item['estimated_hours']:g} hours"
        for item in assignments
    )


def _build_study_plan(
    assignments: list[dict[str, Any]], daily_hours_budget: float, start: date
) -> str:
    """Allocate hours earliest-deadline-first across inclusive calendar days.

    Planning starts on ``start``. A deadline date is usable for study, so an
    assignment due on the start date has one available day. Overdue assignments
    receive no time and all remaining hours are reported as unallocated.
    """
    if not math.isfinite(daily_hours_budget) or daily_hours_budget <= 0:
        raise ValueError("daily_hours_budget must be positive and finite")
    if not assignments:
        return "No assignments saved yet, so there is no study plan to create."

    ordered = _sort_assignments(assignments)
    remaining = {item["id"]: float(item["estimated_hours"]) for item in ordered}
    final_deadline = max(
        (date.fromisoformat(item["due_date"]) for item in ordered if item["due_date"] >= start.isoformat()),
        default=start - timedelta(days=1),
    )
    lines: list[str] = [
        f"Study plan from {start.isoformat()} (deadlines included; "
        f"up to {daily_hours_budget:g} hours per day):"
    ]

    day = start
    while day <= final_deadline:
        capacity = daily_hours_budget
        day_work: list[str] = []
        for assignment in ordered:
            if assignment["due_date"] < day.isoformat():
                continue
            hours_left = remaining[assignment["id"]]
            if hours_left <= 0 or capacity <= 0:
                continue
            allocated = min(hours_left, capacity)
            remaining[assignment["id"]] = hours_left - allocated
            capacity -= allocated
            day_work.append(f"{allocated:g}h {assignment['title']}")
        if day_work:
            lines.append(f"{day.isoformat()}: " + "; ".join(day_work))
        else:
            lines.append(f"{day.isoformat()}: no scheduled study")
        day += timedelta(days=1)

    unallocated = [
        f"{item['title']}: {remaining[item['id']]:g}h cannot fit before "
        f"{item['due_date']} (including that date)"
        for item in ordered
        if remaining[item["id"]] > 1e-9
    ]
    if unallocated:
        lines.append("Work that cannot fit:")
        lines.extend(f"- {item}" for item in unallocated)
    else:
        lines.append("All estimated work fits before its deadlines.")
    return "\n".join(lines)


@tool(args_schema=StudyPlanInput)
def create_study_plan(daily_hours_budget: float, start_date: str | None = None) -> str:
    """Create a deterministic earliest-deadline-first plan from saved assignments.

    The planning start day defaults to today. Both the start date and each due
    date are inclusive; hours that cannot fit by their deadline are reported.
    """
    start = date.fromisoformat(start_date) if start_date else date.today()
    return _build_study_plan(_load_assignments(), daily_hours_budget, start)


@tool(args_schema=TerminalInput)
def terminal(command: str) -> str:
    """Show a proposed shell command and run it only after exact human approval.

    Approval is a confirmation prompt, not a security sandbox. Only run commands
    you understand and trust. Execution has a ten-second timeout and capped output.
    """
    print(f"Proposed terminal command: {command}")
    print(
        "Approval is not a security sandbox; the command runs with this user's "
        "normal permissions."
    )
    try:
        approval = input('Type YES to execute this command (anything else denies): ')
    except (EOFError, KeyboardInterrupt):
        return "Command denied; no approval was received."
    if approval.strip() != "YES":
        return "Command denied; it was not executed."

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=TERMINAL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Command stopped after {TERMINAL_TIMEOUT_SECONDS} seconds."
    except OSError as exc:
        return f"Could not run command: {exc}"

    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    if not output:
        output = "(no output)"
    if len(output) > TERMINAL_OUTPUT_LIMIT:
        output = output[:TERMINAL_OUTPUT_LIMIT] + "\n… output truncated …"
    return f"Exit code: {result.returncode}\n{output}"


def build_agent() -> Any:
    """Load local environment settings and build the course-style Gemini agent."""
    load_dotenv(APP_DIR / ".env", override=False)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key, "
            "or export GOOGLE_API_KEY in the shell."
        )
    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    system_prompt = (
        "You are the Study Planner Agent for khanr2024. Use the assignment tools "
        "to save and list coursework and create schedules. Do not invent saved data. "
        "For terminal requests, use the terminal tool; the person must approve each "
        "command by typing YES. Explain schedule limitations clearly."
    )
    tools = [add_assignment, list_assignments, create_study_plan, terminal]
    return create_agent(model=model, tools=tools, system_prompt=system_prompt)


def _display_stream(agent: Any, prompt: str) -> None:
    """Stream an agent response while displaying tool calls and results."""
    for step in agent.stream({"messages": [{"role": "user", "content": prompt}]}):
        for node_name in ("model", "tools"):
            node = step.get(node_name)
            if not node:
                continue
            for message in node.get("messages", []):
                if node_name == "model":
                    for call in getattr(message, "tool_calls", []):
                        print(f"\n[Tool call] {call['name']}({call['args']})")
                    if message.content:
                        print(message.content)
                else:
                    print(f"[Tool result] {message.content}")


def main() -> None:
    """Run the interactive command-line agent."""
    print("Welcome to Homework 3: Study Planner Agent — khanr2024")
    print("Type a request, or press Enter on a blank line to exit.")
    try:
        agent = build_agent()
    except (RuntimeError, ValueError) as exc:
        print(f"Setup error: {exc}")
        return

    while True:
        try:
            prompt = input("study planner>> ").strip()
            if not prompt:
                return
            _display_stream(agent, prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        except Exception as exc:  # Keep transient model and tool errors user-friendly.
            print(f"Sorry, that request ran into a problem: {exc}")


if __name__ == "__main__":
    main()
