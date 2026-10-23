"""Study-planning MCP server for Rayyan Khan (khanr2024)."""

from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Any

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("study-planner")


def _validate_assignments(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize assignment records used by the planner."""
    if not isinstance(assignments, list):
        raise ValueError("assignments must be a list")
    normalized = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            raise ValueError(f"assignment {index + 1} must be an object")
        title = assignment.get("title")
        deadline = assignment.get("deadline")
        importance = assignment.get("importance")
        estimated_hours = assignment.get("estimated_hours")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"assignment {index + 1} needs a non-empty title")
        if not isinstance(deadline, str):
            raise ValueError(f"{title}: deadline must use YYYY-MM-DD format")
        try:
            date.fromisoformat(deadline)
        except ValueError as exc:
            raise ValueError(f"{title}: deadline must use YYYY-MM-DD format") from exc
        if isinstance(importance, bool) or not isinstance(importance, int) or not 1 <= importance <= 5:
            raise ValueError(f"{title}: importance must be an integer from 1 to 5")
        if isinstance(estimated_hours, bool) or not isinstance(estimated_hours, (int, float)):
            raise ValueError(f"{title}: estimated_hours must be a number")
        if not isfinite(estimated_hours) or estimated_hours < 0:
            raise ValueError(f"{title}: estimated_hours must be finite and non-negative")
        normalized.append(
            {
                "title": title.strip(),
                "deadline": deadline,
                "importance": importance,
                "estimated_hours": float(estimated_hours),
            }
        )
    return normalized


def _validate_budget(available_hours: float) -> float:
    """Return a finite, non-negative study-time budget."""
    if isinstance(available_hours, bool) or not isinstance(available_hours, (int, float)):
        raise ValueError("available_hours must be a number")
    if not isfinite(available_hours) or available_hours < 0:
        raise ValueError("available_hours must be finite and non-negative")
    return float(available_hours)


def _priority_key(assignment: dict[str, Any]) -> tuple[str, int, str]:
    """Sort by earliest deadline, then higher importance, then title."""
    return (assignment["deadline"], -assignment["importance"], assignment["title"].casefold())


@mcp.tool()
def rank_assignments(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank assignments by earliest deadline, then importance from 1 to 5.

    Each assignment needs ``title``, ``deadline`` (YYYY-MM-DD), ``importance``
    (1-5), and ``estimated_hours`` (a non-negative number).
    """
    checked = _validate_assignments(assignments)
    ranked = sorted(checked, key=_priority_key)
    return [{**item, "rank": index} for index, item in enumerate(ranked, start=1)]


@mcp.tool()
def allocate_study_time(
    assignments: list[dict[str, Any]], available_hours: float
) -> dict[str, Any]:
    """Allocate a limited study budget in assignment priority order.

    The result lists hours allocated to each assignment and explicitly marks
    unfinished work when the available budget is insufficient.
    """
    checked = sorted(_validate_assignments(assignments), key=_priority_key)
    remaining = _validate_budget(available_hours)
    original_budget = remaining
    allocations = []
    for item in checked:
        allocated = min(item["estimated_hours"], remaining)
        remaining = max(0.0, remaining - allocated)
        allocations.append(
            {
                **item,
                "allocated_hours": round(allocated, 2),
                "remaining_hours": round(item["estimated_hours"] - allocated, 2),
                "status": "complete" if allocated >= item["estimated_hours"] else "unfinished",
            }
        )
    return {
        "available_hours": round(original_budget, 2),
        "used_hours": round(original_budget - remaining, 2),
        "unallocated_hours": round(remaining, 2),
        "assignments": allocations,
        "unfinished_work": [
            {"title": item["title"], "remaining_hours": item["remaining_hours"]}
            for item in allocations
            if item["status"] == "unfinished"
        ],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
