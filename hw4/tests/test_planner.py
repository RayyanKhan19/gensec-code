"""Unit tests for the Study Planner's pure planning functions."""

from pathlib import Path
from runpy import run_path

import pytest


PLANNER = run_path(str(Path(__file__).resolve().parents[1] / "mcp-server.py"))
rank_assignments = PLANNER["rank_assignments"]
allocate_study_time = PLANNER["allocate_study_time"]


@pytest.fixture
def assignments():
    """Return a small valid assignment set with a deadline tie."""
    return [
        {
            "title": "Research paper",
            "deadline": "2026-10-30",
            "importance": 3,
            "estimated_hours": 4,
        },
        {
            "title": "Calculus set",
            "deadline": "2026-10-25",
            "importance": 2,
            "estimated_hours": 3,
        },
        {
            "title": "Biology quiz prep",
            "deadline": "2026-10-25",
            "importance": 5,
            "estimated_hours": 2,
        },
    ]


def test_rank_assignments_uses_deadline_then_importance(assignments):
    """Earlier deadlines come first, with importance breaking a tie."""
    ranked = rank_assignments(assignments)
    assert [item["title"] for item in ranked] == [
        "Biology quiz prep",
        "Calculus set",
        "Research paper",
    ]
    assert [item["rank"] for item in ranked] == [1, 2, 3]


@pytest.mark.parametrize(
    "bad_hours",
    [-1, float("nan"), float("inf"), "two"],
)
def test_rank_assignments_rejects_invalid_estimates(assignments, bad_hours):
    """Assignment estimates must be finite, numeric, and non-negative."""
    assignments[0]["estimated_hours"] = bad_hours
    with pytest.raises(ValueError, match="estimated_hours"):
        rank_assignments(assignments)


def test_rank_assignments_rejects_bad_deadline_and_importance(assignments):
    """Malformed dates and out-of-range importance values are rejected."""
    assignments[0]["deadline"] = "next Friday"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        rank_assignments(assignments)
    assignments[0]["deadline"] = "2026-10-30"
    assignments[0]["importance"] = 6
    with pytest.raises(ValueError, match="importance"):
        rank_assignments(assignments)


def test_allocate_study_time_reports_unfinished_work(assignments):
    """A short budget leaves accurate, explicit unfinished work."""
    result = allocate_study_time(assignments, available_hours=4)
    assert result["used_hours"] == 4
    assert result["unallocated_hours"] == 0
    assert result["unfinished_work"] == [
        {"title": "Calculus set", "remaining_hours": 1.0},
        {"title": "Research paper", "remaining_hours": 4.0},
    ]
    assert result["assignments"][0]["status"] == "complete"


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), "four"])
def test_allocate_study_time_rejects_invalid_budgets(assignments, budget):
    """Available time must be finite, numeric, and non-negative."""
    with pytest.raises(ValueError, match="available_hours"):
        allocate_study_time(assignments, available_hours=budget)
