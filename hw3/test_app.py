"""Model-free checks for Homework 3 storage, validation, planning, and approval."""

import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app


class StudyPlannerChecks(unittest.TestCase):
    """Exercise the application tools without constructing a model agent."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_file = Path(self.temp_dir.name) / "assignments.json"
        self.file_patch = patch.object(app, "ASSIGNMENTS_FILE", self.data_file)
        self.file_patch.start()
        self.addCleanup(self.file_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def test_assignment_storage_and_due_date_order(self) -> None:
        """Assignments persist locally and list in deadline order."""
        app.add_assignment.invoke(
            {"title": "Essay", "due_date": "2026-11-06", "estimated_hours": 3}
        )
        app.add_assignment.invoke(
            {"title": "Quiz", "due_date": "2026-11-04", "estimated_hours": 1.5}
        )

        output = app.list_assignments.invoke({})

        self.assertLess(output.index("Quiz"), output.index("Essay"))
        self.assertTrue(self.data_file.exists())

    def test_invalid_dates_and_nonpositive_or_nonfinite_hours(self) -> None:
        """Tool schemas reject malformed dates and invalid hour values."""
        base = {"title": "Reading", "due_date": "2026-02-30", "estimated_hours": 1}
        with self.assertRaises(Exception):
            app.add_assignment.invoke(base)
        for hours in (0, -1, float("inf"), float("nan")):
            with self.subTest(hours=hours), self.assertRaises(Exception):
                app.add_assignment.invoke(
                    {"title": "Reading", "due_date": "2026-11-04", "estimated_hours": hours}
                )
        with self.assertRaises(Exception):
            app.create_study_plan.invoke({"daily_hours_budget": float("inf")})

    def test_plan_uses_inclusive_dates_and_reports_unfitted_hours(self) -> None:
        """The schedule is earliest-deadline-first and reports its shortfall."""
        assignments = [
            {"id": 1, "title": "Essay", "due_date": "2026-06-02", "estimated_hours": 3.0},
            {"id": 2, "title": "Quiz", "due_date": "2026-06-01", "estimated_hours": 2.0},
        ]

        output = app._build_study_plan(assignments, 2.0, date(2026, 6, 1))

        self.assertIn("2026-06-01: 2h Quiz", output)
        self.assertIn("2026-06-02: 2h Essay", output)
        self.assertIn("Essay: 1h cannot fit before 2026-06-02", output)

    def test_empty_plan_is_explained(self) -> None:
        """An empty saved list gets a useful message instead of a blank plan."""
        result = app.create_study_plan.invoke({"daily_hours_budget": 2})
        self.assertIn("No assignments saved", result)

    def test_denied_terminal_command_never_runs(self) -> None:
        """Anything other than exact YES denies execution before subprocess."""
        with patch("builtins.input", return_value="yes"), patch.object(
            app.subprocess, "run"
        ) as run_command:
            result = app.terminal.invoke({"command": "echo should-not-run"})

        run_command.assert_not_called()
        self.assertIn("denied", result.lower())

    def test_approved_terminal_command_has_timeout_and_capped_output(self) -> None:
        """Approved execution is bounded and returned output is truncated."""
        completed = SimpleNamespace(returncode=0, stdout="x" * 5_000, stderr="")
        with patch("builtins.input", return_value="YES"), patch.object(
            app.subprocess, "run", return_value=completed
        ) as run_command:
            result = app.terminal.invoke({"command": "echo accepted"})

        run_command.assert_called_once()
        self.assertEqual(run_command.call_args.kwargs["timeout"], app.TERMINAL_TIMEOUT_SECONDS)
        self.assertLessEqual(len(result), app.TERMINAL_OUTPUT_LIMIT + 100)
        self.assertIn("output truncated", result)

    def test_approved_terminal_timeout_is_reported(self) -> None:
        """A command that exceeds the configured timeout returns a clear result."""
        with patch("builtins.input", return_value="YES"), patch.object(
            app.subprocess, "run", side_effect=app.subprocess.TimeoutExpired("sleep", 10)
        ):
            result = app.terminal.invoke({"command": "sleep 100"})

        self.assertIn("stopped after 10 seconds", result)


if __name__ == "__main__":
    unittest.main()
