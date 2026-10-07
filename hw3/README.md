# Homework 3: Study Planner Agent

Interactive command-line study planner for `khanr2024`. The Gemini agent can
save assignments, list them by deadline, create a deterministic study plan,
and propose terminal commands that require explicit approval.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from this directory:

```sh
cd hw3
uv sync
cp .env.example .env
```

Put your Google AI Studio key in `.env` as `GOOGLE_API_KEY`. The optional
`GOOGLE_MODEL` setting defaults to `gemini-2.5-flash`. The key is read from the
environment and is never printed. You can export these variables in your shell
instead of using `.env`.

Start the application with:

```sh
uv run app.py
```

The first tool call creates `assignments.json` in this directory. That local
file is ignored by Git. To use another data file, set `STUDY_PLANNER_DATA` to
its path before starting the program.

## Example prompts

- “Add my biology lab report, due 2026-11-04, estimated at 3.5 hours.”
- “What assignments do I have? List them by deadline.”
- “Make a plan starting 2026-10-10 with a daily budget of 2 hours.”
- “Use the terminal to show the current directory.” The command is displayed
  and runs only if you type `YES` at the approval prompt.

## Scheduling rules

- If no start date is supplied, planning starts today (the computer's local
  date). An explicit start date must use `YYYY-MM-DD`.
- The start day and each deadline day are both available for study.
- Each day has at most the requested budget. Work is allocated by earliest
  deadline first; ties follow assignment creation order.
- Assignments already overdue at the start receive no scheduled hours. Any
  remaining hours that cannot fit by a deadline are listed as unallocated.
- The schedule is calculated with Python; the language model does not calculate
  the hours.

## Limitations and safety

Dates are calendar dates; the planner does not account for class times, holidays,
individual availability, or partial-day schedules. Estimates and the daily
budget are user-provided. Assignment data is a local JSON file without
multi-user access or synchronization.

The terminal tool asks for exact `YES` approval, enforces a ten-second timeout,
and truncates returned output at 4,000 characters. **Approval is not a security
sandbox.** An approved shell command runs with the current user's normal
permissions and can modify or delete files. Review every proposed command and
approve only commands you understand and trust.

## Course example attribution

This application adapts patterns from the course repository's `03_Agents`
examples: `01_tools_python.py` for Gemini agent setup, `04_tools_custom_decorator.py`
and `05_tools_custom_pydantic.py` for custom tools and validated inputs, and
`06_langgraph_linux.py` for human confirmation before terminal execution.
