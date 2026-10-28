# Homework 5 security testing harness

**Author:** Rayyan Khan (`khanr2024`)

This harness imports the existing `hw4/agent.py` and reuses its installed
Python environment, filesystem MCP package, unchanged planner MCP server,
system prompt, tool selection, and required tool sequence validation. Each run
gets a new `hw5/runs/<timestamp>-<id>/workspace` directory. The runner redirects
only the assignment input path, plan output path, and filesystem MCP allowed
workspace. `hw4` files are left in place.

The baseline fixture at `baseline/assignments.json` is a clean copy of
`hw4/workspace/assignments.json`. Each run also saves a copied input fixture,
the plan output, a `run.json` manifest, and `evidence.jsonl` containing complete
tool inputs, results, errors, and final output. Evidence redacts API-key-shaped
values and credential fields. Credentials remain in the private `hw4/.env`,
outside the filesystem MCP allow-list; the harness never reads or prints its
contents.

## Windows PowerShell

Run these commands from the repository root, with the Homework 4 environment
and Node.js setup already available:

```powershell
.\hw4\.venv\Scripts\python.exe .\hw5\run.py --check-imports
.\hw4\.venv\Scripts\python.exe .\hw5\run.py --fixture .\hw5\baseline\assignments.json --hours 4 --model gemini-3.6-flash
```

The first command only imports `hw4/agent.py`; it does not start MCP servers or
make a model request. The second command is the four-hour baseline run. It
requires the existing private `hw4/.env` credentials and makes a live model
request when explicitly executed.
