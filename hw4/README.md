# Homework 4: Study Planner

**Author:** Rayyan Khan (`khanr2024`)

The Study Planner uses a Gemini-backed LangChain agent with two stdio MCP
servers. The custom Python server ranks assignments by deadline and importance,
then allocates a limited number of study hours and reports unfinished work.
The official filesystem server reads `workspace/assignments.json` and writes
`workspace/study_plan.md`.

## Filesystem MCP server source

The filesystem server is the published
[`@modelcontextprotocol/server-filesystem`](https://www.npmjs.com/package/@modelcontextprotocol/server-filesystem)
package from the official
[`modelcontextprotocol/servers` `src/filesystem` source](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem).
I verified the package's repository/source and selected the official `2.0.0`
release before installing it. The exact version is recorded in
`package.json` and `package-lock.json`. This release requires Node.js 24 or
later. The agent passes only the absolute `hw4/workspace` path as its allowed
directory; `.env` stays in the parent `hw4` directory, outside that allow-list.

## Setup on Windows PowerShell

Run these commands from this directory:

```powershell
uv python install 3.12
uv sync --python 3.12
npm.cmd install
Copy-Item .env.example .env
notepad .env
```

Put your private Google AI Studio key in `GOOGLE_API_KEY` and a Gemini model
name enabled for your account in `GOOGLE_MODEL`. Keep `.env` private; it is
ignored by Git and is outside the filesystem MCP server's allowed directory.
Do not put credentials in `workspace`.

## Run and verify

Run the automated tests, including the no-LLM MCP smoke test:

```powershell
uv run --python 3.12 pytest -q
```

The smoke test starts both actual stdio servers, verifies their tools and the
filesystem allow-list, reads the sample assignments through MCP, and invokes
the planning tools. It does not contact Gemini.

After setting credentials, run the live Gemini workflow with a study budget in
hours:

```powershell
uv run --python 3.12 agent.py --hours 5
```

The agent prints tool calls/results and saves the generated plan to
`workspace/study_plan.md`. The live LLM workflow is separate from the no-LLM
smoke test and requires valid Google credentials.
