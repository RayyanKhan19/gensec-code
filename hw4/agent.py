"""LangChain Study Planner agent for Rayyan Khan (khanr2024)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = (PROJECT_ROOT / "workspace").resolve()
ASSIGNMENTS_FILE = (WORKSPACE / "assignments.json").resolve()
PLAN_FILE = (WORKSPACE / "study_plan.md").resolve()
PLANNER_SERVER = (PROJECT_ROOT / "mcp-server.py").resolve()
FILESYSTEM_ENTRY = (
    PROJECT_ROOT / "node_modules" / "@modelcontextprotocol" / "server-filesystem" / "dist" / "index.js"
).resolve()


def server_config() -> dict[str, dict[str, Any]]:
    """Build absolute-path stdio configurations for both MCP servers."""
    python_exe = Path(os.sys.executable).resolve()
    node_exe = shutil.which("node")
    if not node_exe:
        raise RuntimeError("Node.js 24 or later is required; install Node.js and reopen PowerShell.")
    node_path = Path(node_exe).resolve()
    if not FILESYSTEM_ENTRY.is_file():
        raise RuntimeError("Filesystem MCP server is missing; run `npm install` inside hw4 first.")
    if not WORKSPACE.is_dir():
        raise RuntimeError(f"The allowed filesystem directory does not exist: {WORKSPACE}")
    return {
        "study_planner": {
            "transport": "stdio",
            "command": str(python_exe),
            "args": [str(PLANNER_SERVER)],
        },
        "filesystem": {
            "transport": "stdio",
            "command": str(node_path),
            "args": [str(FILESYSTEM_ENTRY), str(WORKSPACE)],
        },
    }


def _compact(value: Any, limit: int = 240) -> str:
    """Return a concise, one-line representation for visible tool events."""
    if isinstance(value, str):
        rendered = " ".join(value.split())
    else:
        rendered = " ".join(json.dumps(value, ensure_ascii=False, default=str).split())
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


class ToolEventPrinter(BaseCallbackHandler):
    """Print compact names, inputs, and results for LangChain tool calls."""

    def __init__(self) -> None:
        """Create the callback and its successful tool-call history."""
        self.completed_tools: list[str] = []

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Print the tool name and a compact representation of its inputs."""
        name = serialized.get("name", "MCP tool")
        print(f"[tool] {name}: {_compact(inputs if inputs is not None else input_str)}")

    def on_tool_end(self, output: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> None:
        """Print a compact representation of a tool result."""
        name = kwargs.get("name")
        if isinstance(name, str):
            self.completed_tools.append(name)
        print(f"[result] {_compact(output)}")

    def on_tool_error(self, error: BaseException, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> None:
        """Print a concise tool failure before LangChain propagates it."""
        print(f"[tool error] {type(error).__name__}: {_compact(str(error))}")


def _message_text(message: Any) -> str:
    """Extract readable text from a LangChain message content value."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


async def run_agent(available_hours: float) -> None:
    """Run the Gemini agent with both MCP servers and the requested budget."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    model_name = os.getenv("GOOGLE_MODEL", "").strip()
    if not api_key or not model_name:
        raise RuntimeError(
            "Set GOOGLE_API_KEY and GOOGLE_MODEL in your environment or private hw4/.env file. "
            "See .env.example for the variable names."
        )

    client = MultiServerMCPClient(server_config())
    tools = await client.get_tools()
    required_tools = {"read_text_file", "rank_assignments", "allocate_study_time", "write_file"}
    found_tools = {tool.name for tool in tools}
    missing = required_tools - found_tools
    if missing:
        raise RuntimeError(f"MCP servers did not provide required tools: {', '.join(sorted(missing))}")

    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a careful study-planning assistant. You must use MCP tools to do the work. "
            "First read the sample assignments with read_text_file using the provided absolute path. "
            "Parse its JSON list, call rank_assignments with that list, then call allocate_study_time "
            "with the same list and the provided available_hours. After you receive both planning "
            "results, use write_file to save a useful Markdown plan at the provided output path. "
            "Include the rank, due date, estimated and allocated hours, and all unfinished work. "
            "Do not claim hours are complete when allocation reports unfinished work. Keep the plan "
            "concise. Do not read or write files except through the filesystem MCP tools."
        ),
    )
    request = (
        f"Prepare a study plan using the MCP workflow. Read assignments from {ASSIGNMENTS_FILE}. "
        f"The available study budget is {available_hours:g} hours. Save the finished Markdown plan to {PLAN_FILE}."
    )
    events = ToolEventPrinter()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"callbacks": [events], "recursion_limit": 18},
    )
    required_order = ["read_text_file", "rank_assignments", "allocate_study_time", "write_file"]
    positions = [events.completed_tools.index(name) if name in events.completed_tools else -1 for name in required_order]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise RuntimeError(
            "The agent did not complete the required MCP sequence "
            "(read assignments, rank, allocate time, save plan). Review the tool events and try again."
        )
    final_message = result["messages"][-1]
    final_text = _message_text(final_message).strip()
    if final_text:
        print(f"\n{final_text}")


def main() -> None:
    """Parse CLI options and start the Study Planner agent."""
    parser = argparse.ArgumentParser(description="Create a Study Planner plan with Gemini and MCP tools.")
    parser.add_argument("--hours", type=float, required=True, help="Available study-time budget in hours.")
    args = parser.parse_args()
    if args.hours < 0:
        parser.error("--hours must be non-negative")
    try:
        asyncio.run(run_agent(args.hours))
    except RuntimeError as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
