"""No-LLM smoke test for the two real stdio MCP server processes."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import ASSIGNMENTS_FILE, WORKSPACE, server_config
from langchain_mcp_adapters.client import MultiServerMCPClient


def _plain_text(value):
    """Flatten common LangChain MCP tool return shapes for assertions."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in value
        )
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def test_real_mcp_servers_without_llm():
    """Connect to both real servers and exercise their discovered tools."""

    async def smoke():
        client = MultiServerMCPClient(server_config())
        tools = await client.get_tools()
        by_name = {tool.name: tool for tool in tools}
        assert {"rank_assignments", "allocate_study_time"} <= by_name.keys()
        assert {"read_text_file", "write_file", "list_allowed_directories"} <= by_name.keys()

        allowed = await by_name["list_allowed_directories"].ainvoke({})
        assert str(WORKSPACE).casefold() in _plain_text(allowed).casefold()

        raw_assignments = await by_name["read_text_file"].ainvoke({"path": str(ASSIGNMENTS_FILE)})
        assignments = json.loads(_plain_text(raw_assignments))
        assert len(assignments) == 3

        ranked = await by_name["rank_assignments"].ainvoke({"assignments": assignments})
        assert "Calculus problem set 6" in _plain_text(ranked)

        allocation = await by_name["allocate_study_time"].ainvoke(
            {"assignments": assignments, "available_hours": 1}
        )
        allocation_text = _plain_text(allocation)
        assert "unfinished_work" in allocation_text
        assert "unfinished" in allocation_text

    asyncio.run(smoke())
