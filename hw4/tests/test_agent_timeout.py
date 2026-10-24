"""No-LLM tests for Gemini retry configuration and workflow timeouts."""

import asyncio
import sys
from pathlib import Path

import pytest
from google.genai.errors import APIError
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent


def test_build_model_disables_retries_and_sets_request_timeout(monkeypatch):
    """Gemini construction explicitly sets finite request and retry limits."""
    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent, "ChatGoogleGenerativeAI", fake_model)
    agent._build_model("test-model", "test-key")

    assert captured["model"] == "test-model"
    assert captured["request_timeout"] == agent.MODEL_REQUEST_TIMEOUT_SECONDS
    assert captured["retries"] == 0


def test_langchain_timeout_is_seconds_and_reaches_genai_as_milliseconds():
    """A 60-second LangChain setting becomes 60000 ms in google-genai."""
    model = ChatGoogleGenerativeAI(
        model="offline-test",
        google_api_key="unit-test-key",
        request_timeout=agent.MODEL_REQUEST_TIMEOUT_SECONDS,
        retries=agent.MODEL_RETRIES,
    )

    request = model._prepare_request([HumanMessage(content="offline config check")])
    http_options = request["config"].http_options

    assert agent.MODEL_REQUEST_TIMEOUT_SECONDS == 60
    assert http_options.timeout == 60_000
    assert http_options.retry_options.attempts == 0
    assert agent.AGENT_WORKFLOW_TIMEOUT_SECONDS == 240


def test_only_four_required_mcp_tools_are_exposed():
    """Extra tools discovered from either MCP server stay out of the agent."""
    class StubTool:
        def __init__(self, name):
            self.name = name

    available = [
        StubTool(name)
        for name in (*agent.REQUIRED_MCP_TOOLS, "list_directory", "move_file", "delete_file")
    ]

    selected = agent._select_workflow_tools(available)

    assert [tool.name for tool in selected] == list(agent.REQUIRED_MCP_TOOLS)


def test_google_api_error_format_is_concise():
    """Google API failures keep status and message, not full response details."""
    error = APIError(
        504,
        {"status": "DEADLINE_EXCEEDED", "message": "request timed out", "details": ["verbose payload"]},
    )

    formatted = agent._format_google_api_error(error)

    assert "HTTP 504" in formatted
    assert "DEADLINE_EXCEEDED" in formatted
    assert "request timed out" in formatted
    assert "verbose payload" not in formatted


def test_main_handles_google_api_error_without_traceback(monkeypatch, capsys):
    """The CLI prints a concise Google API failure and exits cleanly."""
    monkeypatch.setattr(sys, "argv", ["agent.py", "--hours", "4"])

    def fail_without_running_agent(coroutine):
        coroutine.close()
        raise APIError(504, {"status": "DEADLINE_EXCEEDED", "message": "request timed out"})

    monkeypatch.setattr(agent.asyncio, "run", fail_without_running_agent)
    with pytest.raises(SystemExit) as error:
        agent.main()

    output = capsys.readouterr().err
    assert error.value.code == 1
    assert "Google Gemini API error (HTTP 504, DEADLINE_EXCEEDED)" in output
    assert "Traceback" not in output


def test_agent_workflow_timeout_has_clear_error(monkeypatch):
    """A stalled invoke is cancelled and reports its bounded timeout."""
    monkeypatch.setattr(agent, "AGENT_WORKFLOW_TIMEOUT_SECONDS", 0.01)

    class SlowAgent:
        async def ainvoke(self, request, config):
            await asyncio.sleep(1)

    async def invoke():
        with pytest.raises(RuntimeError, match="workflow timed out after 0.01 seconds"):
            await agent._invoke_agent_with_timeout(SlowAgent(), {}, {})

    asyncio.run(invoke())
