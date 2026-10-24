"""No-LLM tests for Gemini retry configuration and workflow timeouts."""

import asyncio
import sys
from pathlib import Path

import pytest

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
