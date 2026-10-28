"""Isolated security-testing runner for the unchanged Homework 4 agent."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HW5_ROOT = Path(__file__).resolve().parent
REPO_ROOT = HW5_ROOT.parent
HW4_ROOT = REPO_ROOT / "hw4"
DEFAULT_FIXTURE = HW5_ROOT / "baseline" / "assignments.json"


def _import_agent() -> Any:
    """Import the original hw4/agent.py from the hw4 dependency environment."""
    if str(HW4_ROOT) not in sys.path:
        sys.path.insert(0, str(HW4_ROOT))
    spec = importlib.util.spec_from_file_location("hw4_agent", HW4_ROOT / "agent.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the Homework 4 agent from {HW4_ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agent = _import_agent()


def _redact(value: Any) -> Any:
    """Redact credentials from evidence while preserving tool-call structure."""
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    secret = os.environ.get("GOOGLE_API_KEY", "").strip()
    if secret:
        rendered = rendered.replace(secret, "[REDACTED]")
    rendered = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", rendered)
    rendered = re.sub(
        r'(?i)("?(?:google_api_key|api_key|authorization|bearer)"?\s*[:=]\s*"?)[^",}\s]+',
        r"\1[REDACTED]",
        rendered,
    )
    return json.loads(rendered)


def _append_evidence(record: dict[str, Any]) -> None:
    evidence_path = getattr(agent, "_hw5_evidence_path", None)
    if evidence_path is None:
        return
    with evidence_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_redact(record), ensure_ascii=False, default=str) + "\n")


class EvidenceToolEventPrinter(agent.ToolEventPrinter):
    """Keep the original callback behavior and additionally save full events."""

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        super().on_tool_start(serialized, input_str, **kwargs)
        _append_evidence(
            {
                "type": "tool_call",
                "name": serialized.get("name", "MCP tool"),
                "inputs": kwargs.get("inputs", input_str),
            }
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        name = self._tool_names_by_run_id.get(run_id, "MCP tool")
        super().on_tool_end(output, **kwargs)
        _append_evidence({"type": "tool_result", "name": name, "result": output})

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        name = self._tool_names_by_run_id.get(run_id, "MCP tool")
        super().on_tool_error(error, **kwargs)
        _append_evidence({"type": "tool_error", "name": name, "error": f"{type(error).__name__}: {error}"})


async def _capture_agent_result(original: Any, agent_obj: Any, request: dict[str, Any], config: dict[str, Any]) -> Any:
    result = await original(agent_obj, request, config)
    final_message = result["messages"][-1]
    _append_evidence({"type": "final_output", "text": agent._message_text(final_message).strip()})
    return result


def _new_run_directory() -> Path:
    runs = HW5_ROOT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    while True:
        run_dir = runs / f"{timestamp}-{uuid.uuid4().hex[:8]}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue


def _run(fixture: Path, hours: float, model_name: str) -> int:
    if not fixture.is_file():
        raise FileNotFoundError(f"Fixture not found: {fixture}")
    # Validate fixture JSON before starting MCP processes or model work.
    with fixture.open("r", encoding="utf-8") as stream:
        json.load(stream)

    run_dir = _new_run_directory()
    workspace = run_dir / "workspace"
    workspace.mkdir()
    assignments = workspace / "assignments.json"
    assignments.write_bytes(fixture.read_bytes())
    evidence_path = run_dir / "evidence.jsonl"
    evidence_path.write_text("", encoding="utf-8")
    agent._hw5_evidence_path = evidence_path

    # Only the assignment, output, and filesystem allow-list are redirected.
    agent.WORKSPACE = workspace.resolve()
    agent.ASSIGNMENTS_FILE = assignments.resolve()
    agent.PLAN_FILE = (workspace / "study_plan.md").resolve()
    agent.ToolEventPrinter = EvidenceToolEventPrinter
    original_invoke = agent._invoke_agent_with_timeout
    agent._invoke_agent_with_timeout = lambda obj, request, config: _capture_agent_result(
        original_invoke, obj, request, config
    )
    os.environ["GOOGLE_MODEL"] = model_name

    manifest = {
        "fixture": str(fixture.resolve()),
        "hours": hours,
        "model": model_name,
        "workspace": str(workspace.resolve()),
        "evidence": str(evidence_path.resolve()),
        "plan": str(agent.PLAN_FILE),
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}")
    print(f"Evidence: {evidence_path}")
    asyncio.run(agent.run_agent(hours))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hw4's agent in an isolated hw5 workspace.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="Assignments JSON fixture.")
    parser.add_argument("--hours", type=float, help="Available study-time budget in hours.")
    parser.add_argument("--model", default="gemini-3.6-flash", help="Gemini model name.")
    parser.add_argument("--check-imports", action="store_true", help="Import hw4/agent.py without calling a model.")
    args = parser.parse_args()
    if args.check_imports:
        print(f"Imported {HW4_ROOT / 'agent.py'}")
        return 0
    if args.hours is None or args.hours < 0:
        parser.error("--hours must be provided and non-negative")
    return _run(args.fixture.resolve(), args.hours, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
