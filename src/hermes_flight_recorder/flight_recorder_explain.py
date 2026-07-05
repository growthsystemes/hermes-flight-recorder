"""`flight-recorder explain` — a one-command, human-readable run summary.

Turns the low-level JSONL into the answer an operator actually wants:

    $ python -m hermes_flight_recorder.flight_recorder_explain events.jsonl --run <id>
    Run: <id>
    Status: failed
    Probable cause: MCP transport error (df_answer_ontology_question)
    Selected capabilities: df_answer_ontology_question, df_entity_dossier
    Tools attempted: 3
    Blocked tools: 1
    Retries: 2  |  Policy denies: 0
    LLM cost: $0.0042
    Privacy: metadata — 0 raw payload leaks

Reads the canonical JSONL only; never reaches the network. `--json` emits the
same data as a machine-readable object.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .flight_recorder_timeline import load_jsonl, redaction_report

_RUN_KEYS = ("run_id", "task_id", "session_id", "turn_id")
_STATUS_DISPLAY = {"ok": "completed", "error": "failed", "blocked": "blocked", "cancelled": "cancelled"}


def events_for_run(events: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    """Events whose run/task/session/turn id matches `run_id`."""
    return [e for e in events if any(str(e.get(k)) == run_id for k in _RUN_KEYS)]


def list_runs(events: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for e in events:
        rid = e.get("run_id") or e.get("task_id") or e.get("session_id")
        if rid is not None:
            seen.setdefault(str(rid), None)
    return list(seen)


def _is_retry(event: dict[str, Any]) -> bool:
    return bool((event.get("attributes") or {}).get("retry")) or event.get("event_type") == "llm.retry"


def _derive_status(run_events: list[dict[str, Any]]) -> str:
    # Prefer the turn/session end status; else fold the observed statuses.
    for et in ("hermes.turn", "hermes.session"):
        ends = [e for e in run_events if e.get("event_type") == et and e.get("phase") == "end"]
        if ends:
            return str(ends[-1].get("status") or "ok")
    statuses = {str(e.get("status") or "ok") for e in run_events}
    for s in ("error", "cancelled", "blocked"):
        if s in statuses:
            return s
    return "ok"


def _probable_cause(run_events: list[dict[str, Any]], status: str) -> str:
    if status == "ok":
        return "completed normally"
    fallbacks = [e for e in run_events if e.get("event_type") == "llm.fallback"]
    denies = [e for e in run_events if e.get("event_type") == "runtime.policy.deny"]
    mcp_errors = [
        e for e in run_events
        if e.get("event_type") == "mcp.transport" and str(e.get("status")) == "error"
    ]
    if denies:
        rt = denies[-1].get("runtime") or {}
        reason = (rt.get("metadata") or {}).get("reason") or rt.get("policy_type") or "policy"
        return f"blocked by policy: {rt.get('policy_type', 'policy')} ({reason})"
    if mcp_errors:
        tool = (mcp_errors[-1].get("tool") or {}).get("name") or "mcp"
        return f"MCP transport error ({tool})"
    if fallbacks:
        reason = (fallbacks[-1].get("attributes") or {}).get("reason") or "synthesis_degraded"
        return f"LLM synthesis degraded: {reason}"
    errs = [e for e in run_events if str(e.get("status")) == "error"]
    if errs:
        return f"error during {errs[-1].get('event_type')}"
    return status


def explain_run(events: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    run_events = events_for_run(events, run_id)
    if not run_events:
        return {"run": run_id, "found": False}

    status = _derive_status(run_events)

    allows = [e for e in run_events if e.get("event_type") == "runtime.policy.allow"]
    if allows:
        selected = list((allows[-1].get("runtime") or {}).get("metadata", {}).get("allowed") or [])
    else:
        selected = sorted({
            str((e.get("tool") or {}).get("name"))
            for e in run_events
            if e.get("event_type") == "tool.call" and (e.get("tool") or {}).get("name")
        })

    tool_calls = [e for e in run_events if e.get("event_type") == "tool.call"]
    blocked = [e for e in tool_calls if str(e.get("status")) == "blocked"]
    attempted = sorted({
        str((e.get("tool") or {}).get("name"))
        for e in tool_calls
        if str(e.get("status")) != "blocked" and (e.get("tool") or {}).get("name")
    })

    llm_cost = 0.0
    for e in run_events:
        if e.get("event_type") == "llm.call":
            c = (e.get("model") or {}).get("cost_usd")
            if isinstance(c, (int, float)):
                llm_cost += float(c)

    retries = sum(1 for e in run_events if _is_retry(e))
    denies = sum(1 for e in run_events if e.get("event_type") == "runtime.policy.deny")

    durations = [
        e.get("duration_ms")
        for e in run_events
        if e.get("event_type") in ("hermes.turn", "hermes.session") and isinstance(e.get("duration_ms"), int)
    ]
    duration_ms = max(durations) if durations else None

    capture_modes = {
        (e.get("privacy") or {}).get("capture_mode")
        for e in run_events
        if isinstance(e.get("privacy"), dict)
    }
    capture_mode = next(iter(capture_modes), "metadata") if len(capture_modes) <= 1 else "mixed"
    redaction = redaction_report(run_events)

    return {
        "run": run_id,
        "found": True,
        "events": len(run_events),
        "status": status,
        "status_display": _STATUS_DISPLAY.get(status, status),
        "probable_cause": _probable_cause(run_events, status),
        "selected_capabilities": selected,
        "tools_attempted": attempted,
        "tools_attempted_count": len(attempted),
        "blocked_tools_count": len(blocked),
        "retries": retries,
        "policy_denies": denies,
        "llm_cost_usd": round(llm_cost, 6),
        "duration_ms": duration_ms,
        "privacy": {
            "capture_mode": capture_mode,
            "raw_payload_leaks": redaction["raw_payload_fields"],
            "preview_fields": redaction["preview_fields"],
            "secret_pattern_hits": redaction["possible_secret_patterns"],
        },
    }


def render_explanation(data: dict[str, Any]) -> str:
    if not data.get("found"):
        return f"Run: {data['run']}\n(no events found for this run)"
    p = data["privacy"]
    leaks = p["raw_payload_leaks"] + p["secret_pattern_hits"]
    lines = [
        f"Run: {data['run']}",
        f"Status: {data['status_display']}",
        f"Probable cause: {data['probable_cause']}",
        f"Selected capabilities: {', '.join(data['selected_capabilities']) or 'none'}",
        f"Tools attempted: {data['tools_attempted_count']}",
        f"Blocked tools: {data['blocked_tools_count']}",
        f"Retries: {data['retries']}  |  Policy denies: {data['policy_denies']}",
        f"LLM cost: ${data['llm_cost_usd']:.4f}",
    ]
    if data["duration_ms"] is not None:
        lines.append(f"Duration: {data['duration_ms']} ms")
    lines.append(f"Privacy: {p['capture_mode']} — {leaks} raw payload leaks")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain a Flight Recorder run in one command.")
    parser.add_argument("path", type=Path, help="Flight Recorder JSONL path.")
    parser.add_argument("--run", help="Run id (run_id/task_id/session_id/turn_id). Omit to list runs.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events = load_jsonl(args.path)
    if not args.run:
        runs = list_runs(events)
        if args.json:
            print(json.dumps({"runs": runs}, ensure_ascii=False))
        else:
            print("Runs:" if runs else "No runs found.")
            for r in runs:
                print(f"  {r}")
        return 0
    data = explain_run(events, args.run)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(render_explanation(data))
    return 0 if data.get("found") else 1


if __name__ == "__main__":
    sys.exit(main())
