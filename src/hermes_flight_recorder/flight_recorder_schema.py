"""Schema validation helpers for Hermes Flight Recorder events."""

from __future__ import annotations

import re
from typing import Any


SCHEMA_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
HEX_16_PATTERN = re.compile(r"^[0-9a-f]{16}$")
HEX_32_PATTERN = re.compile(r"^[0-9a-f]{32}$")

ALLOWED_PHASES = {"start", "end", "instant", "decision"}
ALLOWED_STATUSES = {"ok", "error", "blocked", "cancelled"}
ALLOWED_CAPTURE_MODES = {"metadata", "preview", "full", "forensic"}

# ---------------------------------------------------------------------------
# Event-type vocabulary, partitioned by *who actually emits it*. The split is
# the honest contract behind the "Flight Recorder" name: it documents which
# types have a live producer in this runtime today vs. which are vocabulary kept
# for future agents / replay. The call-site invariant is locked by a test
# (tests/test_flight_recorder.py::test_event_type_split_matches_call_sites).
# ---------------------------------------------------------------------------

# LIVE — emitted unconditionally by the DataForge MCP-router runtime (src/main.py).
# Every entry must have a literal `event_type="…"` call-site in main.py.
LIVE_EVENT_TYPES = {
    "hermes.event",        # admission control + SSE streaming lifecycle
    "hermes.session",      # session lifecycle
    "hermes.turn",         # turn lifecycle + quality signals
    "planning.route",      # objective → capability routing
    "llm.call",            # synthesis call (incl. synthesis-disabled spend)
    "llm.retry",           # backend-synthesis transient retry
    "llm.fallback",        # synthesis degraded to a non-model answer
    "tool.call",           # capability execution (incl. blocked)
    "mcp.transport",       # MCP tools/call + initialize + tools/list (+ retries)
    "web.request",         # result-webhook egress
    "memory.search",       # task-store keyspace scan (admission capacity check)
    "memory.write",        # task-store set
    "memory.read",         # task-store get
    "runtime.policy.allow",  # per-turn authorization summary (pass side)
    "runtime.policy.deny",   # capability / scope / execution-mode / rate-limit refusal
    "mcp.tools.snapshot",    # visible tool/capability catalog at decision time
    "mcp.tools.diff",        # drift in the visible tool/capability catalog
    "eval.score",            # deterministic online quality/governance score
}

# VM-SANDBOX-BETA - real emitters exist via the default-off code-runner surface
# and have passed isolated Docker + Kubernetes sandbox canaries. They are not
# permanent LIVE until governed activation and repeated VM/microVM proof land.
# They MUST NOT appear in main.py while `/internal/code-exec` remains gated.
VM_SANDBOX_BETA_EVENT_TYPES = {
    "process.exec",
    "file.read",
    "file.write",
}

# GATED - real emitters exist but have not yet passed the required activation
# proof. Currently empty after the code-runner events moved to VM-SANDBOX-BETA.
GATED_EVENT_TYPES: set[str] = set()

# RESERVED — declared vocabulary with NO producer in this runtime. They map to
# real upstream NousResearch Hermes toolsets (terminal/file/browser/delegation/
# skills/eval) that the DataForge MCP-router does not run, plus research/HITL
# verbs. Kept for fixture replay and future agents; never fabricated here.
RESERVED_EVENT_TYPES = {
    "hermes.checkpoint",
    "file.delete",
    "file.diff",
    "browser.action",
    "web.extract",
    "screenshot.capture",
    "skill.load",
    "skill.create",
    "skill.update",
    "subagent.start",
    "subagent.end",
    "subagent.delegate",
    "runtime.secret.withheld",
    "trajectory.export",
    "human.label",
}

ALLOWED_EVENT_TYPES = (
    LIVE_EVENT_TYPES
    | VM_SANDBOX_BETA_EVENT_TYPES
    | GATED_EVENT_TYPES
    | RESERVED_EVENT_TYPES
)


def validate_event_schema(event: dict[str, Any]) -> list[str]:
    """Return schema contract violations without raising.

    The recorder is fail-open by default, so validation is intentionally
    dependency-free and returns low-cardinality problem codes suitable for
    status endpoints and local structural reports.
    """
    problems: list[str] = []

    _require_string(event, "schema_version", problems, pattern=SCHEMA_VERSION_PATTERN)
    _require_string(event, "recorder_version", problems, pattern=SCHEMA_VERSION_PATTERN)
    _require_string(event, "event_id", problems)
    _require_string(event, "trace_id", problems, pattern=HEX_32_PATTERN)
    _require_string(event, "span_id", problems, pattern=HEX_16_PATTERN)
    _require_string(event, "event_type", problems)
    _require_string(event, "phase", problems)
    _require_string(event, "timestamp", problems)
    _require_string(event, "status", problems)
    _require_string(event, "actor", problems)

    event_type = event.get("event_type")
    if isinstance(event_type, str) and event_type not in ALLOWED_EVENT_TYPES:
        problems.append("event_type_unknown")

    phase = event.get("phase")
    if isinstance(phase, str) and phase not in ALLOWED_PHASES:
        problems.append("phase_unknown")

    status = event.get("status")
    if isinstance(status, str) and status not in ALLOWED_STATUSES:
        problems.append("status_unknown")

    if "duration_ms" in event and event["duration_ms"] is not None and not isinstance(event["duration_ms"], int):
        problems.append("duration_ms_not_int")

    privacy = event.get("privacy")
    if not isinstance(privacy, dict):
        problems.append("privacy_missing")
    else:
        capture_mode = privacy.get("capture_mode")
        if capture_mode not in ALLOWED_CAPTURE_MODES:
            problems.append("privacy_capture_mode_unknown")
        if "content_export_allowed" in privacy and not isinstance(privacy["content_export_allowed"], bool):
            problems.append("privacy_content_export_allowed_not_bool")

    tool = event.get("tool")
    if tool is not None:
        if not isinstance(tool, dict):
            problems.append("tool_not_object")
        elif event_type in {"tool.call", "mcp.transport"} and not isinstance(tool.get("name"), str):
            problems.append("tool_name_missing")

    model = event.get("model")
    if model is not None and not isinstance(model, dict):
        problems.append("model_not_object")

    side_effects = event.get("side_effects")
    if side_effects is not None:
        if not isinstance(side_effects, list):
            problems.append("side_effects_not_array")
        else:
            for item in side_effects:
                if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                    problems.append("side_effect_invalid")
                    break

    runtime = event.get("runtime")
    if runtime is not None and not isinstance(runtime, dict):
        problems.append("runtime_not_object")

    return problems


def _require_string(
    event: dict[str, Any],
    key: str,
    problems: list[str],
    *,
    pattern: re.Pattern[str] | None = None,
) -> None:
    value = event.get(key)
    if not isinstance(value, str) or not value:
        problems.append(f"{key}_missing")
        return
    if pattern is not None and not pattern.match(value):
        problems.append(f"{key}_invalid")
