"""Unified Hermes Flight Recorder CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .flight_recorder import RECORDER_VERSION, FlightRecorderSettings, HermesFlightRecorder
from .flight_recorder_explain import explain_run, list_runs, render_explanation
from .flight_recorder_index import default_index_path, rebuild_index
from .flight_recorder_query import compact_event, query_index, query_jsonl, render_events
from .flight_recorder_replay import replay_fixture
from .flight_recorder_timeline import (
    load_jsonl,
    redaction_report,
    render_redaction_report,
    render_structural_report,
    render_timeline,
    render_timeline_summary,
    structural_report,
    timeline_summary,
)


def explain_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    summary = timeline_summary(events)
    redaction = redaction_report(events)
    structure = structural_report(events)
    blocked_tools: set[str] = set()
    failed_tools: set[str] = set()
    retry_count = 0
    cost_usd = 0.0
    input_tokens = 0
    output_tokens = 0
    privacy_modes: set[str] = set()

    for event in events:
        privacy = event.get("privacy") if isinstance(event.get("privacy"), dict) else {}
        if privacy.get("capture_mode"):
            privacy_modes.add(str(privacy["capture_mode"]))

        tool = event.get("tool") if isinstance(event.get("tool"), dict) else {}
        tool_name = str(tool.get("name") or "")
        if event.get("status") == "blocked" and tool_name:
            blocked_tools.add(tool_name)
        if event.get("status") == "error" and tool_name:
            failed_tools.add(tool_name)

        attrs = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
        for key in ("retry_count", "retries"):
            if isinstance(attrs.get(key), int):
                retry_count += int(attrs[key])

        model = event.get("model") if isinstance(event.get("model"), dict) else {}
        if isinstance(model.get("cost_usd"), (int, float)):
            cost_usd += float(model["cost_usd"])
        if isinstance(model.get("input_tokens"), int):
            input_tokens += int(model["input_tokens"])
        if isinstance(model.get("output_tokens"), int):
            output_tokens += int(model["output_tokens"])

    failed = int(summary["statuses"].get("error", 0))
    blocked = int(summary["statuses"].get("blocked", 0))
    privacy_failed = bool(
        redaction["raw_payload_fields"] or redaction["preview_fields"] or redaction["possible_secret_patterns"]
    )
    if structure["invalid_events"]:
        likely_cause = "schema violations found in JSONL"
    elif privacy_failed:
        likely_cause = "privacy check failed"
    elif failed:
        likely_cause = "tool, model, or transport error recorded"
    elif blocked:
        likely_cause = "policy or capability block recorded"
    else:
        likely_cause = "completed without recorded errors"

    return {
        "events": summary["events"],
        "statuses": summary["statuses"],
        "event_types": summary["event_types"],
        "likely_cause": likely_cause,
        "blocked_tools": sorted(blocked_tools),
        "failed_tools": sorted(failed_tools),
        "retry_count": retry_count,
        "llm": {
            "cost_usd": round(cost_usd, 8),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "privacy": {
            "capture_modes": sorted(privacy_modes),
            "raw_payload_fields": redaction["raw_payload_fields"],
            "preview_fields": redaction["preview_fields"],
            "possible_secret_patterns": redaction["possible_secret_patterns"],
        },
        "schema": {
            "invalid_events": structure["invalid_events"],
            "unknown_event_type_events": structure.get("unknown_event_type_events", 0),
        },
    }


def render_explain(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Flight Recorder explain:",
            f"- events: {report['events']}",
            f"- statuses: {json.dumps(report['statuses'], ensure_ascii=False, sort_keys=True)}",
            f"- likely cause: {report['likely_cause']}",
            f"- blocked tools: {', '.join(report['blocked_tools']) if report['blocked_tools'] else 'none'}",
            f"- failed tools: {', '.join(report['failed_tools']) if report['failed_tools'] else 'none'}",
            f"- retries: {report['retry_count']}",
            (
                "- llm: "
                f"input_tokens={report['llm']['input_tokens']} "
                f"output_tokens={report['llm']['output_tokens']} "
                f"cost_usd={report['llm']['cost_usd']}"
            ),
            (
                "- privacy: "
                f"modes={','.join(report['privacy']['capture_modes']) or 'unknown'} "
                f"raw={report['privacy']['raw_payload_fields']} "
                f"preview={report['privacy']['preview_fields']} "
                f"secret_patterns={report['privacy']['possible_secret_patterns']}"
            ),
            (
                "- schema: "
                f"invalid={report['schema']['invalid_events']} "
                f"unknown_event_types={report['schema']['unknown_event_type_events']}"
            ),
        ]
    )


def _privacy_failed(report: dict[str, Any]) -> bool:
    return bool(report["raw_payload_fields"] or report["preview_fields"] or report["possible_secret_patterns"])


def _run_replay(args: argparse.Namespace) -> int:
    summary = replay_fixture(
        input_path=args.input,
        output_path=args.output,
        capture_mode=args.capture_mode,
        preview_chars=args.preview_chars,
        append=args.append,
        otlp_payload_path=args.otlp_payload,
        otlp_include_previews=args.otlp_include_previews,
        service_name=args.service_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _run_timeline(args: argparse.Namespace) -> int:
    events = load_jsonl(args.path)
    if args.summary:
        summary = timeline_summary(events)
        print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render_timeline_summary(summary))
        return 0
    if args.structural_report:
        report = structural_report(events)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_structural_report(report))
        return 1 if report["invalid_events"] else 0
    if args.json:
        print(json.dumps(events, ensure_ascii=False, indent=2))
    else:
        print(render_timeline(events, args))
    return 0


def _run_query(args: argparse.Namespace) -> int:
    index_path = args.index or default_index_path(args.path)
    if args.rebuild_index:
        summary = rebuild_index(args.path, index_path)
        if args.summary and not args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))

    filters = {
        "event_type": args.event_type,
        "status": args.status,
        "tool_name": args.tool,
        "run_id": args.run,
        "session_id": args.session,
        "min_duration_ms": args.min_duration_ms,
        "failed": args.failed,
    }
    if index_path.exists():
        events = query_index(index_path, filters, limit=max(1, args.limit))
    else:
        events = query_jsonl(load_jsonl(args.path), filters, limit=max(1, args.limit))

    output: Any = timeline_summary(events) if args.summary else [compact_event(event) for event in events]
    if args.json or args.summary:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(render_events(events))
    return 0


def _run_explain(args: argparse.Namespace) -> int:
    events = load_jsonl(args.path)
    if not args.run:
        runs = list_runs(events)
        if args.json:
            print(json.dumps({"runs": runs}, ensure_ascii=False))
        else:
            print("Runs:" if runs else "No runs found.")
            for run_id in runs:
                print(f"  {run_id}")
        return 0

    data = explain_run(events, args.run)
    print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else render_explanation(data))
    return 0 if data.get("found") else 1


def _run_redact_check(args: argparse.Namespace) -> int:
    report = redaction_report(load_jsonl(args.path))
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_redaction_report(report))
    return 1 if _privacy_failed(report) else 0


def _run_doctor(args: argparse.Namespace) -> int:
    checks_run = 0
    failed = False

    if args.check_privacy:
        checks_run += 1
        report = redaction_report(load_jsonl(args.check_privacy))
        print(render_redaction_report(report))
        failed = failed or _privacy_failed(report)

    if args.check_secret:
        checks_run += 1
        recorder = HermesFlightRecorder.from_env()
        status = recorder.status()
        source = status["hash_secret_source"]
        weak = source in {"default-dev", "weak-env-fallback"}
        print(f"Secret check: hash_secret_source={source} strong={not weak}")
        failed = failed or weak

    if args.check_otlp:
        checks_run += 1
        if args.events is None:
            print("OTLP check: --events is required")
            return 1
        events = load_jsonl(args.events)
        recorder = HermesFlightRecorder(
            FlightRecorderSettings(enabled=True, otlp_include_previews=False, otlp_service_name=args.service_name)
        )
        payload = recorder.otlp_payload(events)
        serialized = json.dumps(payload, ensure_ascii=False)
        leaks = '"*_full"' in serialized or '"*_preview"' in serialized or "_full" in serialized or "_preview" in serialized
        if args.payload_out:
            args.payload_out.parent.mkdir(parents=True, exist_ok=True)
            args.payload_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OTLP check: spans={len(events)} preview_or_full_fields={leaks}")
        failed = failed or leaks

    if not checks_run:
        print("Doctor: no checks selected")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes Flight Recorder local CLI.")
    parser.add_argument("--version", action="version", version=f"hermes-flight-recorder {RECORDER_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="Replay fixture JSON/JSONL into canonical JSONL.")
    replay.add_argument("--input", "-i", required=True, type=Path)
    replay.add_argument("--output", "-o", required=True, type=Path)
    replay.add_argument("--capture-mode", default="metadata", choices=["metadata", "preview", "full", "forensic"])
    replay.add_argument("--preview-chars", default=500, type=int)
    replay.add_argument("--append", action="store_true")
    replay.add_argument("--otlp-payload", type=Path)
    replay.add_argument("--otlp-include-previews", action="store_true")
    replay.add_argument("--service-name", default="hermes-flight-recorder-fixture")
    replay.set_defaults(func=_run_replay)

    timeline = subparsers.add_parser("timeline", help="Render a local JSONL timeline.")
    timeline.add_argument("path", type=Path)
    timeline.add_argument("--show-errors", action="store_true")
    timeline.add_argument("--show-policy", action="store_true")
    timeline.add_argument("--show-side-effects", action="store_true")
    timeline.add_argument("--show-hashes", action="store_true")
    timeline.add_argument("--show-previews", action="store_true")
    timeline.add_argument("--session")
    timeline.add_argument("--turn")
    timeline.add_argument("--json", action="store_true")
    timeline.add_argument("--summary", action="store_true")
    timeline.add_argument("--structural-report", action="store_true")
    timeline.set_defaults(func=_run_timeline)

    query = subparsers.add_parser("query", help="Query JSONL or its SQLite index.")
    query.add_argument("path", type=Path)
    query.add_argument("--index", type=Path)
    query.add_argument("--rebuild-index", action="store_true")
    query.add_argument("--event-type")
    query.add_argument("--status")
    query.add_argument("--tool")
    query.add_argument("--run")
    query.add_argument("--session")
    query.add_argument("--min-duration-ms", type=int)
    query.add_argument("--failed", action="store_true")
    query.add_argument("--limit", type=int, default=50)
    query.add_argument("--summary", action="store_true")
    query.add_argument("--json", action="store_true")
    query.set_defaults(func=_run_query)

    explain = subparsers.add_parser("explain", help="Summarize status, cause, privacy, retries, and LLM usage.")
    explain.add_argument("path", type=Path)
    explain.add_argument("--run", help="Run id (run_id/task_id/session_id/turn_id). Omit to list runs.")
    explain.add_argument("--json", action="store_true")
    explain.set_defaults(func=_run_explain)

    redact = subparsers.add_parser("redact-check", help="Fail if JSONL appears to contain raw sensitive payloads.")
    redact.add_argument("path", type=Path)
    redact.add_argument("--json", action="store_true")
    redact.set_defaults(func=_run_redact_check)

    doctor = subparsers.add_parser("doctor", help="Run local Flight Recorder checks.")
    doctor.add_argument("--check-privacy", type=Path)
    doctor.add_argument("--check-secret", action="store_true")
    doctor.add_argument("--check-otlp", action="store_true")
    doctor.add_argument("--events", type=Path)
    doctor.add_argument("--payload-out", type=Path)
    doctor.add_argument("--service-name", default="hermes-flight-recorder-doctor")
    doctor.set_defaults(func=_run_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
