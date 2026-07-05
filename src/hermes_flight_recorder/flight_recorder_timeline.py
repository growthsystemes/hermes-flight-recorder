"""Local timeline and redaction checks for Hermes Flight Recorder JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SECRET_VALUE_PATTERNS = {
    # `sk-` is anchored to a word boundary so real OpenAI-style keys (`sk-proj-…`)
    # still match while internal identifiers that merely contain the substring
    # (e.g. "ta·sk-·store-scan") do not produce false positives.
    "api_key_like": re.compile(r"(\bsk-[A-Za-z0-9_-]+|api[_-]?key\s*[:=]|secret\s*[:=]|token\s*[:=]|authorization\s*[:=]|bearer\s+[A-Za-z0-9._~+/=-]+)", re.I),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "url": re.compile(r"https?://", re.I),
    "private_key": re.compile(r"-----BEGIN .*PRIVATE KEY-----"),
}
PATH_PATTERN = re.compile(r"(^|[\s\"'=])(/[A-Za-z0-9._~/-]+|[A-Za-z]:\\[^\s\"']+)")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if isinstance(value, dict):
                events.append(value)
    return events


def event_label(event: dict[str, Any], *, show_hashes: bool = False, show_previews: bool = False) -> str:
    event_type = str(event.get("event_type") or "event")
    phase = str(event.get("phase") or "instant")
    status = str(event.get("status") or "ok")
    duration = f" {event['duration_ms']}ms" if event.get("duration_ms") is not None else ""
    parts = [f"{event_type} {phase} {status}{duration}"]

    tool = event.get("tool") if isinstance(event.get("tool"), dict) else {}
    if tool.get("name"):
        parts.append(f"tool={tool['name']}")

    model = event.get("model") if isinstance(event.get("model"), dict) else {}
    if model.get("name"):
        provider = model.get("provider") or "unknown"
        parts.append(f"model={provider}/{model['name']}")

    runtime = event.get("runtime") if isinstance(event.get("runtime"), dict) else {}
    if runtime.get("decision"):
        parts.append(f"decision={runtime['decision']}")

    if show_hashes:
        for container in (tool, runtime):
            for key, value in container.items():
                if key.endswith("_hmac") or key.endswith("_sha256"):
                    parts.append(f"{key}={value}")
                    break

    if show_previews:
        for container in (tool, runtime):
            for key, value in container.items():
                if key.endswith("_preview"):
                    parts.append(f"{key}={value}")
                    break

    session = event.get("session_id")
    turn = event.get("turn_id")
    if session:
        parts.append(f"session={session}")
    if turn and turn != session:
        parts.append(f"turn={turn}")
    return " ".join(str(part) for part in parts)


def render_timeline(events: list[dict[str, Any]], args: argparse.Namespace) -> str:
    filtered = [
        event for event in events
        if (not args.session or str(event.get("session_id")) == args.session)
        and (not args.turn or str(event.get("turn_id")) == args.turn)
        and (args.show_errors or event.get("status") != "error")
        and (args.show_policy or not str(event.get("event_type", "")).startswith("runtime.policy"))
        and (args.show_side_effects or not event.get("side_effects"))
    ]
    filtered.sort(key=lambda event: str(event.get("timestamp") or event.get("start_ts") or event.get("end_ts") or ""))

    children: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, Any]] = {}
    for event in filtered:
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            by_id[event_id] = event
    for event in filtered:
        parent_id = event.get("parent_event_id")
        children[parent_id if isinstance(parent_id, str) and parent_id in by_id else None].append(event)

    lines: list[str] = []

    def visit(event: dict[str, Any], depth: int) -> None:
        lines.append("  " * depth + event_label(event, show_hashes=args.show_hashes, show_previews=args.show_previews))
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            for child in children.get(event_id, []):
                visit(child, depth + 1)

    for root in children.get(None, []):
        visit(root, 0)
    return "\n".join(lines)


def scalar_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(scalar_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(scalar_values(item))
        return values
    if isinstance(value, str):
        return [value]
    return []


def redaction_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    raw_payload_fields = 0
    preview_fields = 0
    pattern_hits: dict[str, int] = {name: 0 for name in SECRET_VALUE_PATTERNS}
    path_hits = 0
    urls_raw = 0
    emails_raw = 0

    for event in events:
        for key, value in walk_items(event):
            if key.endswith("_full"):
                raw_payload_fields += 1
            if key.endswith("_preview"):
                preview_fields += 1
            if key.endswith("_hmac") or key.endswith("_sha256") or key in {"event_hash", "previous_event_hash"}:
                continue
            if key in {"schema_version", "recorder_version", "semconv_version", "otel_mapping_version", "event_type"}:
                continue
            if isinstance(value, (dict, list)):
                # walk_items() already emits a separate entry for every leaf
                # inside this container; scanning it again here via
                # scalar_values() would re-match every nested string once per
                # ancestor level, inflating pattern_hits/urls_raw/emails_raw
                # by (nesting depth + 1)x for a single real occurrence.
                continue
            for text in scalar_values(value):
                for name, pattern in SECRET_VALUE_PATTERNS.items():
                    if pattern.search(text):
                        pattern_hits[name] += 1
                        if name == "url":
                            urls_raw += 1
                        if name == "email":
                            emails_raw += 1
                if PATH_PATTERN.search(text):
                    path_hits += 1

    possible_secret_patterns = sum(pattern_hits.values()) + path_hits
    return {
        "events": len(events),
        "raw_payload_fields": raw_payload_fields,
        "preview_fields": preview_fields,
        "possible_secret_patterns": possible_secret_patterns,
        "urls_raw": urls_raw,
        "paths_raw": path_hits,
        "emails_raw": emails_raw,
        "pattern_hits": {key: count for key, count in pattern_hits.items() if count},
    }


def walk_items(value: Any) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            items.append((key_text, item))
            items.extend(walk_items(item))
    elif isinstance(value, list):
        for item in value:
            items.extend(walk_items(item))
    return items


def render_redaction_report(report: dict[str, Any]) -> str:
    return "\n".join([
        "Redaction report:",
        f"- events: {report['events']}",
        f"- raw payload fields exported: {report['raw_payload_fields']}",
        f"- preview fields exported: {report['preview_fields']}",
        f"- possible secret patterns: {report['possible_secret_patterns']}",
        f"- urls raw: {report['urls_raw']}",
        f"- paths raw: {report['paths_raw']}",
        f"- emails raw: {report['emails_raw']}",
    ])


def timeline_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(event.get("status") or "ok") for event in events)
    event_types = Counter(str(event.get("event_type") or "event") for event in events)
    tools = Counter(
        str(tool.get("name"))
        for event in events
        for tool in [event.get("tool") if isinstance(event.get("tool"), dict) else {}]
        if tool.get("name")
    )
    durations = [event.get("duration_ms") for event in events if isinstance(event.get("duration_ms"), int)]
    return {
        "events": len(events),
        "event_types": dict(sorted(event_types.items())),
        "statuses": dict(sorted(statuses.items())),
        "tools": dict(sorted(tools.items())),
        "duration_ms": {
            "count": len(durations),
            "max": max(durations) if durations else None,
            "total": sum(durations) if durations else 0,
        },
    }


def render_timeline_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Timeline summary:",
        f"- events: {summary['events']}",
        f"- statuses: {json.dumps(summary['statuses'], ensure_ascii=False, sort_keys=True)}",
        f"- event types: {json.dumps(summary['event_types'], ensure_ascii=False, sort_keys=True)}",
        f"- tools: {json.dumps(summary['tools'], ensure_ascii=False, sort_keys=True)}",
        f"- duration max ms: {summary['duration_ms']['max']}",
        f"- duration total ms: {summary['duration_ms']['total']}",
    ]
    return "\n".join(lines)


def structural_report(events: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from .flight_recorder_schema import validate_event_schema
    except Exception as exc:
        return {
            "events": len(events),
            "schema_violations": [f"schema_validator_unavailable:{type(exc).__name__}"],
            "invalid_events": len(events),
            "unknown_event_type_events": 0,
            "unknown_event_types": [],
        }

    violations: list[dict[str, Any]] = []
    unknown_types: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        problems = validate_event_schema(event)
        if not problems:
            continue
        # An unrecognized event_type, on its own, is a forward-compatibility
        # warning -- not a malformation. The offline analyzer stays fail-open like
        # the live recorder so a newly-added event type (e.g. a future memory.*,
        # db.* or side-effect type) never hard-fails an evidence/canary gate run
        # from a checkout whose schema allowlist is older than the producer image.
        if problems == ["event_type_unknown"]:
            unknown_types.append({
                "index": index,
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
            })
            continue
        violations.append({
            "index": index,
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "problems": problems,
        })

    # OTEL trace correctness: span-id uniqueness + parent referential integrity.
    # A span_id may legitimately be shared only by a start/end pair of ONE span;
    # two instant/decision events, a repeated phase, or two event_types on one
    # span_id means distinct events collapsed onto one span (breaks trace viewers).
    span_groups: dict[str, dict[str, Any]] = {}
    present_span_ids: set[str] = set()
    for event in events:
        sid = event.get("span_id")
        if not sid:
            continue
        present_span_ids.add(sid)
        rec = span_groups.setdefault(sid, {"phases": [], "types": set()})
        rec["phases"].append(event.get("phase"))
        rec["types"].add(event.get("event_type"))
    duplicate_span_ids: list[dict[str, Any]] = []
    for sid, rec in span_groups.items():
        phases = rec["phases"]
        repeated_phase = len(phases) != len(set(phases))
        pointlike_shared = len(phases) > 1 and any(p in ("instant", "decision") for p in phases)
        multi_type = len([t for t in rec["types"] if t]) > 1
        if repeated_phase or pointlike_shared or multi_type:
            duplicate_span_ids.append({
                "span_id": sid,
                "count": len(phases),
                "phases": phases,
                "event_types": sorted(t for t in rec["types"] if t),
            })
    dangling_parents: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        parent = event.get("parent_span_id")
        if parent and parent not in present_span_ids:
            dangling_parents.append({
                "index": index,
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "parent_span_id": parent,
            })

    return {
        "events": len(events),
        "invalid_events": len(violations),
        "schema_violations": violations,
        "unknown_event_type_events": len(unknown_types),
        "unknown_event_types": unknown_types,
        "duplicate_span_id_groups": len(duplicate_span_ids),
        "duplicate_span_ids": duplicate_span_ids,
        "dangling_parent_events": len(dangling_parents),
        "dangling_parents": dangling_parents,
    }


def render_structural_report(report: dict[str, Any]) -> str:
    lines = [
        "Structural report:",
        f"- events: {report['events']}",
        f"- invalid events: {report['invalid_events']}",
    ]
    if report["schema_violations"]:
        lines.append(f"- schema violations: {json.dumps(report['schema_violations'], ensure_ascii=False)}")
    else:
        lines.append("- schema violations: 0")
    unknown_count = report.get("unknown_event_type_events", 0)
    if unknown_count:
        lines.append(
            f"- unknown event types (warning, fail-open): {unknown_count} "
            f"{json.dumps(report.get('unknown_event_types', []), ensure_ascii=False)}"
        )
    else:
        lines.append("- unknown event types: 0")
    dup = report.get("duplicate_span_id_groups", 0)
    if dup:
        lines.append(
            f"- duplicate span_ids (OTEL uniqueness violation): {dup} "
            f"{json.dumps(report.get('duplicate_span_ids', []), ensure_ascii=False)}"
        )
    else:
        lines.append("- duplicate span_ids: 0")
    dangling = report.get("dangling_parent_events", 0)
    if dangling:
        lines.append(
            f"- dangling parent_span_ids (broken trace linkage): {dangling} "
            f"{json.dumps(report.get('dangling_parents', []), ensure_ascii=False)}"
        )
    else:
        lines.append("- dangling parent_span_ids: 0")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Hermes Flight Recorder JSONL as a local timeline.")
    parser.add_argument("path", type=Path, help="Flight Recorder JSONL path.")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--show-policy", action="store_true")
    parser.add_argument("--show-side-effects", action="store_true")
    parser.add_argument("--show-hashes", action="store_true")
    parser.add_argument("--show-previews", action="store_true")
    parser.add_argument("--session")
    parser.add_argument("--turn")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--redaction-report", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--structural-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events = load_jsonl(args.path)
    if args.redaction_report:
        report = redaction_report(events)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_redaction_report(report))
        failed = report["raw_payload_fields"] or report["preview_fields"] or report["possible_secret_patterns"]
        return 1 if failed else 0
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


if __name__ == "__main__":
    raise SystemExit(main())
