"""Structured query CLI for Hermes Flight Recorder JSONL and SQLite index."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from .flight_recorder_index import default_index_path, rebuild_index
from .flight_recorder_timeline import load_jsonl, timeline_summary


def query_jsonl(events: list[dict[str, Any]], filters: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    results = [event for event in events if event_matches(event, filters)]
    results.sort(key=lambda event: str(event.get("timestamp") or ""), reverse=True)
    return results[:limit]


def event_matches(event: dict[str, Any], filters: dict[str, Any]) -> bool:
    if filters.get("event_type") and event.get("event_type") != filters["event_type"]:
        return False
    if filters.get("status") and event.get("status") != filters["status"]:
        return False
    if filters.get("failed") and event.get("status") not in {"error", "blocked"}:
        return False
    if filters.get("run_id") and event.get("run_id") != filters["run_id"]:
        return False
    if filters.get("session_id") and event.get("session_id") != filters["session_id"]:
        return False
    if filters.get("min_duration_ms") is not None:
        duration = event.get("duration_ms")
        if not isinstance(duration, int) or duration < int(filters["min_duration_ms"]):
            return False
    tool_name = filters.get("tool_name")
    if tool_name:
        tool = event.get("tool") if isinstance(event.get("tool"), dict) else {}
        if tool.get("name") != tool_name:
            return False
    return True


def query_index(index_path: Path, filters: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    where: list[str] = []
    params: list[Any] = []
    if filters.get("event_type"):
        where.append("event_type = ?")
        params.append(filters["event_type"])
    if filters.get("status"):
        where.append("status = ?")
        params.append(filters["status"])
    if filters.get("failed"):
        where.append("status IN ('error', 'blocked')")
    if filters.get("run_id"):
        where.append("run_id = ?")
        params.append(filters["run_id"])
    if filters.get("session_id"):
        where.append("session_id = ?")
        params.append(filters["session_id"])
    if filters.get("tool_name"):
        where.append("tool_name = ?")
        params.append(filters["tool_name"])
    if filters.get("min_duration_ms") is not None:
        where.append("duration_ms >= ?")
        params.append(int(filters["min_duration_ms"]))

    sql = """
        SELECT jsonl_path, jsonl_offset, jsonl_bytes
        FROM events
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    events: list[dict[str, Any]] = []
    connection = sqlite3.connect(str(index_path))
    try:
        for jsonl_path, offset, byte_count in connection.execute(sql, params):
            event = read_jsonl_event(Path(jsonl_path), int(offset), int(byte_count))
            if event is not None:
                events.append(event)
    finally:
        connection.close()
    return events


def read_jsonl_event(path: Path, offset: int, byte_count: int) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(byte_count)
        event = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return event if isinstance(event, dict) else None


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    tool = event.get("tool") if isinstance(event.get("tool"), dict) else {}
    return {
        "timestamp": event.get("timestamp"),
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "phase": event.get("phase"),
        "status": event.get("status"),
        "tool_name": tool.get("name"),
        "duration_ms": event.get("duration_ms"),
        "session_id": event.get("session_id"),
        "run_id": event.get("run_id"),
        "trace_id": event.get("trace_id"),
        "span_id": event.get("span_id"),
    }


def render_events(events: list[dict[str, Any]]) -> str:
    lines = []
    for event in events:
        item = compact_event(event)
        parts = [
            str(item["timestamp"] or ""),
            str(item["event_type"] or "event"),
            str(item["phase"] or "instant"),
            str(item["status"] or "ok"),
        ]
        if item.get("duration_ms") is not None:
            parts.append(f"{item['duration_ms']}ms")
        if item.get("tool_name"):
            parts.append(f"tool={item['tool_name']}")
        if item.get("run_id"):
            parts.append(f"run={item['run_id']}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query Hermes Flight Recorder events.")
    parser.add_argument("path", type=Path, help="Flight Recorder JSONL path.")
    parser.add_argument("--index", type=Path, help="SQLite index path. Defaults to <jsonl>.sqlite3 when rebuilding.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the SQLite index before querying.")
    parser.add_argument("--event-type")
    parser.add_argument("--status")
    parser.add_argument("--tool")
    parser.add_argument("--run")
    parser.add_argument("--session")
    parser.add_argument("--min-duration-ms", type=int)
    parser.add_argument("--failed", action="store_true", help="Only error or blocked events.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
