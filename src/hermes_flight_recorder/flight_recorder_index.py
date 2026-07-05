"""SQLite index for Hermes Flight Recorder JSONL files."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


INDEX_SCHEMA_VERSION = 1


class FlightRecorderIndex:
    """Small local index over canonical JSONL events.

    The JSONL file remains the source of truth. SQLite stores only sortable and
    filterable metadata plus JSONL byte offsets, so it can be rebuilt at any
    time from the raw log.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _ensure_schema(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    event_type TEXT NOT NULL,
                    phase TEXT,
                    status TEXT,
                    session_id TEXT,
                    turn_id TEXT,
                    run_id TEXT,
                    task_id TEXT,
                    trace_id TEXT,
                    span_id TEXT,
                    parent_span_id TEXT,
                    tool_name TEXT,
                    duration_ms INTEGER,
                    jsonl_path TEXT NOT NULL,
                    jsonl_offset INTEGER NOT NULL,
                    jsonl_bytes INTEGER NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_type_status ON events(event_type, status)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_tool ON events(tool_name)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_duration ON events(duration_ms)")
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(INDEX_SCHEMA_VERSION),),
            )
            connection.commit()
        finally:
            connection.close()

    def _execute_write(self, sql: str, params: tuple[Any, ...]) -> None:
        """Run a single write, self-healing the schema if the DB vanished.

        The SQLite file is local and disposable; it can be removed out from under
        a long-lived recorder by disk cleanup, JSONL rotation/retention, or a
        canary clear step. A fresh `sqlite3.connect` then recreates an *empty*
        file with no tables, so the next write would fail "no such table" and the
        recorder would silently stop indexing for the rest of its lifetime. We
        re-ensure the schema and retry once so indexing self-heals.
        """
        try:
            self._run_write(sql, params)
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc).lower():
                raise
            self._ensure_schema()
            self._run_write(sql, params)

    def _run_write(self, sql: str, params: tuple[Any, ...]) -> None:
        connection = self._connect()
        try:
            connection.execute(sql, params)
            connection.commit()
        finally:
            connection.close()

    def index_event(self, event: dict[str, Any], *, jsonl_path: str, jsonl_offset: int, jsonl_bytes: int) -> None:
        tool = event.get("tool") if isinstance(event.get("tool"), dict) else {}
        self._execute_write(
            """
            INSERT OR REPLACE INTO events(
                event_id, timestamp, event_type, phase, status, session_id,
                turn_id, run_id, task_id, trace_id, span_id, parent_span_id,
                tool_name, duration_ms, jsonl_path, jsonl_offset, jsonl_bytes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("event_id"),
                event.get("timestamp"),
                event.get("event_type"),
                event.get("phase"),
                event.get("status"),
                event.get("session_id"),
                event.get("turn_id"),
                event.get("run_id"),
                event.get("task_id"),
                event.get("trace_id"),
                event.get("span_id"),
                event.get("parent_span_id"),
                tool.get("name"),
                event.get("duration_ms"),
                jsonl_path,
                int(jsonl_offset),
                int(jsonl_bytes),
            ),
        )

    def update_jsonl_path(self, old_path: str | Path, new_path: str | Path) -> None:
        self._execute_write(
            "UPDATE events SET jsonl_path = ? WHERE jsonl_path = ?",
            (str(new_path), str(old_path)),
        )

    def delete_jsonl_path(self, jsonl_path: str | Path) -> None:
        self._execute_write("DELETE FROM events WHERE jsonl_path = ?", (str(jsonl_path),))


def default_index_path(jsonl_path: str | Path) -> Path:
    path = Path(jsonl_path)
    suffix = path.suffix + ".sqlite3" if path.suffix else ".sqlite3"
    return path.with_suffix(suffix)


def rebuild_index(jsonl_path: str | Path, index_path: str | Path | None = None) -> dict[str, Any]:
    jsonl = Path(jsonl_path)
    index = FlightRecorderIndex(index_path or default_index_path(jsonl))
    indexed = 0
    skipped = 0
    offset = 0
    with jsonl.open("rb") as handle:
        for raw_line in handle:
            line_offset = offset
            offset += len(raw_line)
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line.decode("utf-8"))
            except Exception:
                skipped += 1
                continue
            if not isinstance(event, dict):
                skipped += 1
                continue
            index.index_event(
                event,
                jsonl_path=str(jsonl),
                jsonl_offset=line_offset,
                jsonl_bytes=len(raw_line),
            )
            indexed += 1
    return {
        "jsonl_path": str(jsonl),
        "index_path": str(index.path),
        "indexed": indexed,
        "skipped": skipped,
    }
