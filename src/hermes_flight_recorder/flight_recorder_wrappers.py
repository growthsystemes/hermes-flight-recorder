"""Runtime wrappers for Flight Recorder side-effect capture.

These helpers are intentionally small and dependency-free. They perform the
side effect, record a privacy-safe event, and then return or re-raise exactly as
the wrapped operation would.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .flight_recorder import HermesFlightRecorder, utc_now_iso


@dataclass(frozen=True)
class FlightRecorderContext:
    run_id: str
    session_id: str | None = None
    turn_id: str | None = None
    task_id: str | None = None
    source: str = "backend"
    trace_id: str | None = None
    parent_span_id: str | None = None


def _context(context: FlightRecorderContext | dict[str, Any] | None, default_run_id: str) -> dict[str, Any]:
    if isinstance(context, FlightRecorderContext):
        value = {
            "run_id": context.run_id,
            "session_id": context.session_id,
            "turn_id": context.turn_id,
            "task_id": context.task_id,
            "source": context.source,
            "trace_id": context.trace_id,
            "parent_span_id": context.parent_span_id,
        }
    elif isinstance(context, dict):
        value = dict(context)
    else:
        value = {}
    value["run_id"] = str(value.get("run_id") or value.get("task_id") or default_run_id)
    value["session_id"] = value.get("session_id") or value["run_id"]
    value["turn_id"] = value.get("turn_id") or value["session_id"]
    value["task_id"] = value.get("task_id") or value["run_id"]
    value["source"] = str(value.get("source") or "backend")
    return value


def _record_side_effect_event(
    recorder: HermesFlightRecorder,
    *,
    event_type: str,
    effect: dict[str, Any],
    status: str,
    started_at: float,
    start_ts: str,
    context: FlightRecorderContext | dict[str, Any] | None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = _context(context, default_run_id=f"{event_type}-{int(started_at * 1_000_000)}")
    run_id = str(ctx["run_id"])
    end_ts = utc_now_iso()
    return recorder.record(
        event_type=event_type,
        phase="end",
        trace_id=str(ctx.get("trace_id") or recorder.trace_id(run_id)),
        span_id=recorder.span_id(run_id, event_type, int(started_at * 1_000_000)),
        parent_span_id=ctx.get("parent_span_id"),
        session_id=str(ctx["session_id"]) if ctx.get("session_id") is not None else None,
        turn_id=str(ctx["turn_id"]) if ctx.get("turn_id") is not None else None,
        run_id=run_id,
        task_id=str(ctx["task_id"]) if ctx.get("task_id") is not None else None,
        source=str(ctx["source"]),
        status=status,
        actor="runtime",
        start_ts=start_ts,
        end_ts=end_ts,
        duration_ms=int((time.time() - started_at) * 1000),
        side_effects=[effect],
        attributes=attributes,
    )


def process_exec(
    recorder: HermesFlightRecorder,
    command: Sequence[str] | str,
    *,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    check: bool = False,
    context: FlightRecorderContext | dict[str, Any] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a process and record a `process.exec` side-effect event.

    Output is captured as bytes so metadata mode can hash stdout/stderr without
    leaking decoded content. The raw command, cwd, stdout, and stderr are never
    written by the recorder in metadata mode.
    """
    started_at = time.time()
    start_ts = utc_now_iso()
    cwd_text = str(cwd) if cwd is not None else None
    try:
        result = subprocess.run(
            command,
            cwd=cwd_text,
            timeout=timeout,
            check=False,
            capture_output=True,
        )
    except subprocess.TimeoutExpired as exc:
        effect = recorder.side_effect_payload(
            "process.exec",
            target=cwd_text,
            command=command,
            stdout=exc.stdout,
            stderr=exc.stderr,
            metadata={
                "operation": "exec",
                "timed_out": True,
                "timeout_seconds": timeout,
            },
        )
        _record_side_effect_event(
            recorder,
            event_type="process.exec",
            effect=effect,
            status="error",
            started_at=started_at,
            start_ts=start_ts,
            context=context,
            attributes={"error_type": "TimeoutExpired"},
        )
        raise

    status = "ok" if result.returncode == 0 else "error"
    effect = recorder.side_effect_payload(
        "process.exec",
        target=cwd_text,
        command=command,
        stdout=result.stdout,
        stderr=result.stderr,
        metadata={
            "operation": "exec",
            "exit_code": result.returncode,
            "stdout_bytes": len(result.stdout or b""),
            "stderr_bytes": len(result.stderr or b""),
        },
    )
    _record_side_effect_event(
        recorder,
        event_type="process.exec",
        effect=effect,
        status=status,
        started_at=started_at,
        start_ts=start_ts,
        context=context,
        attributes={"exit_code": result.returncode},
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def file_write(
    recorder: HermesFlightRecorder,
    path: str | Path,
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    context: FlightRecorderContext | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a file and record a `file.write` side-effect event."""
    target = Path(path)
    started_at = time.time()
    start_ts = utc_now_iso()
    before: bytes | None = None
    existed_before = target.exists()
    if existed_before and target.is_file():
        before = target.read_bytes()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            after_bytes = data
        else:
            after_bytes = data.encode(encoding)
        target.write_bytes(after_bytes)
    except Exception as exc:
        effect = recorder.side_effect_payload(
            "file.write",
            target=str(target),
            before=before,
            metadata={"operation": "write", "existed_before": existed_before},
        )
        _record_side_effect_event(
            recorder,
            event_type="file.write",
            effect=effect,
            status="error",
            started_at=started_at,
            start_ts=start_ts,
            context=context,
            attributes={"error_type": type(exc).__name__},
        )
        raise

    effect = recorder.side_effect_payload(
        "file.write",
        target=str(target),
        before=before,
        after=after_bytes,
        metadata={
            "operation": "write",
            "existed_before": existed_before,
            "bytes_before": len(before or b""),
            "bytes_after": len(after_bytes),
        },
    )
    return _record_side_effect_event(
        recorder,
        event_type="file.write",
        effect=effect,
        status="ok",
        started_at=started_at,
        start_ts=start_ts,
        context=context,
        attributes={"bytes_after": len(after_bytes)},
    )


def file_read(
    recorder: HermesFlightRecorder,
    path: str | Path,
    *,
    encoding: str | None = "utf-8",
    context: FlightRecorderContext | dict[str, Any] | None = None,
) -> str | bytes:
    """Read a file and record a `file.read` side-effect event."""
    target = Path(path)
    started_at = time.time()
    start_ts = utc_now_iso()
    try:
        content = target.read_bytes()
    except Exception as exc:
        effect = recorder.side_effect_payload(
            "file.read",
            target=str(target),
            metadata={"operation": "read"},
        )
        _record_side_effect_event(
            recorder,
            event_type="file.read",
            effect=effect,
            status="error",
            started_at=started_at,
            start_ts=start_ts,
            context=context,
            attributes={"error_type": type(exc).__name__},
        )
        raise

    effect = recorder.side_effect_payload(
        "file.read",
        target=str(target),
        after=content,
        metadata={
            "operation": "read",
            "bytes": len(content),
        },
    )
    _record_side_effect_event(
        recorder,
        event_type="file.read",
        effect=effect,
        status="ok",
        started_at=started_at,
        start_ts=start_ts,
        context=context,
        attributes={"bytes": len(content)},
    )
    if encoding is None:
        return content
    return content.decode(encoding)
