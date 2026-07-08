"""Local-first Hermes Flight Recorder.

This module is intentionally dependency-light and fail-open. It records
observable Hermes events to append-oriented JSONL without affecting agent traffic
when the recorder is disabled or when local storage fails.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import inspect
import json
import logging
import os
import queue
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

_LOGGER = logging.getLogger("hermes.flight_recorder")
F = TypeVar("F", bound=Callable[..., Any])


# 0.3.0 (2026-06-25): additive governance/interoperability pass - tool catalog
# snapshot/diff events, deterministic eval.score events, W3C trace context
# helpers, and an OpenInference OTLP projection.
SCHEMA_VERSION = "0.3.0"
# RECORDER_VERSION tracks the public PyPI package release, which restarts its
# own numbering at 0.1.0 for the first public release (2026-07-05) regardless
# of internal pre-publication iteration (see CHANGELOG.md "Internal iteration
# history", which reached 0.3.1 before the public decision). The embedded
# Hermes runtime copy under deployments/docker/hermes-agent/src/ keeps its own
# internal version independently of this public package.
RECORDER_VERSION = "0.1.3"
REDACTION_POLICY_VERSION = "0.1.0"  # unchanged: no redaction-behaviour change
SEMCONV_VERSION = "otel-genai-compat-2026-06-20"
OTEL_MAPPING_VERSION = "0.2.0"
OPENINFERENCE_MAPPING_VERSION = "0.1.0"
# OpenInference is projected on OTLP spans, not stored as a canonical JSONL
# envelope field.
HASH_SCOPE = "canonical_redacted_event_v1"
DEFAULT_DEV_HASH_SECRET = "hermes-flight-recorder-dev-fixture-secret-v1"
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "key",
    "password",
    "secret",
    "signature",
    "token",
)
# Numeric usage COUNTS match "token" by substring but are not sensitive. Allow them
# through redaction when (and only when) the value is numeric (fail-closed: an
# allowlisted key carrying a string still gets redacted).
NON_SENSITIVE_COUNT_KEYS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "max_tokens"}
)
SECRET_PATTERNS = (
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;}]+"),
)


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_hmac(value: Any, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), canonical_json(value).encode("utf-8"), hashlib.sha256).hexdigest()
    return "hmac-sha256:" + digest


def stable_hex_id(*parts: Any, length: int) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def trace_id_for(*parts: Any) -> str:
    return stable_hex_id(*parts, length=32)


def span_id_for(*parts: Any) -> str:
    return stable_hex_id(*parts, length=16)


def traceparent_for(trace_id: str, span_id: str, *, sampled: bool = True) -> str:
    """Build a W3C traceparent value for an existing Hermes span."""
    flags = "01" if sampled else "00"
    return f"00-{trace_id}-{span_id}-{flags}"


def trace_context_payload(
    trace_id: str,
    span_id: str,
    *,
    tracestate: str | None = None,
    baggage: str | None = None,
) -> dict[str, str]:
    """Return privacy-safe W3C context fields for JSON-RPC _meta propagation."""
    payload = {"traceparent": traceparent_for(trace_id, span_id)}
    if tracestate:
        payload["tracestate"] = str(tracestate)
    if baggage:
        payload["baggage"] = str(baggage)
    return payload


def redact_text(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.lower().replace("-", "_")
            if normalized in NON_SENSITIVE_COUNT_KEYS and isinstance(item, (int, float)) and not isinstance(item, bool):
                redacted[key_text] = item
            elif any(part in normalized for part in SENSITIVE_KEY_PARTS):
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = redact_value(item)
        return redacted
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def preview_value(value: Any, max_chars: int) -> str:
    text = canonical_json(redact_value(value))
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def normalize_capture_mode(value: str) -> str:
    mode = (value or "metadata").strip().lower()
    if mode not in {"metadata", "preview", "full", "forensic"}:
        return "metadata"
    return mode


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else value


@dataclass(frozen=True)
class FlightRecorderSettings:
    enabled: bool = False
    path: str = "/tmp/hermes-flight-recorder/events.jsonl"
    capture_mode: str = "metadata"
    preview_chars: int = 500
    fsync: bool = False
    otlp_enabled: bool = False
    otlp_endpoint: str = ""
    otlp_headers: str = ""
    # Generic default for the public package - callers embedding this in a
    # specific agent (e.g. Hermes) should set FLIGHT_RECORDER_OTLP_SERVICE_NAME
    # or pass otlp_service_name explicitly rather than rely on this default.
    otlp_service_name: str = "hermes-flight-recorder"
    otlp_timeout_seconds: float = 2.0
    otlp_max_buffer: int = 500
    otlp_flush_interval_seconds: float = 5.0
    otlp_include_previews: bool = False
    hash_strategy: str = "hmac"
    hash_secret_file: str = ""
    hash_secret_env: str = ""
    require_strong_secret: bool = False
    schema_validation_enabled: bool = True
    schema_validation_strict: bool = False
    index_enabled: bool = False
    index_path: str = ""
    sample_rate: float = 1.0
    max_event_bytes: int = 0
    async_writes_enabled: bool = False
    write_queue_max_events: int = 1000
    rotate_bytes: int = 0
    retention_files: int = 0


class FlightRecorderSpan:
    """Synchronous context manager that emits start/end events for one span."""

    def __init__(
        self,
        recorder: "HermesFlightRecorder",
        event_type: str,
        *,
        run_id: str = "default-run",
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        task_id: str | None = None,
        source: str = "backend",
        actor: str = "agent",
        status: str = "ok",
        model: dict[str, Any] | None = None,
        tool: dict[str, Any] | None = None,
        tool_name: str | None = None,
        tool_category: str = "custom",
        arguments: Any | None = None,
        result: Any | None = None,
        error: Any | None = None,
        side_effects: list[dict[str, Any]] | None = None,
        runtime: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ):
        self.recorder = recorder
        self.event_type = event_type
        self.run_id = run_id
        self.trace_id = trace_id or recorder.trace_id(run_id)
        self.span_id = span_id or uuid.uuid4().hex[:16]
        self.parent_span_id = parent_span_id
        self.session_id = session_id or run_id
        self.turn_id = turn_id
        self.task_id = task_id or run_id
        self.source = source
        self.actor = actor
        self.status = status
        self.model = model
        self.tool = tool
        self.tool_name = tool_name
        self.tool_category = tool_category
        self.arguments = arguments
        self.result = result
        self.error = error
        self.side_effects = side_effects
        self.runtime = runtime
        self.attributes = dict(attributes or {})
        self.start_ts: str | None = None
        self.end_ts: str | None = None
        self._started_monotonic: float | None = None
        self.start_event: dict[str, Any] = {}
        self.end_event: dict[str, Any] = {}

    def __enter__(self) -> "FlightRecorderSpan":
        self.start_ts = utc_now_iso()
        self._started_monotonic = time.monotonic()
        self.start_event = self.recorder.record(
            **self._event_kwargs(phase="start", start_ts=self.start_ts, status=self.status)
        )
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        self.end(exc)
        return False

    def set_result(self, result: Any) -> None:
        self.result = result

    def set_error(self, error: Any) -> None:
        self.error = error
        self.status = "error"

    def set_model(self, model: dict[str, Any]) -> None:
        self.model = model

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def end(self, exc: BaseException | None = None) -> dict[str, Any]:
        if exc is not None:
            self.status = "error"
            if self.error is None:
                self.error = {"type": type(exc).__name__, "message": str(exc)}
        self.end_ts = utc_now_iso()
        duration_ms = None
        if self._started_monotonic is not None:
            duration_ms = max(0, int((time.monotonic() - self._started_monotonic) * 1000))
        self.end_event = self.recorder.record(
            **self._event_kwargs(
                phase="end",
                start_ts=self.start_ts,
                end_ts=self.end_ts,
                duration_ms=duration_ms,
                status=self.status,
            )
        )
        return self.end_event

    def _event_kwargs(self, **overrides: Any) -> dict[str, Any]:
        tool = self.tool
        if tool is None and self.tool_name:
            tool = self.recorder.tool_payload(
                self.tool_name,
                arguments=self.arguments,
                result=self.result,
                error=self.error,
                category=self.tool_category,
            )
        kwargs: dict[str, Any] = {
            "event_type": self.event_type,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "source": self.source,
            "status": self.status,
            "actor": self.actor,
            "model": self.model,
            "tool": tool,
            "side_effects": self.side_effects,
            "runtime": self.runtime,
            "attributes": self.attributes or None,
        }
        kwargs.update(overrides)
        return kwargs


class AsyncFlightRecorderSpan:
    """Async context manager equivalent of :class:`FlightRecorderSpan`."""

    def __init__(self, recorder: "HermesFlightRecorder", event_type: str, **kwargs: Any):
        self._span = FlightRecorderSpan(recorder, event_type, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._span, name)

    async def __aenter__(self) -> FlightRecorderSpan:
        self._span.start_ts = utc_now_iso()
        self._span._started_monotonic = time.monotonic()
        self._span.start_event = await self._span.recorder.arecord(
            **self._span._event_kwargs(phase="start", start_ts=self._span.start_ts, status=self._span.status)
        )
        return self._span

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        if exc is not None:
            self._span.status = "error"
            if self._span.error is None:
                self._span.error = {"type": type(exc).__name__, "message": str(exc)}
        self._span.end_ts = utc_now_iso()
        duration_ms = None
        if self._span._started_monotonic is not None:
            duration_ms = max(0, int((time.monotonic() - self._span._started_monotonic) * 1000))
        self._span.end_event = await self._span.recorder.arecord(
            **self._span._event_kwargs(
                phase="end",
                start_ts=self._span.start_ts,
                end_ts=self._span.end_ts,
                duration_ms=duration_ms,
                status=self._span.status,
            )
        )
        return False


class HermesFlightRecorder:
    """Append-oriented JSONL recorder for observable Hermes events."""

    def __init__(self, settings: FlightRecorderSettings):
        # Resolve the hashing secret first: if a strong secret is required but
        # only a dev/weak fallback is available, the recorder fails CLOSED on the
        # config (refuses to enable) rather than hashing with a public key. This
        # is a config-time guard, never a fail-closed on live agent traffic.
        self.hash_strategy = normalize_hash_strategy(settings.hash_strategy)
        self._hash_secret, self._hash_secret_source = resolve_hash_secret(settings)
        self._strong_secret_required = bool(settings.require_strong_secret)
        self._weak_secret_blocked = bool(
            settings.enabled
            and self._strong_secret_required
            and self._hash_secret_source in {"default-dev", "weak-env-fallback"}
        )
        if self._weak_secret_blocked:
            _LOGGER.warning(
                "flight recorder disabled: require_strong_secret set but hash_secret_source=%s",
                self._hash_secret_source,
            )
            settings = replace(settings, enabled=False)
        self.settings = settings
        self.capture_mode = normalize_capture_mode(settings.capture_mode)
        self.preview_chars = max(64, int(settings.preview_chars or 500))
        self.otlp_max_buffer = max(1, int(settings.otlp_max_buffer or 500))
        self.otlp_flush_interval_seconds = max(0.1, float(settings.otlp_flush_interval_seconds or 5.0))
        self.sample_rate = min(1.0, max(0.0, float(settings.sample_rate if settings.sample_rate is not None else 1.0)))
        self.max_event_bytes = max(0, int(settings.max_event_bytes or 0))
        self.write_queue_max_events = max(1, int(settings.write_queue_max_events or 1000))
        self.rotate_bytes = max(0, int(settings.rotate_bytes or 0))
        self.retention_files = max(0, int(settings.retention_files or 0))
        self._last_hash: str | None = None
        self._write_failed = False
        self._otlp_failed = False
        self._otlp_dropped = 0
        self._otlp_buffer: list[dict[str, Any]] = []
        self._schema_invalid_events = 0
        self._last_schema_errors: list[str] = []
        self._sampled_events = 0
        self._oversize_events = 0
        self._index_failed = False
        self._rotations = 0
        self._retention_deleted_files = 0
        self._retention_failed = False
        self._write_queue_dropped_events = 0
        self._record_failed = False
        self._record_dropped_events = 0
        # Serializes the local append/rotate/index path so concurrent callers
        # (multi-threaded sync writes, or the async writer thread) cannot
        # interleave or tear JSONL lines. Uncontended on the single-writer
        # async path and on the single-event-loop live request path.
        self._write_lock = threading.Lock()
        self._index = self._create_index()
        self._write_queue: queue.Queue[tuple[dict[str, Any], bytes]] | None = None
        self._write_thread: threading.Thread | None = None
        if self.settings.enabled and self.settings.async_writes_enabled:
            self._write_queue = queue.Queue(maxsize=self.write_queue_max_events)
            self._write_thread = threading.Thread(target=self._write_worker, name="hermes-flight-recorder-writer", daemon=True)
            self._write_thread.start()

    @classmethod
    def from_config(cls, config: Any) -> "HermesFlightRecorder":
        return cls(
            FlightRecorderSettings(
                enabled=bool(getattr(config, "flight_recorder_enabled", False)),
                path=str(getattr(config, "flight_recorder_path", "/tmp/hermes-flight-recorder/events.jsonl")),
                capture_mode=str(getattr(config, "flight_recorder_capture_mode", "metadata")),
                preview_chars=int(getattr(config, "flight_recorder_preview_chars", 500)),
                fsync=bool(getattr(config, "flight_recorder_fsync", False)),
                otlp_enabled=bool(getattr(config, "flight_recorder_otlp_enabled", False)),
                otlp_endpoint=str(getattr(config, "flight_recorder_otlp_endpoint", "")),
                otlp_headers=str(getattr(config, "flight_recorder_otlp_headers", "")),
                otlp_service_name=str(getattr(config, "flight_recorder_otlp_service_name", "hermes-flight-recorder")),
                otlp_timeout_seconds=float(getattr(config, "flight_recorder_otlp_timeout_seconds", 2.0)),
                otlp_max_buffer=int(getattr(config, "flight_recorder_otlp_max_buffer", 500)),
                otlp_flush_interval_seconds=float(getattr(config, "flight_recorder_otlp_flush_interval_seconds", 5.0)),
                otlp_include_previews=bool(getattr(config, "flight_recorder_otlp_include_previews", False)),
                hash_strategy=str(getattr(config, "flight_recorder_hash_strategy", "hmac")),
                hash_secret_file=str(getattr(config, "flight_recorder_hash_secret_file", "")),
                hash_secret_env=str(getattr(config, "flight_recorder_hash_secret_env", "")),
                require_strong_secret=bool(getattr(config, "flight_recorder_require_strong_secret", False)),
                schema_validation_enabled=bool(getattr(config, "flight_recorder_schema_validation_enabled", True)),
                schema_validation_strict=bool(getattr(config, "flight_recorder_schema_validation_strict", False)),
                index_enabled=bool(getattr(config, "flight_recorder_index_enabled", False)),
                index_path=str(getattr(config, "flight_recorder_index_path", "")),
                sample_rate=float(getattr(config, "flight_recorder_sample_rate", 1.0)),
                max_event_bytes=int(getattr(config, "flight_recorder_max_event_bytes", 0)),
                async_writes_enabled=bool(getattr(config, "flight_recorder_async_writes_enabled", False)),
                write_queue_max_events=int(getattr(config, "flight_recorder_write_queue_max_events", 1000)),
                rotate_bytes=int(getattr(config, "flight_recorder_rotate_bytes", 0)),
                retention_files=int(getattr(config, "flight_recorder_retention_files", 0)),
            )
        )

    @classmethod
    def from_env(cls, prefix: str = "FLIGHT_RECORDER_") -> "HermesFlightRecorder":
        """Build a recorder directly from FLIGHT_RECORDER_* environment variables."""
        key = lambda suffix: f"{prefix}{suffix}"  # noqa: E731 - compact local mapper
        return cls(
            FlightRecorderSettings(
                enabled=env_bool(key("ENABLED"), False),
                path=env_str(key("PATH"), "/tmp/hermes-flight-recorder/events.jsonl"),
                capture_mode=env_str(key("CAPTURE_MODE"), "metadata"),
                preview_chars=env_int(key("PREVIEW_CHARS"), 500),
                fsync=env_bool(key("FSYNC"), False),
                otlp_enabled=env_bool(key("OTLP_ENABLED"), False),
                otlp_endpoint=env_str(key("OTLP_ENDPOINT"), ""),
                otlp_headers=env_str(key("OTLP_HEADERS"), ""),
                otlp_service_name=env_str(key("OTLP_SERVICE_NAME"), "hermes-flight-recorder"),
                otlp_timeout_seconds=env_float(key("OTLP_TIMEOUT_SECONDS"), 2.0),
                otlp_max_buffer=env_int(key("OTLP_MAX_BUFFER"), 500),
                otlp_include_previews=env_bool(key("OTLP_INCLUDE_PREVIEWS"), False),
                hash_strategy=env_str(key("HASH_STRATEGY"), "hmac"),
                hash_secret_file=env_str(key("HASH_SECRET_FILE"), ""),
                hash_secret_env=env_str(key("HASH_SECRET_ENV"), ""),
                require_strong_secret=env_bool(key("REQUIRE_STRONG_SECRET"), False),
                schema_validation_enabled=env_bool(key("SCHEMA_VALIDATION_ENABLED"), True),
                schema_validation_strict=env_bool(key("SCHEMA_VALIDATION_STRICT"), False),
                index_enabled=env_bool(key("INDEX_ENABLED"), False),
                index_path=env_str(key("INDEX_PATH"), ""),
                sample_rate=env_float(key("SAMPLE_RATE"), 1.0),
                max_event_bytes=env_int(key("MAX_EVENT_BYTES"), 0),
                async_writes_enabled=env_bool(key("ASYNC_WRITES_ENABLED"), False),
                write_queue_max_events=env_int(key("WRITE_QUEUE_MAX_EVENTS"), 1000),
                rotate_bytes=env_int(key("ROTATE_BYTES"), 0),
                retention_files=env_int(key("RETENTION_FILES"), 0),
            )
        )

    def status(self) -> dict[str, Any]:
        otlp_active = self.settings.enabled and self.settings.otlp_enabled
        return {
            "enabled": self.settings.enabled,
            "path": self.settings.path if self.settings.enabled else None,
            "capture_mode": self.capture_mode,
            "write_failed": self._write_failed,
            "otlp_enabled": otlp_active,
            "otlp_endpoint_configured": bool(self.settings.otlp_endpoint),
            "otlp_buffered_events": len(self._otlp_buffer),
            "otlp_dropped_events": self._otlp_dropped,
            "otlp_max_buffer": self.otlp_max_buffer,
            "otlp_flush_interval_seconds": self.otlp_flush_interval_seconds,
            "otlp_failed": self._otlp_failed,
            "hash_strategy": self.hash_strategy,
            "hash_secret_source": self._hash_secret_source,
            "strong_secret_required": self._strong_secret_required,
            "weak_secret_blocked": self._weak_secret_blocked,
            "record_failed": self._record_failed,
            "record_dropped_events": self._record_dropped_events,
            "schema_validation_enabled": self.settings.schema_validation_enabled,
            "schema_validation_strict": self.settings.schema_validation_strict,
            "schema_invalid_events": self._schema_invalid_events,
            "last_schema_errors": self._last_schema_errors,
            "index_enabled": bool(self.settings.enabled and self.settings.index_enabled),
            "index_path": str(self._index.path) if self._index is not None and self.settings.enabled else None,
            "index_failed": self._index_failed,
            "sample_rate": self.sample_rate,
            "sampled_events": self._sampled_events,
            "max_event_bytes": self.max_event_bytes,
            "oversize_events": self._oversize_events,
            "current_file_bytes": self._current_file_bytes(),
            "async_writes_enabled": bool(self.settings.enabled and self.settings.async_writes_enabled),
            "write_queue_depth": self._write_queue.qsize() if self._write_queue is not None else 0,
            "write_queue_dropped_events": self._write_queue_dropped_events,
            "write_queue_max_events": self.write_queue_max_events,
            "rotate_bytes": self.rotate_bytes,
            "rotations": self._rotations,
            "rotated_files": self._rotated_file_count(),
            "retention_files": self.retention_files,
            "retention_deleted_files": self._retention_deleted_files,
            "retention_failed": self._retention_failed,
        }

    def trace_id(self, run_id: str) -> str:
        return trace_id_for("hermes", run_id)

    def span_id(self, run_id: str, *parts: Any) -> str:
        return span_id_for("hermes", run_id, *parts)

    def digest_value(self, value: Any, *, sensitive: bool = True) -> str:
        if sensitive and self.hash_strategy == "hmac":
            return stable_hmac(value, self._hash_secret)
        return stable_hash(value)

    def digest_field_name(self, prefix: str, *, sensitive: bool = True) -> str:
        if sensitive and self.hash_strategy == "hmac":
            return f"{prefix}_hmac"
        return f"{prefix}_sha256"

    def digest_payload_value(self, value: Any, *, sensitive: bool = True) -> str:
        return self.digest_value(value, sensitive=sensitive)

    def _put_digest(self, payload: dict[str, Any], key: str, value: Any, *, sensitive: bool = True) -> None:
        payload[self.digest_field_name(key, sensitive=sensitive)] = self.digest_value(value, sensitive=sensitive)

    def tool_payload(
        self,
        tool_name: str,
        arguments: Any | None = None,
        result: Any | None = None,
        error: Any | None = None,
        *,
        category: str = "custom",
        result_count: int | None = None,
        confidence: float | None = None,
        status_code: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": tool_name,
            "category": category,
        }
        # Low-cardinality result-quality metadata (no raw content): how many rows
        # the tool returned, an optional confidence, and the transport status code.
        if isinstance(result_count, int):
            payload["result_count"] = result_count
        if isinstance(confidence, (int, float)):
            payload["confidence"] = float(confidence)
        if isinstance(status_code, int):
            payload["status_code"] = int(status_code)
        if arguments is not None:
            self._put_digest(payload, "args", arguments)
            if self.capture_mode in {"preview", "full", "forensic"}:
                payload["args_preview"] = preview_value(arguments, self.preview_chars)
            if self.capture_mode in {"full", "forensic"}:
                payload["args_full"] = redact_value(arguments)
        if result is not None:
            self._put_digest(payload, "result", result)
            if self.capture_mode in {"preview", "full", "forensic"}:
                payload["result_preview"] = preview_value(result, self.preview_chars)
            if self.capture_mode in {"full", "forensic"}:
                payload["result_full"] = redact_value(result)
        if error is not None:
            payload["error_type"] = type(error).__name__ if not isinstance(error, str) else "error"
            self._put_digest(payload, "error", str(error))
            if self.capture_mode in {"preview", "full", "forensic"}:
                payload["error_preview"] = preview_value(str(error), self.preview_chars)
            if self.capture_mode in {"full", "forensic"}:
                payload["error_full"] = redact_value(str(error))
        return payload

    def model_payload(
        self,
        provider: str | None = None,
        name: str | None = None,
        usage: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        *,
        cost_usd: float | None = None,
        reasoning_tokens: int | None = None,
        request_model: str | None = None,
        response_model: str | None = None,
        prompt_hmac: str | None = None,
        response_hmac: str | None = None,
    ) -> dict[str, Any]:
        usage = usage if isinstance(usage, dict) else {}
        parameters = parameters if isinstance(parameters, dict) else {}
        payload: dict[str, Any] = {
            "provider": provider or "unknown",
            "name": name or parameters.get("model") or "unknown",
        }
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("totalInputTokens")
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("totalOutputTokens")
        total_tokens = usage.get("total_tokens") or usage.get("totalTokens")
        if isinstance(input_tokens, int):
            payload["input_tokens"] = input_tokens
        if isinstance(output_tokens, int):
            payload["output_tokens"] = output_tokens
        if isinstance(total_tokens, int):
            payload["total_tokens"] = total_tokens
        reasoning = reasoning_tokens if isinstance(reasoning_tokens, int) else usage.get("reasoning_tokens")
        if isinstance(reasoning, int):
            payload["reasoning_tokens"] = reasoning
        if request_model:
            payload["request_model"] = str(request_model)
        if response_model:
            payload["response_model"] = str(response_model)
        if isinstance(cost_usd, (int, float)):
            payload["cost_usd"] = round(float(cost_usd), 6)
        if prompt_hmac:
            payload["prompt_hmac"] = str(prompt_hmac)
        if response_hmac:
            payload["response_hmac"] = str(response_hmac)
        for key in ("temperature", "max_tokens", "stream"):
            if key in parameters:
                payload[key] = parameters[key]
        return payload

    def side_effect_payload(
        self,
        effect_type: str,
        *,
        target: Any | None = None,
        command: Any | None = None,
        before: Any | None = None,
        after: Any | None = None,
        diff: Any | None = None,
        stdout: Any | None = None,
        stderr: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": effect_type}
        for key, value in (
            ("target", target),
            ("command", command),
            ("before", before),
            ("after", after),
            ("diff", diff),
            ("stdout", stdout),
            ("stderr", stderr),
        ):
            if value is None:
                continue
            self._put_digest(payload, key, value)
            if self.capture_mode in {"preview", "full", "forensic"}:
                payload[f"{key}_preview"] = preview_value(value, self.preview_chars)
            if self.capture_mode in {"full", "forensic"}:
                payload[f"{key}_full"] = redact_value(value)
        if metadata:
            payload["metadata"] = redact_value(metadata)
        return payload

    def runtime_policy_payload(
        self,
        *,
        decision: str,
        policy_type: str = "egress",
        target: Any | None = None,
        sandbox_id: Any | None = None,
        policy_decision_id: Any | None = None,
        policy_version: Any | None = None,
        secret_withheld: bool | None = None,
        correlation: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "policy_type": policy_type,
            "decision": decision,
        }
        if policy_type == "egress":
            payload["egress_decision"] = decision
        for key, value in (
            ("target", target),
            ("sandbox_id", sandbox_id),
            ("policy_version", policy_version),
        ):
            if value is None:
                continue
            self._put_digest(payload, key, value)
            if self.capture_mode in {"preview", "full", "forensic"}:
                payload[f"{key}_preview"] = preview_value(value, self.preview_chars)
            if self.capture_mode in {"full", "forensic"}:
                payload[f"{key}_full"] = redact_value(value)
        if policy_decision_id is not None:
            self._put_digest(payload, "policy_decision_id", policy_decision_id)
        if secret_withheld is not None:
            payload["secret_withheld"] = bool(secret_withheld)
        if correlation:
            payload["correlation"] = correlation
        if metadata:
            payload["metadata"] = redact_value(metadata)
        return payload

    def record(
        self,
        *,
        event_type: str,
        phase: str = "instant",
        trace_id: str,
        span_id: str,
        parent_event_id: str | None = None,
        parent_span_id: str | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        source: str = "backend",
        status: str = "ok",
        actor: str = "agent",
        start_ts: str | None = None,
        end_ts: str | None = None,
        duration_ms: int | None = None,
        model: dict[str, Any] | None = None,
        tool: dict[str, Any] | None = None,
        side_effects: list[dict[str, Any]] | None = None,
        runtime: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fail-open public entrypoint: must NEVER raise into agent traffic.

        Payload construction, redaction, hashing, and strict-mode validation can
        all raise; the write/index/otlp tail is already internally fail-open.
        We wrap the whole pipeline so a recorder bug can never break a live turn.
        """
        try:
            return self._record_impl(
                event_type=event_type,
                phase=phase,
                trace_id=trace_id,
                span_id=span_id,
                parent_event_id=parent_event_id,
                parent_span_id=parent_span_id,
                session_id=session_id,
                turn_id=turn_id,
                run_id=run_id,
                task_id=task_id,
                source=source,
                status=status,
                actor=actor,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_ms=duration_ms,
                model=model,
                tool=tool,
                side_effects=side_effects,
                runtime=runtime,
                attributes=attributes,
            )
        except Exception as exc:  # noqa: BLE001 - fail-open is the contract
            self._record_failed = True
            self._record_dropped_events += 1
            _LOGGER.debug("flight recorder suppressed record() error: %s", exc)
            return {}

    async def arecord(self, **kwargs: Any) -> dict[str, Any]:
        """Async-friendly wrapper around ``record``.

        The recorder remains dependency-free and file-backed. This helper keeps
        asyncio-native agents from blocking their event loop when they want the
        simple synchronous writer semantics.
        """
        return await asyncio.to_thread(self.record, **kwargs)

    def span(self, event_type: str, **kwargs: Any) -> FlightRecorderSpan:
        """Create a generic start/end span context manager.

        Example: ``with recorder.span("llm.call", model=...):``. The helper is
        additive sugar over ``record`` and emits canonical JSONL events.
        """
        return FlightRecorderSpan(self, event_type, **kwargs)

    def aspan(self, event_type: str, **kwargs: Any) -> AsyncFlightRecorderSpan:
        """Create an async start/end span context manager."""
        return AsyncFlightRecorderSpan(self, event_type, **kwargs)

    def trace_tool_call(
        self,
        tool_name: str | None = None,
        *,
        run_id: str = "default-run",
        capture_arguments: bool = True,
        tool_category: str = "custom",
        **span_kwargs: Any,
    ) -> Callable[[F], F]:
        """Decorate a sync or async Python function as a generic tool call.

        The decorator records function arguments/results through the same
        privacy pipeline as ``record_tool_call``. It re-raises user exceptions
        after recording the failed tool span.
        """
        def decorator(func: F) -> F:
            name = tool_name or getattr(func, "__name__", "tool")

            if inspect.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    arguments = {"args": args, "kwargs": kwargs} if capture_arguments else None
                    async with self.aspan(
                        "tool.call",
                        tool_name=name,
                        tool_category=tool_category,
                        arguments=arguments,
                        run_id=run_id,
                        **span_kwargs,
                    ) as span:
                        result = await func(*args, **kwargs)
                        span.set_result(result)
                        return result

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                arguments = {"args": args, "kwargs": kwargs} if capture_arguments else None
                with self.span(
                    "tool.call",
                    tool_name=name,
                    tool_category=tool_category,
                    arguments=arguments,
                    run_id=run_id,
                    **span_kwargs,
                ) as span:
                    result = func(*args, **kwargs)
                    span.set_result(result)
                    return result

            return wrapper  # type: ignore[return-value]

        return decorator

    def record_tool_call(
        self,
        *,
        tool_name: str,
        arguments: Any | None = None,
        result: Any | None = None,
        error: Any | None = None,
        category: str = "custom",
        status: str = "ok",
        phase: str = "end",
        run_id: str = "default-run",
        session_id: str | None = None,
        turn_id: str | None = None,
        task_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        source: str = "backend",
        actor: str = "agent",
        start_ts: str | None = None,
        end_ts: str | None = None,
        duration_ms: int | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a complete tool.call event with privacy-safe payload handling."""
        if error is not None and status == "ok":
            status = "error"
        return self.record(
            event_type="tool.call",
            phase=phase,
            trace_id=trace_id or self.trace_id(run_id),
            span_id=span_id or uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id,
            session_id=session_id or run_id,
            turn_id=turn_id,
            run_id=run_id,
            task_id=task_id or run_id,
            source=source,
            status=status,
            actor=actor,
            start_ts=start_ts,
            end_ts=end_ts,
            duration_ms=duration_ms,
            tool=self.tool_payload(tool_name, arguments=arguments, result=result, error=error, category=category),
            attributes=attributes,
        )

    def _record_impl(
        self,
        *,
        event_type: str,
        phase: str = "instant",
        trace_id: str,
        span_id: str,
        parent_event_id: str | None = None,
        parent_span_id: str | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        source: str = "backend",
        status: str = "ok",
        actor: str = "agent",
        start_ts: str | None = None,
        end_ts: str | None = None,
        duration_ms: int | None = None,
        model: dict[str, Any] | None = None,
        tool: dict[str, Any] | None = None,
        side_effects: list[dict[str, Any]] | None = None,
        runtime: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp = end_ts if phase == "end" and end_ts else start_ts or end_ts or utc_now_iso()
        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "recorder_version": RECORDER_VERSION,
            "semconv_version": SEMCONV_VERSION,
            "otel_mapping_version": OTEL_MAPPING_VERSION,
            "event_id": f"evt_{uuid.uuid4().hex}",
            "parent_event_id": parent_event_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "run_id": run_id,
            "task_id": task_id,
            "source": source,
            "event_type": event_type,
            "phase": phase,
            "timestamp": timestamp,
            "monotonic_ns": time.monotonic_ns(),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_ms": duration_ms,
            "status": status,
            "actor": actor,
            "privacy": {
                "capture_mode": self.capture_mode,
                "redaction_policy_version": REDACTION_POLICY_VERSION,
                "content_export_allowed": False,
                "hash_strategy": self.hash_strategy,
            },
        }
        if model:
            event["model"] = redact_value(model)
        if tool:
            event["tool"] = redact_value(tool)
        if side_effects:
            event["side_effects"] = redact_value(side_effects)
        if runtime:
            event["runtime"] = redact_value(runtime)
        if attributes:
            event["attributes"] = redact_value(attributes)
        keep_null_fields = {
            # parent_event_id / openinference_mapping_version intentionally NOT kept
            # when null: never populated today (causality is carried by parent_span_id;
            # OpenInference is projected on OTLP spans), so they no longer pollute
            # the canonical JSONL envelope. Both remain latent: if ever set non-null
            # they are still emitted (below).
            "parent_span_id",
            "session_id",
            "turn_id",
            "run_id",
            "task_id",
        }
        event = {key: value for key, value in event.items() if value is not None or key in keep_null_fields}

        if self.capture_mode == "forensic":
            hash_algorithm = "hmac-sha256" if self.hash_strategy == "hmac" else "sha256"
            event["integrity"] = {
                "previous_event_hash": self._last_hash,
                "hash_algorithm": hash_algorithm,
                "hash_scope": HASH_SCOPE,
            }
            event_hash = self.digest_value({k: v for k, v in event.items() if k != "integrity"})
            event["integrity"]["event_hash"] = event_hash
            self._last_hash = event_hash

        self._validate_event(event)
        if self._should_sample_out(event):
            self._sampled_events += 1
            return event
        self._write(event)
        self._buffer_otlp(event)
        return event

    def _write(self, event: dict[str, Any]) -> None:
        if not self.settings.enabled:
            return
        line = self._serialized_line(event)
        if line is None:
            return
        if self._write_queue is not None:
            try:
                self._write_queue.put_nowait((event, line))
            except queue.Full:
                self._write_queue_dropped_events += 1
            return
        self._write_line(event, line)

    def _write_line(self, event: dict[str, Any], line: bytes) -> None:
        try:
            with self._write_lock:
                path = Path(self.settings.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed(path, len(line))
                jsonl_offset = path.stat().st_size if path.exists() else 0
                with path.open("ab") as handle:
                    handle.write(line)
                    if self.settings.fsync:
                        handle.flush()
                        os.fsync(handle.fileno())
                self._index_event(event, jsonl_path=path, jsonl_offset=jsonl_offset, jsonl_bytes=len(line))
        except Exception:
            self._write_failed = True

    def _write_worker(self) -> None:
        assert self._write_queue is not None
        while True:
            event, line = self._write_queue.get()
            try:
                self._write_line(event, line)
            finally:
                self._write_queue.task_done()

    def flush_writes(self, timeout_seconds: float | None = None) -> dict[str, Any]:
        """Wait for queued local writes to settle.

        The sync writer has nothing to flush. The async writer is daemon-backed,
        so this method only waits for the queue to drain and never stops traffic.
        """
        if self._write_queue is None:
            return {"async_writes_enabled": False, "queue_depth": 0, "settled": True}
        deadline = None if timeout_seconds is None else time.time() + max(0.0, timeout_seconds)
        while self._write_queue.unfinished_tasks:
            if deadline is not None and time.time() >= deadline:
                return {
                    "async_writes_enabled": True,
                    "queue_depth": self._write_queue.qsize(),
                    "settled": False,
                }
            time.sleep(0.01)
        return {"async_writes_enabled": True, "queue_depth": 0, "settled": True}

    def _serialized_line(self, event: dict[str, Any]) -> bytes | None:
        line = (json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode("utf-8")
        if self.max_event_bytes <= 0 or len(line) <= self.max_event_bytes:
            return line
        self._oversize_events += 1
        compact = {
            key: value
            for key, value in event.items()
            if key not in {"tool", "model", "side_effects", "runtime", "attributes"}
        }
        compact["attributes"] = {"oversize_original_bytes": len(line)}
        compact_line = (json.dumps(compact, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode("utf-8")
        if len(compact_line) <= self.max_event_bytes:
            return compact_line
        return None

    def _should_sample_out(self, event: dict[str, Any]) -> bool:
        if not self.settings.enabled or self.sample_rate >= 1.0:
            return False
        if event.get("status") in {"error", "blocked"}:
            return False
        if str(event.get("event_type") or "").startswith("runtime.policy."):
            return False
        return random.random() >= self.sample_rate

    def _validate_event(self, event: dict[str, Any]) -> None:
        if not self.settings.schema_validation_enabled:
            return
        try:
            from .flight_recorder_schema import validate_event_schema

            problems = validate_event_schema(event)
        except Exception as exc:
            problems = [f"schema_validator_error:{type(exc).__name__}"]
        if not problems:
            self._last_schema_errors = []
            return
        self._schema_invalid_events += 1
        self._last_schema_errors = problems[:10]
        if self.settings.schema_validation_strict:
            raise ValueError(f"Flight Recorder event schema invalid: {', '.join(problems)}")

    def _create_index(self) -> Any | None:
        if not self.settings.enabled or not self.settings.index_enabled:
            return None
        try:
            from .flight_recorder_index import FlightRecorderIndex, default_index_path

            index_path = Path(self.settings.index_path) if self.settings.index_path else default_index_path(self.settings.path)
            return FlightRecorderIndex(index_path)
        except Exception:
            self._index_failed = True
            return None

    def _index_event(self, event: dict[str, Any], *, jsonl_path: Path, jsonl_offset: int, jsonl_bytes: int) -> None:
        if self._index is None:
            return
        try:
            self._index.index_event(
                event,
                jsonl_path=str(jsonl_path),
                jsonl_offset=jsonl_offset,
                jsonl_bytes=jsonl_bytes,
            )
        except Exception:
            self._index_failed = True

    def _rotate_if_needed(self, path: Path, incoming_bytes: int) -> None:
        if self.rotate_bytes <= 0 or not path.exists():
            return
        try:
            if path.stat().st_size + incoming_bytes <= self.rotate_bytes:
                return
            rotated = path.with_name(f"{path.stem}.{int(time.time() * 1000)}{path.suffix}")
            path.replace(rotated)
            self._rotations += 1
            if self._index is not None:
                try:
                    self._index.update_jsonl_path(path, rotated)
                except Exception:
                    self._index_failed = True
            self._apply_retention(path)
        except Exception:
            self._write_failed = True

    def _apply_retention(self, active_path: Path) -> None:
        if self.retention_files <= 0:
            return
        try:
            rotated = self._rotated_files(active_path)
            rotated.sort(key=lambda item: item.stat().st_mtime, reverse=True)
            for stale in rotated[self.retention_files:]:
                try:
                    stale.unlink()
                    self._retention_deleted_files += 1
                    if self._index is not None:
                        self._index.delete_jsonl_path(stale)
                except Exception:
                    self._retention_failed = True
        except Exception:
            self._retention_failed = True

    def _current_file_bytes(self) -> int:
        if not self.settings.enabled:
            return 0
        try:
            path = Path(self.settings.path)
            return path.stat().st_size if path.exists() else 0
        except Exception:
            return 0

    def _rotated_file_count(self) -> int:
        if not self.settings.enabled:
            return 0
        try:
            return len(self._rotated_files(Path(self.settings.path)))
        except Exception:
            return 0

    def _rotated_files(self, active_path: Path) -> list[Path]:
        return [
            item
            for item in active_path.parent.glob(f"{active_path.stem}.*{active_path.suffix}")
            if item.is_file()
        ]

    def _buffer_otlp(self, event: dict[str, Any]) -> None:
        if not self.settings.enabled or not self.settings.otlp_enabled:
            return
        if len(self._otlp_buffer) >= self.otlp_max_buffer:
            self._otlp_buffer.pop(0)
            self._otlp_dropped += 1
        self._otlp_buffer.append(event)

    async def flush_otlp(self, http_client_factory: Any | None = None) -> dict[str, Any]:
        """Export buffered events as OTLP/HTTP JSON spans.

        This method is disabled unless `otlp_enabled` and an endpoint are set.
        Failures are recorded in status and do not raise.
        """
        if not self.settings.enabled or not self.settings.otlp_enabled or not self.settings.otlp_endpoint:
            return {"enabled": self.settings.otlp_enabled, "exported": 0, "skipped": True}
        if not self._otlp_buffer:
            return {"enabled": True, "exported": 0, "skipped": False}

        events = list(self._otlp_buffer)
        payload = self.otlp_payload(events)
        headers = {
            "Content-Type": "application/json",
            **parse_otlp_headers(self.settings.otlp_headers),
        }

        try:
            if http_client_factory is None:
                try:
                    import httpx
                except ModuleNotFoundError as exc:
                    self._otlp_failed = True
                    return {
                        "enabled": True,
                        "exported": 0,
                        "skipped": False,
                        "error_type": type(exc).__name__,
                        "error": (
                            "OTLP export requires optional dependency httpx. "
                            "Install it with: pip install hermes-flight-recorder[otlp]"
                        ),
                    }

                http_client_factory = lambda: httpx.AsyncClient(timeout=self.settings.otlp_timeout_seconds)
            async with http_client_factory() as client:
                response = await client.post(self.settings.otlp_endpoint, headers=headers, json=payload)
                response.raise_for_status()
            del self._otlp_buffer[:len(events)]
            self._otlp_failed = False
            return {"enabled": True, "exported": len(events), "skipped": False}
        except Exception as exc:
            self._otlp_failed = True
            return {"enabled": True, "exported": 0, "skipped": False, "error_type": type(exc).__name__}

    def otlp_flush_active(self) -> bool:
        """Whether a background OTLP flush loop should run.

        True only when the recorder is enabled AND OTLP export is configured.
        Keeps the loop strictly opt-in so a disabled-by-default recorder never
        spawns a background task.
        """
        return bool(self.settings.enabled and self.settings.otlp_enabled and self.settings.otlp_endpoint)

    async def otlp_flush_loop(self, http_client_factory: Any | None = None) -> None:
        """Background loop: periodically export buffered OTLP spans.

        Runs until cancelled. Without this, ``flush_otlp`` is only invoked at
        task end — a long run or concurrent runs overflow ``otlp_max_buffer``
        and drop the oldest spans before they ever reach the collector. This
        loop drains the buffer on a fixed cadence so spans flow continuously.

        Fail-open: a flush error is already captured by ``flush_otlp``
        (``otlp_failed``); any unexpected error here is swallowed and the loop
        continues. Only ``CancelledError`` stops it (on shutdown).
        """
        interval = max(0.1, float(self.otlp_flush_interval_seconds or 5.0))
        while True:
            try:
                await asyncio.sleep(interval)
                await self.flush_otlp(http_client_factory=http_client_factory)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - fail-open belt-and-suspenders
                self._otlp_failed = True

    def otlp_payload(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        spans = [event_to_otlp_span(event, include_previews=self.settings.otlp_include_previews) for event in events]
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            otlp_attribute("service.name", self.settings.otlp_service_name),
                            otlp_attribute("telemetry.sdk.language", "python"),
                            otlp_attribute("telemetry.sdk.name", "hermes-flight-recorder"),
                            otlp_attribute("hermes.flight_recorder.schema_version", SCHEMA_VERSION),
                            otlp_attribute("hermes.flight_recorder.recorder_version", RECORDER_VERSION),
                            otlp_attribute("hermes.flight_recorder.otel_mapping_version", OTEL_MAPPING_VERSION),
                            otlp_attribute("hermes.flight_recorder.openinference_mapping_version", OPENINFERENCE_MAPPING_VERSION),
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "hermes.flight_recorder", "version": SCHEMA_VERSION},
                            "spans": spans,
                        }
                    ],
                }
            ]
        }


def parse_otlp_headers(raw_headers: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in (raw_headers or "").split(","):
        if not item.strip() or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key:
            headers[key] = value.strip()
    return headers


def normalize_hash_strategy(value: str) -> str:
    strategy = (value or "hmac").strip().lower()
    if strategy not in {"hmac", "sha256"}:
        return "hmac"
    return strategy


def resolve_hash_secret(settings: FlightRecorderSettings) -> tuple[str, str]:
    if settings.hash_secret_file:
        try:
            secret = Path(settings.hash_secret_file).read_text(encoding="utf-8").strip()
            if secret:
                return secret, "file"
        except Exception:
            pass

    if settings.hash_secret_env:
        configured = settings.hash_secret_env.strip()
        if configured:
            value = os.environ.get(configured)
            if value:
                return value, "env"
            # The env var NAME was configured but the variable itself is unset or
            # empty. Never fall back to using the variable name as the secret —
            # that silently produces a public, guessable HMAC key.
            return DEFAULT_DEV_HASH_SECRET, "weak-env-fallback"

    return DEFAULT_DEV_HASH_SECRET, "default-dev"


def iso_to_unix_nano(value: str | None) -> str:
    if not value:
        return str(time.time_ns())
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return str(int(parsed.timestamp() * 1_000_000_000))
    except Exception:
        return str(time.time_ns())


def otlp_any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, dict)):
        return {"stringValue": canonical_json(value)}
    if value is None:
        return {"stringValue": ""}
    return {"stringValue": str(value)}


def otlp_attribute(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "value": otlp_any_value(value)}


def flatten_event_attributes(event: dict[str, Any], include_previews: bool = False) -> list[dict[str, Any]]:
    attrs: list[dict[str, Any]] = [
        otlp_attribute("hermes.event.type", event.get("event_type")),
        otlp_attribute("hermes.event.phase", event.get("phase")),
        otlp_attribute("hermes.event.source", event.get("source")),
        otlp_attribute("hermes.event.status", event.get("status")),
        otlp_attribute("hermes.actor", event.get("actor")),
        otlp_attribute("hermes.event.id", event.get("event_id")),
        otlp_attribute("hermes.event.parent_id", event.get("parent_event_id")),
        otlp_attribute("hermes.flight_recorder.schema_version", event.get("schema_version")),
        otlp_attribute("hermes.flight_recorder.recorder_version", event.get("recorder_version")),
        otlp_attribute("hermes.flight_recorder.semconv_version", event.get("semconv_version")),
        otlp_attribute("hermes.flight_recorder.otel_mapping_version", event.get("otel_mapping_version")),
        otlp_attribute("hermes.flight_recorder.openinference_mapping_version", OPENINFERENCE_MAPPING_VERSION),
        otlp_attribute("openinference.span.kind", openinference_span_kind_for_event(event)),
        otlp_attribute("graph.node.id", event.get("span_id")),
        otlp_attribute("graph.node.name", span_name_for_event(event)),
        otlp_attribute("graph.node.parent_id", event.get("parent_span_id")),
        otlp_attribute("w3c.traceparent", traceparent_for(str(event.get("trace_id") or ""), str(event.get("span_id") or ""))),
        otlp_attribute("session.id", event.get("session_id")),
        otlp_attribute("hermes.run.id", event.get("run_id")),
        otlp_attribute("hermes.task.id", event.get("task_id")),
    ]
    if isinstance(event.get("privacy"), dict):
        privacy = event["privacy"]
        attrs.extend([
            otlp_attribute("hermes.privacy.capture_mode", privacy.get("capture_mode")),
            otlp_attribute("hermes.privacy.redaction_policy_version", privacy.get("redaction_policy_version")),
            otlp_attribute("hermes.privacy.content_export_allowed", bool(privacy.get("content_export_allowed"))),
            otlp_attribute("hermes.privacy.hash_strategy", privacy.get("hash_strategy")),
        ])
    if isinstance(event.get("tool"), dict):
        tool = event["tool"]
        if tool.get("name"):
            attrs.append(otlp_attribute("gen_ai.operation.name", "execute_tool"))
            attrs.append(otlp_attribute("gen_ai.tool.name", tool.get("name")))
            if tool.get("category") == "mcp":
                attrs.append(otlp_attribute("mcp.method.name", "tools/call"))
        for key, value in tool.items():
            if key.endswith("_full") or (key.endswith("_preview") and not include_previews):
                continue
            attrs.append(otlp_attribute(f"hermes.tool.{key}", value))
    if isinstance(event.get("side_effects"), list):
        side_effects = [item for item in event["side_effects"] if isinstance(item, dict)]
        attrs.append(otlp_attribute("hermes.side_effect.count", len(side_effects)))
        if side_effects:
            attrs.append(otlp_attribute("hermes.side_effect.types", sorted({str(item.get("type")) for item in side_effects if item.get("type")})))
        for index, item in enumerate(side_effects[:10]):
            prefix = f"hermes.side_effect.{index}"
            for key, value in item.items():
                if key.endswith("_full") or (key.endswith("_preview") and not include_previews):
                    continue
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        attrs.append(otlp_attribute(f"{prefix}.metadata.{nested_key}", nested_value))
                    continue
                attrs.append(otlp_attribute(f"{prefix}.{key}", value))
    if isinstance(event.get("runtime"), dict):
        for key, value in event["runtime"].items():
            if key.endswith("_full") or (key.endswith("_preview") and not include_previews):
                continue
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    attrs.append(otlp_attribute(f"hermes.runtime.metadata.{nested_key}", nested_value))
                continue
            attrs.append(otlp_attribute(f"hermes.runtime.{key}", value))
    if isinstance(event.get("model"), dict):
        model = event["model"]
        provider = model.get("provider")
        name = model.get("name")
        attrs.append(otlp_attribute("gen_ai.provider.name", provider))
        attrs.append(otlp_attribute("gen_ai.request.model", name))
        if isinstance(model.get("input_tokens"), int):
            attrs.append(otlp_attribute("gen_ai.usage.input_tokens", model["input_tokens"]))
        if isinstance(model.get("output_tokens"), int):
            attrs.append(otlp_attribute("gen_ai.usage.output_tokens", model["output_tokens"]))
        if isinstance(model.get("total_tokens"), int):
            attrs.append(otlp_attribute("hermes.model.total_tokens", model["total_tokens"]))
        if isinstance(model.get("reasoning_tokens"), int):
            attrs.append(otlp_attribute("gen_ai.usage.reasoning_tokens", model["reasoning_tokens"]))
        if isinstance(model.get("cost_usd"), (int, float)):
            attrs.append(otlp_attribute("gen_ai.usage.cost", model["cost_usd"]))
            attrs.append(otlp_attribute("llm.cost.total", model["cost_usd"]))
        for key in ("request_model", "response_model", "prompt_hmac", "response_hmac"):
            if model.get(key):
                attrs.append(otlp_attribute(f"hermes.model.{key}", model[key]))
        for key in ("temperature", "max_tokens", "stream"):
            if key in model:
                attrs.append(otlp_attribute(f"hermes.model.{key}", model[key]))
    if isinstance(event.get("attributes"), dict):
        for key, value in event["attributes"].items():
            attrs.append(otlp_attribute(f"hermes.{key}", value))
    return [attr for attr in attrs if attr["value"] != {"stringValue": "None"}]


def openinference_span_kind_for_event(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type") or "")
    if event_type in {"hermes.session", "hermes.turn"}:
        return "AGENT"
    if event_type == "planning.route":
        return "CHAIN"
    if event_type in {"llm.call", "llm.retry", "llm.fallback"}:
        return "LLM"
    if event_type in {"tool.call", "mcp.transport", "mcp.tools.snapshot", "mcp.tools.diff"}:
        return "TOOL"
    if event_type.startswith("runtime.policy."):
        return "GUARDRAIL"
    if event_type in {"memory.search", "memory.read"}:
        return "RETRIEVER"
    if event_type == "eval.score":
        return "EVALUATOR"
    return "CHAIN"


def span_name_for_event(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type") or "hermes.event")
    tool = event.get("tool") if isinstance(event.get("tool"), dict) else {}
    if event_type == "tool.call" and tool.get("name"):
        return f"tool.call {tool['name']}"
    model = event.get("model") if isinstance(event.get("model"), dict) else {}
    if event_type == "llm.call" and model.get("name"):
        return f"llm.call {model['name']}"
    return event_type


def span_kind_for_event(event: dict[str, Any]) -> int:
    event_type = event.get("event_type")
    if event_type == "llm.call":
        return 3  # CLIENT
    if event_type == "tool.call":
        return 1  # INTERNAL
    return 1


def event_to_otlp_span(event: dict[str, Any], include_previews: bool = False) -> dict[str, Any]:
    start = event.get("start_ts") or event.get("end_ts")
    end = event.get("end_ts") or start
    span = {
        "traceId": str(event.get("trace_id") or ""),
        "spanId": str(event.get("span_id") or ""),
        "name": span_name_for_event(event),
        "kind": span_kind_for_event(event),
        "startTimeUnixNano": iso_to_unix_nano(start),
        "endTimeUnixNano": iso_to_unix_nano(end),
        "attributes": flatten_event_attributes(event, include_previews=include_previews),
        "status": {"code": 2 if event.get("status") == "error" else 1},
    }
    if event.get("parent_span_id"):
        span["parentSpanId"] = str(event["parent_span_id"])
    return span
