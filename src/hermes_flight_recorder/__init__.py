"""Hermes Flight Recorder - privacy-first, in-process observability for agents.

Append-oriented JSONL recorder with optional OTLP export, redaction, HMAC
correlation, schema validation, and a SQLite index.

Public API::

    from hermes_flight_recorder import HermesFlightRecorder, FlightRecorderSettings

    recorder = HermesFlightRecorder.from_env()
    # or: recorder = HermesFlightRecorder.from_config(config)
    recorder.record(event_type="tool.call", ...)

CLIs installed as console scripts: ``hermes-fr`` plus the legacy aliases
``fr-replay``, ``fr-timeline``, ``fr-query``, and ``fr-explain``.
"""

from __future__ import annotations

from .flight_recorder import (
    OPENINFERENCE_MAPPING_VERSION,
    OTEL_MAPPING_VERSION,
    RECORDER_VERSION,
    SCHEMA_VERSION,
    SEMCONV_VERSION,
    FlightRecorderSettings,
    HermesFlightRecorder,
    event_to_otlp_span,
    redact_value,
    trace_context_payload,
    utc_now_iso,
)

# Neutral alias to ease generic (non-Hermes) adoption. Kept additive so the
# Hermes-specific name remains the stable, documented entrypoint.
FlightRecorder = HermesFlightRecorder

__version__ = RECORDER_VERSION

__all__ = [
    "HermesFlightRecorder",
    "FlightRecorder",
    "FlightRecorderSettings",
    "utc_now_iso",
    "event_to_otlp_span",
    "redact_value",
    "trace_context_payload",
    "SCHEMA_VERSION",
    "RECORDER_VERSION",
    "SEMCONV_VERSION",
    "OTEL_MAPPING_VERSION",
    "OPENINFERENCE_MAPPING_VERSION",
    "__version__",
]
