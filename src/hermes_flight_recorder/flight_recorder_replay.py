"""Fixture replay helper for Hermes Flight Recorder.

The replay CLI turns synthetic hook payloads into canonical recorder events.
It is meant for local schema/privacy iteration before enabling the recorder in
live Hermes pods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .flight_recorder import FlightRecorderSettings, HermesFlightRecorder


def load_fixture_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        events = [json.loads(line) for line in text.splitlines() if line.strip()]
        return {"events": events}

    payload = json.loads(text)
    if isinstance(payload, list):
        return {"events": payload}
    if isinstance(payload, dict):
        if isinstance(payload.get("events"), list):
            return payload
        return {"events": [payload]}
    raise ValueError(f"Unsupported fixture payload in {path}")


def load_fixture_events(path: Path) -> list[dict[str, Any]]:
    document = load_fixture_document(path)
    events = document.get("events")
    if not isinstance(events, list):
        raise ValueError(f"Fixture {path} does not contain an events list")
    return [_as_dict(event, path) for event in events]


def replay_fixture(
    *,
    input_path: Path,
    output_path: Path,
    capture_mode: str = "metadata",
    preview_chars: int = 500,
    append: bool = False,
    otlp_payload_path: Path | None = None,
    otlp_include_previews: bool = False,
    service_name: str = "hermes-flight-recorder-fixture",
) -> dict[str, Any]:
    document = load_fixture_document(input_path)
    raw_events = [_as_dict(event, input_path) for event in document.get("events", [])]
    run_id = str(document.get("run_id") or document.get("session_id") or "fixture-run")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not append:
        output_path.write_text("", encoding="utf-8")

    recorder = HermesFlightRecorder(
        FlightRecorderSettings(
            enabled=True,
            path=str(output_path),
            capture_mode=capture_mode,
            preview_chars=preview_chars,
            otlp_include_previews=otlp_include_previews,
            otlp_service_name=service_name,
        )
    )

    recorded: list[dict[str, Any]] = []
    for index, raw_event in enumerate(raw_events):
        recorded.append(_record_fixture_event(recorder, raw_event, document, run_id, index))

    if otlp_payload_path is not None:
        otlp_payload_path.parent.mkdir(parents=True, exist_ok=True)
        otlp_payload_path.write_text(
            json.dumps(recorder.otlp_payload(recorded), ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    return {
        "events": len(recorded),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "capture_mode": recorder.capture_mode,
        "otlp_payload_path": str(otlp_payload_path) if otlp_payload_path else None,
    }


def _record_fixture_event(
    recorder: HermesFlightRecorder,
    raw_event: dict[str, Any],
    document: dict[str, Any],
    run_id: str,
    index: int,
) -> dict[str, Any]:
    event_type = str(raw_event.get("event_type") or raw_event.get("type") or "hermes.event")
    phase = str(raw_event.get("phase") or _default_phase(raw_event))
    status = str(raw_event.get("status") or ("error" if raw_event.get("error") else "ok"))
    event_run_id = str(raw_event.get("run_id") or run_id)
    session_id = raw_event.get("session_id") or document.get("session_id") or event_run_id
    turn_id = raw_event.get("turn_id") or document.get("turn_id")
    task_id = raw_event.get("task_id") or document.get("task_id") or event_run_id
    trace_id = str(raw_event.get("trace_id") or recorder.trace_id(event_run_id))

    return recorder.record(
        event_type=event_type,
        phase=phase,
        trace_id=trace_id,
        span_id=str(raw_event.get("span_id") or recorder.span_id(event_run_id, event_type, index)),
        parent_span_id=raw_event.get("parent_span_id"),
        session_id=str(session_id) if session_id is not None else None,
        turn_id=str(turn_id) if turn_id is not None else None,
        run_id=event_run_id,
        task_id=str(task_id) if task_id is not None else None,
        source=str(raw_event.get("source") or document.get("source") or "fixture"),
        status=status,
        actor=str(raw_event.get("actor") or "agent"),
        start_ts=raw_event.get("start_ts"),
        end_ts=raw_event.get("end_ts"),
        duration_ms=_optional_int(raw_event.get("duration_ms")),
        model=_model_payload(recorder, raw_event),
        tool=_tool_payload(recorder, raw_event),
        side_effects=_side_effect_payloads(recorder, raw_event),
        runtime=_runtime_payload(recorder, raw_event),
        attributes=_optional_dict(raw_event.get("attributes")),
    )


def _as_dict(value: Any, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Fixture event in {path} must be an object")
    return value


def _default_phase(raw_event: dict[str, Any]) -> str:
    if raw_event.get("result") is not None or raw_event.get("error") is not None:
        return "end"
    if raw_event.get("arguments") is not None or raw_event.get("tool_name") is not None:
        return "start"
    return "instant"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_dict(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _tool_payload(recorder: HermesFlightRecorder, raw_event: dict[str, Any]) -> dict[str, Any] | None:
    raw_tool = raw_event.get("tool") if isinstance(raw_event.get("tool"), dict) else {}
    tool_name = raw_event.get("tool_name") or raw_tool.get("name")
    if not tool_name:
        return raw_tool or None

    arguments = raw_event.get("arguments", raw_tool.get("arguments"))
    result = raw_event.get("result", raw_tool.get("result"))
    error = raw_event.get("error", raw_tool.get("error"))
    payload = recorder.tool_payload(str(tool_name), arguments=arguments, result=result, error=error)
    for key, value in raw_tool.items():
        if key not in {"name", "arguments", "result", "error"} and key not in payload:
            payload[key] = value
    return payload


def _model_payload(recorder: HermesFlightRecorder, raw_event: dict[str, Any]) -> dict[str, Any] | None:
    raw_model = raw_event.get("model") if isinstance(raw_event.get("model"), dict) else {}
    provider = raw_event.get("provider", raw_model.get("provider"))
    name = raw_event.get("model_name", raw_model.get("name"))
    usage = raw_event.get("usage", raw_model.get("usage"))
    parameters = raw_event.get("parameters", raw_model.get("parameters"))
    if provider or name or isinstance(usage, dict) or isinstance(parameters, dict):
        payload = recorder.model_payload(
            provider=str(provider) if provider is not None else None,
            name=str(name) if name is not None else None,
            usage=usage if isinstance(usage, dict) else None,
            parameters=parameters if isinstance(parameters, dict) else None,
        )
        for key, value in raw_model.items():
            if key not in {"provider", "name", "usage", "parameters"} and key not in payload:
                payload[key] = value
        return payload
    return raw_model or None


def _side_effect_payloads(recorder: HermesFlightRecorder, raw_event: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw_side_effects = raw_event.get("side_effects")
    if raw_side_effects is None and isinstance(raw_event.get("side_effect"), dict):
        raw_side_effects = [raw_event["side_effect"]]
    if not isinstance(raw_side_effects, list):
        return None

    payloads: list[dict[str, Any]] = []
    for item in raw_side_effects:
        if not isinstance(item, dict):
            continue
        effect_type = item.get("type")
        if not effect_type:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        for key in ("exit_code", "bytes", "operation", "status"):
            if key in item and key not in metadata:
                metadata[key] = item[key]
        payloads.append(
            recorder.side_effect_payload(
                str(effect_type),
                target=item.get("target") or item.get("path") or item.get("url"),
                command=item.get("command"),
                before=item.get("before"),
                after=item.get("after"),
                diff=item.get("diff"),
                stdout=item.get("stdout"),
                stderr=item.get("stderr"),
                metadata=metadata or None,
            )
        )
    return payloads or None


def _runtime_payload(recorder: HermesFlightRecorder, raw_event: dict[str, Any]) -> dict[str, Any] | None:
    raw_policy = raw_event.get("runtime_policy")
    if isinstance(raw_policy, dict):
        return recorder.runtime_policy_payload(
            decision=str(raw_policy.get("decision") or raw_policy.get("egress_decision") or "unknown"),
            policy_type=str(raw_policy.get("policy_type") or "egress"),
            target=raw_policy.get("target") or raw_policy.get("network_peer") or raw_policy.get("url") or raw_policy.get("domain"),
            sandbox_id=raw_policy.get("sandbox_id"),
            policy_decision_id=raw_policy.get("policy_decision_id"),
            policy_version=raw_policy.get("policy_version"),
            secret_withheld=raw_policy.get("secret_withheld") if isinstance(raw_policy.get("secret_withheld"), bool) else None,
            correlation=raw_policy.get("correlation"),
            metadata=raw_policy.get("metadata") if isinstance(raw_policy.get("metadata"), dict) else None,
        )
    return _optional_dict(raw_event.get("runtime"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Hermes Flight Recorder fixture events.")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Fixture JSON or JSONL path.")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Canonical JSONL output path.")
    parser.add_argument("--capture-mode", default="metadata", choices=["metadata", "preview", "full", "forensic"])
    parser.add_argument("--preview-chars", default=500, type=int)
    parser.add_argument("--append", action="store_true", help="Append to output instead of truncating it first.")
    parser.add_argument("--otlp-payload", type=Path, help="Optional OTLP/HTTP JSON payload output path.")
    parser.add_argument("--otlp-include-previews", action="store_true")
    parser.add_argument("--service-name", default="hermes-flight-recorder-fixture")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
