"""Minimal, agent-agnostic Flight Recorder demo.

hermes-flight-recorder was originally built inside a production agent named
Hermes, but the library itself has zero dependency on any specific agent
framework or platform: it is a generic JSONL flight recorder for any agent
loop (LLM call -> tool call -> result).

This example wraps a made-up "research assistant" loop with no external
services, no network, no Docker, and no Kubernetes - just:

    pip install hermes-flight-recorder
    python minimal-demo.py
    hermes-fr timeline events.jsonl --show-errors --show-policy
"""

from hermes_flight_recorder import FlightRecorder, FlightRecorderSettings

rec = FlightRecorder(
    FlightRecorderSettings(
        enabled=True,
        path="events.jsonl",
        capture_mode="metadata",
    )
)

run_id = "demo-run-1"
trace_id = rec.trace_id(run_id)

# 1. Session start - one flight per "conversation" with the agent.
session_span = rec.span_id(run_id, "session")
rec.record(
    event_type="hermes.session",
    phase="start",
    trace_id=trace_id,
    span_id=session_span,
    run_id=run_id,
    session_id=run_id,
)

# 2. Turn start - one flight per user message the agent responds to.
turn_span = rec.span_id(run_id, "turn")
rec.record(
    event_type="hermes.turn",
    phase="start",
    trace_id=trace_id,
    span_id=turn_span,
    parent_span_id=session_span,
    run_id=run_id,
    session_id=run_id,
    turn_id=run_id,
)

# 3. The agent decides to call a tool - e.g. a weather lookup.
rec.record_tool_call(
    tool_name="get_current_weather",
    arguments={"city": "Lisbon"},
    result={"temperature_c": 21, "conditions": "clear"},
    status="ok",
    run_id=run_id,
    session_id=run_id,
    turn_id=run_id,
)

# 4. The agent synthesizes a final answer with a (fake) LLM call.
llm_span = rec.span_id(run_id, "llm")
rec.record(
    event_type="llm.call",
    phase="end",
    trace_id=trace_id,
    span_id=llm_span,
    parent_span_id=turn_span,
    run_id=run_id,
    session_id=run_id,
    turn_id=run_id,
    status="ok",
    duration_ms=420,
    model=rec.model_payload(
        provider="example-llm-provider",
        name="example-model-1",
        usage={"input_tokens": 128, "output_tokens": 64},
        cost_usd=0.0021,
    ),
)

# 5. Turn end, session end.
rec.record(
    event_type="hermes.turn",
    phase="end",
    trace_id=trace_id,
    span_id=turn_span,
    parent_span_id=session_span,
    run_id=run_id,
    session_id=run_id,
    turn_id=run_id,
    status="ok",
)
rec.record(
    event_type="hermes.session",
    phase="end",
    trace_id=trace_id,
    span_id=session_span,
    run_id=run_id,
    session_id=run_id,
    status="ok",
)

rec.flush_writes()
print("Wrote events.jsonl - inspect it with: hermes-fr timeline events.jsonl --summary")
