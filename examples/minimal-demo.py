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

# 1. Session start - one flight per "conversation" with the agent.
with rec.span("hermes.session", run_id=run_id) as session:
    # 2. Turn start - one flight per user message the agent responds to.
    with rec.span(
        "hermes.turn",
        run_id=run_id,
        parent_span_id=session.span_id,
        turn_id=run_id,
    ) as turn:
        # 3. The agent decides to call a tool - e.g. a weather lookup.
        with rec.span(
            "tool.call",
            tool_name="get_current_weather",
            arguments={"city": "Lisbon"},
            run_id=run_id,
            parent_span_id=turn.span_id,
            turn_id=run_id,
        ) as tool:
            tool.set_result({"temperature_c": 21, "conditions": "clear"})

        # 4. The agent synthesizes a final answer with a (fake) LLM call.
        with rec.span(
            "llm.call",
            run_id=run_id,
            parent_span_id=turn.span_id,
            turn_id=run_id,
            model=rec.model_payload(provider="example-llm-provider", name="example-model-1"),
        ) as llm:
            llm.set_model(
                rec.model_payload(
                    provider="example-llm-provider",
                    name="example-model-1",
                    usage={"input_tokens": 128, "output_tokens": 64},
                    cost_usd=0.0021,
                )
            )

rec.flush_writes()
print("Wrote events.jsonl - inspect it with: hermes-fr timeline events.jsonl --summary")
