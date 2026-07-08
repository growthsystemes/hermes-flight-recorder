# hermes-flight-recorder

Privacy-first, in-process flight recorder for LLM agents. Originally built
inside a production agent named Hermes; the library itself has no dependency
on any specific agent framework or platform.

Hermes Flight Recorder is a local-first black box recorder for agents:
canonical JSONL, privacy-safe traces, replay, explain, query, redact-check, and
optional OTLP export. It is not a dashboard; JSONL is the local source of truth.

It answers one operational question for an agent run:

> What did the agent receive, decide to call, execute, modify, cost, and what
> did the environment allow or block?

## Features

- Canonical append-only JSONL event schema.
- Optional OTLP/HTTP export without the OpenTelemetry SDK.
- Metadata-first redaction with HMAC correlation.
- W3C trace context helpers and OpenInference OTLP projection.
- Local replay, timeline, query, explain, redact-check, and doctor CLIs.
- Disabled by default; `metadata` mode writes no raw payloads.

## Install

```bash
pip install hermes-flight-recorder
```

The core package has no runtime dependencies.

For OTLP/HTTP export, install the optional transport extra:

```bash
pip install "hermes-flight-recorder[otlp]"
```

The recorder does not depend on the OpenTelemetry SDK.

## Quickstart

No Hermes, no Docker, no Kubernetes required - the core package has zero
runtime dependencies, so this runs anywhere Python 3.10+ runs:

```bash
pip install hermes-flight-recorder
python examples/minimal-demo.py
hermes-fr timeline events.jsonl --summary
```

`examples/minimal-demo.py` wraps a generic made-up agent loop (a fake tool
call plus a fake LLM call) with no external services and no network calls -
the library is intentionally agent-agnostic and has no dependency on any
specific agent framework or platform.

## Library Usage

```python
from hermes_flight_recorder import FlightRecorder, FlightRecorderSettings

recorder = FlightRecorder(
    FlightRecorderSettings(enabled=True, path="events.jsonl", capture_mode="metadata")
)

with recorder.span(
    "tool.call",
    tool_name="get_current_weather",
    arguments={"city": "Lisbon"},
    run_id="demo",
) as span:
    span.set_result({"temperature_c": 21, "conditions": "clear"})
```

`FlightRecorder` is exported as a neutral alias of `HermesFlightRecorder`.
`HermesFlightRecorder.from_env()` / `FlightRecorder.from_env()` reads
`FLIGHT_RECORDER_*` variables directly for non-Hermes adopters.

For the lowest-level integration point, call `record(...)` directly. For the
common case, prefer:

- `with recorder.span("llm.call", ...)` for start/end span pairs.
- `async with recorder.aspan("llm.call", ...)` inside asyncio-native agents.
- `@recorder.trace_tool_call("tool_name")` to trace a sync or async Python
  function as a privacy-safe `tool.call`.

Example decorator:

```python
@recorder.trace_tool_call("lookup_customer", run_id="demo")
def lookup_customer(customer_id: str) -> dict[str, str]:
    return {"customer_id": customer_id, "status": "active"}
```

In `metadata` mode, arguments, results, prompts, and responses are written as
HMAC digests, not raw content.

## Public API

The stable public surface is the top-level `hermes_flight_recorder` export:

- `HermesFlightRecorder`
- `FlightRecorder`
- `FlightRecorderSettings`
- `FlightRecorderSpan`
- `AsyncFlightRecorderSpan`
- `utc_now_iso`
- `event_to_otlp_span`
- `redact_value`
- `trace_context_payload`
- `SCHEMA_VERSION`
- `RECORDER_VERSION`
- `SEMCONV_VERSION`
- `OTEL_MAPPING_VERSION`
- `OPENINFERENCE_MAPPING_VERSION`
- `__version__`

Other functions in submodules and `_`-prefixed attributes are internal. The
canonical JSONL event schema is the stable contract. OTLP/OpenInference is a
best-effort projection and is versioned separately.

## CLIs

Installed console scripts:

```bash
hermes-fr   --version
hermes-fr   timeline events.jsonl
hermes-fr   explain events.jsonl
hermes-fr   explain events.jsonl --run <run-id>
hermes-fr   redact-check events.jsonl
hermes-fr   doctor --check-privacy events.jsonl
fr-replay   --input fixture.json --output events.jsonl --capture-mode metadata
fr-timeline events.jsonl --summary --redaction-report --structural-report
fr-query    events.jsonl --rebuild-index --summary --json
fr-explain  events.jsonl --run <run-id>
```

Equivalent module forms are available, for example:

```bash
python -m hermes_flight_recorder.flight_recorder_timeline events.jsonl --summary
```

Local verification flow:

```bash
hermes-fr replay --input examples/policy-deny.json --output events.jsonl --capture-mode metadata
hermes-fr timeline events.jsonl --show-policy --show-hashes
hermes-fr explain events.jsonl
hermes-fr explain events.jsonl --run fixture-policy-deny
hermes-fr redact-check events.jsonl
hermes-fr doctor --check-otlp --events events.jsonl --payload-out otlp.json
```

## Privacy Model

In `metadata` mode, sensitive tool arguments, results, prompts, responses,
URLs, paths, and side-effect targets are never written raw. They are stored as
keyed HMAC digests so the same value can correlate across events without being
exposed.

Configure a strong per-environment key with `FlightRecorderSettings`:

```python
recorder = HermesFlightRecorder(FlightRecorderSettings(
    enabled=True,
    capture_mode="metadata",
    hash_strategy="hmac",
    hash_secret_env="FLIGHT_RECORDER_HMAC_KEY",
    require_strong_secret=True,
))
```

If `require_strong_secret=True` and the key is missing, the recorder disables
itself instead of emitting weakly keyed hashes. The status field
`weak_secret_blocked` reports that condition.

## OTLP Export

OTLP export is disabled unless the recorder is enabled and an endpoint is set.
Install `hermes-flight-recorder[otlp]` before calling `flush_otlp()` without a
custom HTTP client. If `httpx` is missing, `flush_otlp()` fails open, leaves the
buffer intact for retry, and returns an install hint in the error payload.

The exporter sends OTLP/HTTP JSON directly:

```python
recorder = HermesFlightRecorder(FlightRecorderSettings(
    enabled=True,
    otlp_enabled=True,
    otlp_endpoint="http://collector:4318/v1/traces",
    otlp_service_name="my-agent",  # defaults to "hermes-flight-recorder" if unset
))
```

The projection includes:

- `gen_ai.*` model/tool attributes where available.
- `mcp.*` transport attributes for MCP calls.
- `openinference.span.kind`.
- `graph.node.id`, `graph.node.name`, `graph.node.parent_id`.
- `w3c.traceparent`.
- `hermes.*` privacy, runtime policy, and event metadata.

Previews are excluded from OTLP unless explicitly enabled with
`otlp_include_previews=True`.

## Async Agents and Local Writes

`record(...)` is synchronous and writes to local JSONL. Async applications can
use `await recorder.arecord(...)` or `async with recorder.aspan(...)` to avoid
blocking the event loop with local file I/O. For high-throughput async agents,
also consider `FlightRecorderSettings(async_writes_enabled=True)` and call
`recorder.flush_writes()` during graceful shutdown.

The JSONL writer is thread-safe inside one Python process. If you run multiple
worker processes that all write to the same path, use one JSONL file per worker
or PID. Cross-process rotation locking is not part of the 0.1.x contract.

## Rotation and Retention Presets

These are human presets for `rotate_bytes` / `retention_files`; configure them
through `FlightRecorderSettings` or the equivalent environment variables.

| Preset | rotate_bytes | retention_files | Intent |
|---|---:|---:|---|
| `local-dev` | 16 MiB | 2 | Small laptop-safe local logs. |
| `ci` | 8 MiB | 1 | Short-lived fixture and smoke output. |
| `staging` | 64 MiB | 3 | Bounded canary and soak evidence. |
| `forensic` | 256 MiB | 16 | Longer local chain for incident review. |

## Integrations

Optional, dependency-free auto-instrumentation adapters live under
`hermes_flight_recorder.integrations`. Each adapter wraps a client in place --
no manual `span()`/`record()` calls needed at the call site -- and never
imports the SDK it instruments (duck-typed), so importing an adapter never
requires that SDK to be installed.

```python
from openai import OpenAI
from hermes_flight_recorder import HermesFlightRecorder, FlightRecorderSettings
from hermes_flight_recorder.integrations.openai import instrument_openai

recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=True, path="events.jsonl"))
client = instrument_openai(OpenAI(), recorder, run_id="demo")

client.chat.completions.create(model="gpt-4o-mini", messages=[...])
# -> a "llm.call" start/end pair is now in events.jsonl automatically.
```

`instrument_openai` also works with `AsyncOpenAI`, Azure OpenAI, and any
OpenAI-compatible client exposing the same `chat.completions.create` shape.
Streaming (`stream=True`) is handled correctly: the span stays open for the
full lifetime of the stream and only closes once it is exhausted, so duration
reflects actual generation time rather than time-to-first-chunk. Pass a
`cost_fn(model_name, usage_dict) -> float | None` if you want `cost_usd`
populated -- there is no built-in pricing table (it would go stale).

```python
from anthropic import Anthropic
from hermes_flight_recorder import HermesFlightRecorder, FlightRecorderSettings
from hermes_flight_recorder.integrations.anthropic import instrument_anthropic

recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=True, path="events.jsonl"))
client = instrument_anthropic(Anthropic(), recorder, run_id="demo")

client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=1024, messages=[...])
# -> a "llm.call" start/end pair is now in events.jsonl automatically.
```

`instrument_anthropic` also works with `AsyncAnthropic`. Streaming
(`stream=True` on `.create()`) is handled correctly, including Anthropic's
split usage reporting: unlike OpenAI's uniform per-chunk object, Anthropic's
stream is a sequence of `.type`-discriminated events -- `input_tokens` comes
from the `message_start` event, `output_tokens` from a later `message_delta`
event, and the adapter merges both before closing the span. Only
`client.messages.create(...)` is instrumented; Anthropic's separate
`client.messages.stream(...)` context-manager helper (`text_stream`,
`get_final_message()`) is a distinct SDK code path and is **not** covered --
calls made exclusively through `.stream()` will not emit spans.

LangChain callback handler adapter is tracked as future work.

## Versioning

Package version `0.1.3` matches `RECORDER_VERSION=0.1.3` — this is the public
PyPI release number, which restarted independently of internal pre-publication
iteration at `0.1.0` (see `CHANGELOG.md`).
`SCHEMA_VERSION=0.3.0` remains stable for this consolidation.

The JSONL schema is the durable contract. Compatibility within 0.3.x is
additive: readers accept unknown fields, and the SQLite index is disposable and
can be rebuilt from JSONL.

The 0.3 schema was validated against live production traffic on 2026-06-25:

- 199 metadata-only canary events.
- Redaction: 0 raw payload fields, 0 preview fields, 0 possible secret patterns.
- Structural report: 0 invalid events, 0 duplicate span IDs, 0 dangling parents.
- 0.3 gates: `mcp.tools.snapshot`, `eval.score`, and OTLP projection keys present.
- Live OTLP soak: 0 buffered events, 0 dropped events, 0 OTLP failure.

## Status

Pre-1.0 (`0.x`). Published on PyPI. The JSONL schema and public API are
stable within `0.x` (additive-only changes); breaking changes will bump to
`1.0`.

## License

MIT - see `LICENSE`.
