# Changelog

Public releases (PyPI) are numbered independently from the internal iteration
history below: the public version line restarts at `0.1.0` for the first
public release, regardless of how far the internal package had already
iterated (`0.3.1`) before publication. See "Internal iteration history" for
the pre-publication numbering that produced the `0.1.0` public release.

## Public releases

### 0.1.3 - 2026-07-06

- Added `hermes_flight_recorder.integrations.anthropic.instrument_anthropic(client, recorder, ...)`:
  wraps `client.messages.create` (sync or async, streaming or not) in place
  so every Anthropic SDK call auto-emits a canonical `llm.call` span. Handles
  Anthropic's split streaming usage correctly -- unlike OpenAI's uniform
  per-chunk object, Anthropic's stream is a sequence of `.type`-discriminated
  events (`message_start`, `content_block_delta`, `message_delta`,
  `message_stop`); `input_tokens` is read from `message_start.message.usage`
  and `output_tokens` from a later `message_delta.usage`, then merged.
  Verified this event shape against the real, installed `anthropic` SDK
  (0.116.0) before writing the branching logic, not assumed.
  Known, documented gap: only `client.messages.create(...)` is instrumented.
  Anthropic's separate `client.messages.stream(...)` context-manager helper
  (`text_stream`, `get_final_message()`) is a distinct SDK code path and is
  **not** covered -- calls made exclusively through `.stream()` will not
  emit spans. Tracked as explicit future work.
- Fixed a real bug in `instrument_openai` (shipped in 0.1.2, not yet
  published anywhere): the real OpenAI/Anthropic SDKs wrap `create` in
  dispatcher decorators (overload/required-args resolution), so
  `inspect.iscoroutinefunction` on the bound method is always `False` even
  for `AsyncOpenAI` -- the adapter's sync/async detection silently
  misclassified every real async client as sync. This didn't surface in the
  original tests because the fake test client's `create` was a plain
  `async def` with no such wrapping. Effect in practice: no crash, just a
  span silently closed with no usage/model captured, because the sync
  wrapper path finalized the span using the *unawaited* coroutine object
  before the real call ever ran. Fixed by unwrapping through `__wrapped__`
  (`inspect.iscoroutinefunction(inspect.unwrap(original_create))`) in both
  adapters; regression test added to both suites using a fake `create`
  wrapped the same way the real SDKs wrap theirs. Verified directly against
  the real, installed `openai` and `anthropic` packages, not just their type
  stubs.
- Package tests: `Ran 84 tests ... OK` (73 after 0.1.2, +11 for the
  Anthropic adapter and its regression coverage).
- Not published anywhere yet (this version does not exist on GitHub or PyPI
  yet) -- built and tested locally in the monorepo.

### 0.1.2 - 2026-07-06

- Added `hermes_flight_recorder.integrations.openai.instrument_openai(client, recorder, ...)`:
  wraps `client.chat.completions.create` (sync or async, streaming or not) in
  place so every OpenAI SDK call auto-emits a canonical `llm.call` span --
  no per-call-site `span()`/`record()` needed. Duck-typed (never imports
  `openai`), so it also works with Azure OpenAI and OpenAI-compatible
  clients. Streaming responses close their span only once the stream is
  exhausted, so duration reflects actual generation time. Optional
  `cost_fn` hook for callers who want `cost_usd` populated (no built-in
  pricing table). First of the auto-instrumentation adapters tracked as
  future work in `agents/hermes/plugin-tracking/progress/NEXT-WORK.md`.
- Not published anywhere yet (this version does not exist on GitHub or PyPI
  yet) -- built and tested locally in the monorepo.

### 0.1.1 - 2026-07-05

- Added generic span helpers for public adopters: `recorder.span(...)`,
  `recorder.aspan(...)`, `recorder.arecord(...)`, and
  `@recorder.trace_tool_call(...)`. They emit the existing canonical JSONL
  events and do not depend on any agent framework.
- Added the `py.typed` marker so type checkers can consume the package's
  inline type hints.
- Added a standalone GitHub Actions CI workflow for the public repository
  shape: unit tests, build, `twine check`, import smoke, and CLI smoke.
- Documented async usage and the current multi-process JSONL boundary: the
  writer is thread-safe inside one process; multi-worker deployments should
  use one JSONL file per worker/PID until cross-process rotation locking is
  explicitly supported.
- Removed remaining references to the internal platform name
  from public-facing text (README, package description/keywords, example
  docstring, two source comments) — the library was always agent/platform
  agnostic; this just removes the last vestiges of internal branding from
  what ships publicly, including package author metadata.

### 0.1.0 - 2026-07-05

First public release on PyPI. Ships the consolidated feature set built up
through internal iterations `0.1.0`-`0.3.1` (see "Internal iteration history"
below): stable JSONL schema `0.3.0`, zero core runtime dependencies, optional
`[otlp]` extra, `hermes-fr` CLI (`timeline`/`explain`/`query`/`redact-check`/
`doctor`), HMAC-based redaction with metadata/preview/full/forensic capture
modes, and the `LIVE` / `VM-SANDBOX-BETA` / `GATED` / `RESERVED` event-type
split. The sandbox/code-runner capability is not shipped in this release.

## Internal iteration history

These versions were built and tested internally but never published to
PyPI; they are kept here for provenance of what became the public `0.1.0`.

### 0.3.1 (internal) - 2026-06-28

- Consolidated Hermes flight recorder lineage; source-of-truth relocation and
  documentation guardrail additions ahead of the public release decision.

### 0.3.0 (internal) - 2026-06-25

- Regenerated the package from the staging-validated Hermes Flight Recorder 0.3 runtime.
- Added `mcp.tools.snapshot`, `mcp.tools.diff`, and `eval.score` schema support.
- Added W3C trace context helpers and OpenInference OTLP projection attributes.
- Added structural report gates for duplicate span IDs and dangling parents.
- Added `fr-explain` console script.
- Declared `httpx` as the OTLP/HTTP export dependency.

### 0.1.0 (internal) - 2026-06-23

- Initial package extraction from the consolidated Flight Recorder core.
