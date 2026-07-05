# Changelog

Public releases (PyPI) are numbered independently from the internal iteration
history below: the public version line restarts at `0.1.0` for the first
public release, regardless of how far the internal package had already
iterated (`0.3.1`) before publication. See "Internal iteration history" for
the pre-publication numbering that produced the `0.1.0` public release.

## Public releases

### Unreleased

- Removed remaining references to the internal "DataForge" platform name
  from public-facing text (README, package description/keywords, example
  docstring, two source comments) — the library was always agent/platform
  agnostic; this just removes the last vestiges of internal branding from
  what ships publicly. `pyproject.toml`'s `authors` field intentionally
  left unchanged pending a decision. Not yet published to PyPI (see
  `README.md`, which is current on GitHub but not yet reflected on the
  frozen `0.1.0` PyPI page — PyPI shows the README as of the version it was
  built with, not the latest source).

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
