"""Auto-instrumentation for the OpenAI Python SDK (and compatible clients).

Wraps ``client.chat.completions.create`` in place so every call emits a
canonical ``llm.call`` span through a :class:`HermesFlightRecorder`, with no
per-call-site instrumentation. This module never imports ``openai`` -- the
adapter is duck-typed against the ``client.chat.completions.create`` shape,
so it works unmodified with the OpenAI SDK, Azure OpenAI, and any
OpenAI-compatible client exposing the same interface.

Example::

    from openai import OpenAI
    from hermes_flight_recorder import HermesFlightRecorder, FlightRecorderSettings
    from hermes_flight_recorder.integrations.openai import instrument_openai

    recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=True, path="events.jsonl"))
    client = instrument_openai(OpenAI(), recorder, run_id="demo")

    client.chat.completions.create(model="gpt-4o-mini", messages=[...])
    # -> a "llm.call" start/end pair is now in events.jsonl, no span() call needed.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, AsyncIterator, Callable, Iterator

from ..flight_recorder import HermesFlightRecorder

_WRAPPED_ATTR = "_hermes_flight_recorder_wrapped"


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    to_dict = getattr(usage, "model_dump", None)
    if callable(to_dict):
        try:
            dumped = to_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


def instrument_openai(
    client: Any,
    recorder: HermesFlightRecorder,
    *,
    run_id: str = "default-run",
    provider: str = "openai",
    cost_fn: Callable[[str, dict[str, Any]], float | None] | None = None,
    **span_kwargs: Any,
) -> Any:
    """Instrument ``client.chat.completions.create`` in place (sync or async).

    Every call emits a canonical ``llm.call`` span (start/end) through
    ``recorder``: request/response model, token usage when the SDK exposes
    it, and duration -- all through the existing privacy/redaction pipeline
    (``recorder.model_payload``), same as a hand-written ``recorder.span(...)``
    call.

    Streaming (``stream=True``) is supported correctly: the span stays open
    for the lifetime of the returned stream and only closes once the stream
    is exhausted (or abandoned/errors), so duration reflects the actual
    generation time rather than just the time to get the first chunk. Final
    token usage is only captured for streams created with
    ``stream_options={"include_usage": True}`` (an OpenAI SDK option) --
    without it, the end event still records model/duration/status but no
    token counts.

    ``cost_fn(model_name, usage_dict) -> float | None`` is an optional hook
    for callers who want ``cost_usd`` populated; this adapter does not ship a
    hardcoded pricing table (it would go stale).

    Duck-typed: this module never imports ``openai``, so it works with any
    client exposing the same ``client.chat.completions.create`` shape
    (OpenAI, Azure OpenAI, OpenAI-compatible gateways). Returns ``client``
    for chaining. Idempotent: instrumenting an already-instrumented client is
    a no-op.
    """
    completions = client.chat.completions
    if getattr(completions, _WRAPPED_ATTR, False):
        return client

    original_create = completions.create

    def _build_model(request_model: str | None) -> dict[str, Any]:
        return recorder.model_payload(provider=provider, name=request_model, request_model=request_model)

    def _finalize_model(span: Any, request_model: str | None, response_model: str | None, usage: Any) -> None:
        usage_dict = _usage_dict(usage)
        model_name = response_model or request_model
        cost = None
        if cost_fn is not None and model_name:
            try:
                cost = cost_fn(model_name, usage_dict)
            except Exception:
                cost = None
        span.set_model(
            recorder.model_payload(
                provider=provider,
                name=model_name,
                usage=usage_dict,
                request_model=request_model,
                response_model=response_model,
                cost_usd=cost,
            )
        )

    def _wrap_sync_stream(stream: Iterator[Any], span: Any, request_model: str | None) -> Iterator[Any]:
        def generator() -> Iterator[Any]:
            last_usage = None
            response_model = None
            completed = False
            try:
                for chunk in stream:
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        last_usage = chunk_usage
                    response_model = getattr(chunk, "model", None) or response_model
                    yield chunk
                completed = True
            except GeneratorExit:
                span.status = "cancelled"
                span.end()
                raise
            except BaseException as exc:
                span.end(exc)
                raise
            if completed:
                _finalize_model(span, request_model, response_model, last_usage)
                span.end()

        return generator()

    async def _wrap_async_stream(stream: AsyncIterator[Any], span: Any, request_model: str | None) -> AsyncIterator[Any]:
        last_usage = None
        response_model = None
        completed = False
        try:
            async for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    last_usage = chunk_usage
                response_model = getattr(chunk, "model", None) or response_model
                yield chunk
            completed = True
        except GeneratorExit:
            span.status = "cancelled"
            span.end()
            raise
        except BaseException as exc:
            span.end(exc)
            raise
        if completed:
            _finalize_model(span, request_model, response_model, last_usage)
            span.end()

    # The real SDK wraps `create` in decorators (overload/required-args dispatch),
    # so `iscoroutinefunction` on the bound method itself is always False even for
    # AsyncOpenAI -- unwrap through `__wrapped__` first or async clients silently
    # get routed through the sync path. Verified against the real SDK (0.x), not
    # just its type stubs.
    if inspect.iscoroutinefunction(inspect.unwrap(original_create)):

        @functools.wraps(original_create)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            request_model = kwargs.get("model")
            span = recorder.span("llm.call", run_id=run_id, model=_build_model(request_model), **span_kwargs)
            span.__enter__()
            try:
                response = await original_create(*args, **kwargs)
            except BaseException as exc:
                span.end(exc)
                raise
            if kwargs.get("stream"):
                return _wrap_async_stream(response, span, request_model)
            _finalize_model(span, request_model, getattr(response, "model", None), getattr(response, "usage", None))
            span.end()
            return response

        completions.create = async_wrapper
    else:

        @functools.wraps(original_create)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            request_model = kwargs.get("model")
            span = recorder.span("llm.call", run_id=run_id, model=_build_model(request_model), **span_kwargs)
            span.__enter__()
            try:
                response = original_create(*args, **kwargs)
            except BaseException as exc:
                span.end(exc)
                raise
            if kwargs.get("stream"):
                return _wrap_sync_stream(response, span, request_model)
            _finalize_model(span, request_model, getattr(response, "model", None), getattr(response, "usage", None))
            span.end()
            return response

        completions.create = sync_wrapper

    setattr(completions, _WRAPPED_ATTR, True)
    return client
