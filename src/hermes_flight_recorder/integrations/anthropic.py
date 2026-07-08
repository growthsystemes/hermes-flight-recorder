"""Auto-instrumentation for the Anthropic Python SDK.

Wraps ``client.messages.create`` in place so every call emits a canonical
``llm.call`` span through a :class:`HermesFlightRecorder`, with no
per-call-site instrumentation. This module never imports ``anthropic`` -- the
adapter is duck-typed against the ``client.messages.create`` shape.

Only ``client.messages.create(...)`` (streaming and non-streaming) is
instrumented. Anthropic's higher-level ``client.messages.stream(...)``
context-manager helper (``with client.messages.stream(...) as stream:``,
``stream.text_stream``, ``stream.get_final_message()``) is a separate SDK
code path -- calls made exclusively through ``.stream()`` are **not** covered
by this adapter. Tracked as explicit future work, not silently assumed away.

Example::

    from anthropic import Anthropic
    from hermes_flight_recorder import HermesFlightRecorder, FlightRecorderSettings
    from hermes_flight_recorder.integrations.anthropic import instrument_anthropic

    recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=True, path="events.jsonl"))
    client = instrument_anthropic(Anthropic(), recorder, run_id="demo")

    client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=1024, messages=[...])
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
        return {k: v for k, v in usage.items() if v is not None}
    to_dict = getattr(usage, "model_dump", None)
    if callable(to_dict):
        try:
            dumped = to_dict()
            if isinstance(dumped, dict):
                return {k: v for k, v in dumped.items() if v is not None}
        except Exception:
            pass
    return {
        key: getattr(usage, key)
        for key in ("input_tokens", "output_tokens")
        if getattr(usage, key, None) is not None
    }


def _merge_stream_usage(input_usage: Any, output_usage: Any) -> dict[str, Any]:
    """Merge Anthropic's split streaming usage into one dict.

    ``input_tokens`` is only reliably known from the ``message_start`` event
    (``message.usage``); ``output_tokens`` only becomes final on a later
    ``message_delta`` event's own ``usage``. ``MessageDeltaUsage`` also has an
    optional ``input_tokens`` field (verified against the real SDK: it's
    ``None`` in the common case) -- ``_usage_dict`` already drops ``None``
    values, so a present value there correctly overrides the start-event
    figure instead of a stray ``None`` blanking it out.
    """
    merged = _usage_dict(input_usage)
    merged.update(_usage_dict(output_usage))
    return merged


def instrument_anthropic(
    client: Any,
    recorder: HermesFlightRecorder,
    *,
    run_id: str = "default-run",
    provider: str = "anthropic",
    cost_fn: Callable[[str, dict[str, Any]], float | None] | None = None,
    **span_kwargs: Any,
) -> Any:
    """Instrument ``client.messages.create`` in place (sync or async).

    Every call emits a canonical ``llm.call`` span (start/end) through
    ``recorder``: request/response model, token usage when the SDK exposes
    it, and duration -- all through the existing privacy/redaction pipeline
    (``recorder.model_payload``), same as a hand-written ``recorder.span(...)``
    call. ``HermesFlightRecorder.model_payload`` already recognizes
    Anthropic's ``input_tokens``/``output_tokens`` field names natively; no
    changes were needed there.

    Streaming (``stream=True``) is supported correctly, including Anthropic's
    split usage reporting: unlike a uniform per-chunk object, Anthropic's
    stream is a sequence of ``.type``-discriminated events
    (``message_start``, ``content_block_start``, ``content_block_delta``,
    ``content_block_stop``, ``message_delta``, ``message_stop``).
    ``input_tokens`` is read from ``message_start.message.usage`` and
    ``output_tokens`` from the later ``message_delta.usage`` -- merged via
    :func:`_merge_stream_usage`. The span stays open for the lifetime of the
    returned stream and only closes once it is exhausted (or
    abandoned/errors), so duration reflects the actual generation time rather
    than just the time to get the first event.

    Only ``client.messages.create(...)`` is instrumented. Anthropic's
    separate ``client.messages.stream(...)`` context-manager helper is a
    distinct code path and is **not** covered -- see the module docstring.

    ``cost_fn(model_name, usage_dict) -> float | None`` is an optional hook
    for callers who want ``cost_usd`` populated; this adapter does not ship a
    hardcoded pricing table (it would go stale).

    Duck-typed: this module never imports ``anthropic``, so it works with any
    client exposing the same ``client.messages.create`` shape. Returns
    ``client`` for chaining. Idempotent: instrumenting an already-instrumented
    client is a no-op.
    """
    messages = client.messages
    if getattr(messages, _WRAPPED_ATTR, False):
        return client

    original_create = messages.create

    def _build_model(request_model: str | None) -> dict[str, Any]:
        return recorder.model_payload(provider=provider, name=request_model, request_model=request_model)

    def _finalize_model(span: Any, request_model: str | None, response_model: str | None, usage: dict[str, Any]) -> None:
        model_name = response_model or request_model
        cost = None
        if cost_fn is not None and model_name:
            try:
                cost = cost_fn(model_name, usage)
            except Exception:
                cost = None
        span.set_model(
            recorder.model_payload(
                provider=provider,
                name=model_name,
                usage=usage,
                request_model=request_model,
                response_model=response_model,
                cost_usd=cost,
            )
        )

    def _consume_event(event: Any, state: dict[str, Any]) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            message = getattr(event, "message", None)
            if message is not None:
                state["response_model"] = getattr(message, "model", None) or state["response_model"]
                state["input_usage"] = getattr(message, "usage", None)
        elif event_type == "message_delta":
            state["output_usage"] = getattr(event, "usage", None)

    def _wrap_sync_stream(stream: Iterator[Any], span: Any, request_model: str | None) -> Iterator[Any]:
        def generator() -> Iterator[Any]:
            state: dict[str, Any] = {"response_model": None, "input_usage": None, "output_usage": None}
            completed = False
            try:
                for event in stream:
                    _consume_event(event, state)
                    yield event
                completed = True
            except GeneratorExit:
                span.status = "cancelled"
                span.end()
                raise
            except BaseException as exc:
                span.end(exc)
                raise
            if completed:
                usage = _merge_stream_usage(state["input_usage"], state["output_usage"])
                _finalize_model(span, request_model, state["response_model"], usage)
                span.end()

        return generator()

    async def _wrap_async_stream(stream: AsyncIterator[Any], span: Any, request_model: str | None) -> AsyncIterator[Any]:
        state: dict[str, Any] = {"response_model": None, "input_usage": None, "output_usage": None}
        completed = False
        try:
            async for event in stream:
                _consume_event(event, state)
                yield event
            completed = True
        except GeneratorExit:
            span.status = "cancelled"
            span.end()
            raise
        except BaseException as exc:
            span.end(exc)
            raise
        if completed:
            usage = _merge_stream_usage(state["input_usage"], state["output_usage"])
            _finalize_model(span, request_model, state["response_model"], usage)
            span.end()

    # See integrations/openai.py for why the unwrap is required: the real SDK
    # wraps `create` in decorators (overload/required-args dispatch), so
    # `iscoroutinefunction` on the bound method itself is always False even
    # for AsyncAnthropic. Verified against the real, installed `anthropic`
    # SDK (0.116.0), not just its type stubs.
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
            _finalize_model(
                span,
                request_model,
                getattr(response, "model", None),
                _usage_dict(getattr(response, "usage", None)),
            )
            span.end()
            return response

        messages.create = async_wrapper
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
            _finalize_model(
                span,
                request_model,
                getattr(response, "model", None),
                _usage_dict(getattr(response, "usage", None)),
            )
            span.end()
            return response

        messages.create = sync_wrapper

    setattr(messages, _WRAPPED_ATTR, True)
    return client
