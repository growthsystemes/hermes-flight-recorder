"""Tests for hermes_flight_recorder.integrations.anthropic.

Uses lightweight fake clients that mimic the Anthropic SDK's
``client.messages.create`` shape -- no real ``anthropic`` dependency,
consistent with the package's zero-dependency test philosophy. Streaming
event fakes mirror the real SDK's ``.type``-discriminated shape, verified
against the actual installed ``anthropic`` package (0.116.0) before writing
this adapter: ``message_start.message.usage.input_tokens`` and
``message_delta.usage.output_tokens`` are the two fields that matter;
``message_delta.usage.input_tokens`` exists but is normally ``None``.
"""

from __future__ import annotations

import functools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hermes_flight_recorder import FlightRecorderSettings, HermesFlightRecorder
from hermes_flight_recorder.integrations.anthropic import instrument_anthropic


class _FakeUsage:
    def __init__(self, input_tokens=None, output_tokens=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def model_dump(self) -> dict:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def _message_start_event(model: str, input_tokens: int):
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(model=model, usage=_FakeUsage(input_tokens=input_tokens, output_tokens=0)),
    )


def _content_block_delta_event():
    return SimpleNamespace(type="content_block_delta")


def _message_delta_event(output_tokens: int):
    # Real SDK: MessageDeltaUsage.input_tokens is Optional[int] = None in the
    # common case -- this fake matches that so the merge-filters-None logic
    # is actually exercised, not just assumed.
    return SimpleNamespace(type="message_delta", usage=_FakeUsage(input_tokens=None, output_tokens=output_tokens))


def _message_stop_event():
    return SimpleNamespace(type="message_stop")


class _FakeClient:
    def __init__(self, messages):
        self.messages = messages


class _FakeSyncMessages:
    def __init__(self, response=None, stream_events=None, error=None):
        self._response = response
        self._stream_events = stream_events
        self._error = error

    def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        if kwargs.get("stream"):
            return iter(self._stream_events)
        return self._response


class _FakeAsyncMessages:
    def __init__(self, response=None, stream_events=None, error=None):
        self._response = response
        self._stream_events = stream_events
        self._error = error

    async def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        if kwargs.get("stream"):
            async def agen():
                for event in self._stream_events:
                    yield event

            return agen()
        return self._response


def _sdk_dispatch_wrapper(func):
    """Mirrors integrations/openai.py's regression fake -- see that file's
    equivalent for the full explanation. Verified this pattern reproduces the
    real Anthropic SDK's `iscoroutinefunction(bound_method) == False` shape
    directly against the installed `anthropic` package."""

    @functools.wraps(func)
    def dispatcher(*args, **kwargs):
        return func(*args, **kwargs)

    return dispatcher


class _FakeAsyncMessagesSDKWrapped:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    @_sdk_dispatch_wrapper
    async def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        return self._response


def _make_recorder(path: Path) -> HermesFlightRecorder:
    return HermesFlightRecorder(FlightRecorderSettings(enabled=True, path=str(path), capture_mode="metadata"))


def _load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class AnthropicInstrumentationSyncTest(unittest.TestCase):
    def test_non_streaming_call_emits_llm_call_span_with_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(
                model="claude-3-5-sonnet-20241022",
                usage=_FakeUsage(input_tokens=12, output_tokens=6),
            )
            client = _FakeClient(_FakeSyncMessages(response=response))

            instrument_anthropic(client, recorder, run_id="run-1")
            result = client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[])

            self.assertIs(result, response)
            events = _load_events(path)
            self.assertEqual(len(events), 2)
            start, end = events
            self.assertEqual(start["event_type"], "llm.call")
            self.assertEqual(start["phase"], "start")
            self.assertEqual(end["phase"], "end")
            self.assertEqual(end["status"], "ok")
            self.assertEqual(end["model"]["input_tokens"], 12)
            self.assertEqual(end["model"]["output_tokens"], 6)
            self.assertEqual(end["model"]["name"], "claude-3-5-sonnet-20241022")
            # Anthropic has no total_tokens field at all -- confirm it's simply absent, not zero.
            self.assertNotIn("total_tokens", end["model"])

    def test_exception_marks_span_error_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            client = _FakeClient(_FakeSyncMessages(error=RuntimeError("boom")))
            instrument_anthropic(client, recorder, run_id="run-1")

            with self.assertRaises(RuntimeError):
                client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[])

            events = _load_events(path)
            self.assertEqual(events[-1]["phase"], "end")
            self.assertEqual(events[-1]["status"], "error")

    def test_streaming_call_merges_input_and_output_usage_across_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            events_seq = [
                _message_start_event("claude-3-5-sonnet-20241022", input_tokens=20),
                _content_block_delta_event(),
                _content_block_delta_event(),
                _message_delta_event(output_tokens=8),
                _message_stop_event(),
            ]
            client = _FakeClient(_FakeSyncMessages(stream_events=events_seq))
            instrument_anthropic(client, recorder, run_id="run-1")

            stream = client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[], stream=True)
            # create() returned but the stream hasn't been consumed -- only the
            # start event should exist so far.
            self.assertEqual(len(_load_events(path)), 1)

            collected = list(stream)

            self.assertEqual(len(collected), 5)
            events = _load_events(path)
            self.assertEqual(len(events), 2)
            end = events[-1]
            self.assertEqual(end["status"], "ok")
            # input_tokens comes from message_start, output_tokens from the
            # later message_delta -- proving the cross-event merge works,
            # not just a flat per-chunk getattr like the OpenAI adapter.
            self.assertEqual(end["model"]["input_tokens"], 20)
            self.assertEqual(end["model"]["output_tokens"], 8)

    def test_streaming_call_abandoned_early_closes_span_as_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            events_seq = [
                _message_start_event("claude-3-5-sonnet-20241022", input_tokens=20),
                _content_block_delta_event(),
                _content_block_delta_event(),
                _message_delta_event(output_tokens=8),
                _message_stop_event(),
            ]
            client = _FakeClient(_FakeSyncMessages(stream_events=events_seq))
            instrument_anthropic(client, recorder, run_id="run-1")

            stream = client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[], stream=True)
            next(stream)
            stream.close()

            events = _load_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["status"], "cancelled")

    def test_double_instrumentation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(model="claude-3-5-sonnet-20241022", usage=_FakeUsage())
            client = _FakeClient(_FakeSyncMessages(response=response))

            instrument_anthropic(client, recorder, run_id="run-1")
            wrapped_once = client.messages.create
            instrument_anthropic(client, recorder, run_id="run-1")

            self.assertIs(client.messages.create, wrapped_once)

    def test_instruments_fine_without_a_stream_attribute(self):
        # Documents-by-test that instrument_anthropic has no hidden
        # dependency on client.messages.stream existing -- v1 only wraps
        # .create(), by design (see module docstring).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(model="claude-3-5-sonnet-20241022", usage=_FakeUsage())
            fake_messages = _FakeSyncMessages(response=response)
            self.assertFalse(hasattr(fake_messages, "stream"))
            client = _FakeClient(fake_messages)

            instrument_anthropic(client, recorder, run_id="run-1")
            client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[])

            self.assertEqual(len(_load_events(path)), 2)

    def test_cost_fn_populates_cost_usd(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(
                model="claude-3-5-sonnet-20241022",
                usage=_FakeUsage(input_tokens=1000, output_tokens=500),
            )
            client = _FakeClient(_FakeSyncMessages(response=response))

            def cost_fn(model_name, usage):
                return usage.get("input_tokens", 0) * 0.003 + usage.get("output_tokens", 0) * 0.015

            instrument_anthropic(client, recorder, run_id="run-1", cost_fn=cost_fn)
            client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[])

            events = _load_events(path)
            self.assertAlmostEqual(events[-1]["model"]["cost_usd"], 10.5)


class AnthropicInstrumentationAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_async_non_streaming_call_emits_llm_call_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(
                model="claude-3-5-sonnet-20241022",
                usage=_FakeUsage(input_tokens=4, output_tokens=2),
            )
            client = _FakeClient(_FakeAsyncMessages(response=response))

            instrument_anthropic(client, recorder, run_id="run-async")
            result = await client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[])

            self.assertIs(result, response)
            events = _load_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["model"]["input_tokens"], 4)

    async def test_async_exception_marks_span_error_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            client = _FakeClient(_FakeAsyncMessages(error=ValueError("nope")))
            instrument_anthropic(client, recorder, run_id="run-async")

            with self.assertRaises(ValueError):
                await client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[])

            events = _load_events(path)
            self.assertEqual(events[-1]["phase"], "end")
            self.assertEqual(events[-1]["status"], "error")

    async def test_async_streaming_call_merges_usage_across_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            events_seq = [
                _message_start_event("claude-3-5-sonnet-20241022", input_tokens=15),
                _content_block_delta_event(),
                _message_delta_event(output_tokens=9),
                _message_stop_event(),
            ]
            client = _FakeClient(_FakeAsyncMessages(stream_events=events_seq))
            instrument_anthropic(client, recorder, run_id="run-async")

            stream = await client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[], stream=True)
            self.assertEqual(len(_load_events(path)), 1)

            collected = [event async for event in stream]

            self.assertEqual(len(collected), 4)
            events = _load_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["model"]["input_tokens"], 15)
            self.assertEqual(events[-1]["model"]["output_tokens"], 9)

    async def test_async_create_wrapped_like_real_sdk_is_still_detected_as_async(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(
                model="claude-3-5-sonnet-20241022",
                usage=_FakeUsage(input_tokens=7, output_tokens=3),
            )
            client = _FakeClient(_FakeAsyncMessagesSDKWrapped(response=response))

            instrument_anthropic(client, recorder, run_id="run-async-wrapped")
            # Before the inspect.unwrap fix (see integrations/openai.py for
            # the full story), this silently produced a span with no usage
            # instead of raising -- regression coverage for the real,
            # verified SDK-dispatcher-wrapping shape.
            result = await client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[])

            self.assertIs(result, response)
            events = _load_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["model"]["input_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
