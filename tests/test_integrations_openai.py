"""Tests for hermes_flight_recorder.integrations.openai.

Uses lightweight fake clients that mimic the OpenAI SDK's
``client.chat.completions.create`` shape -- no real ``openai`` dependency,
consistent with the package's zero-dependency test philosophy.
"""

from __future__ import annotations

import functools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hermes_flight_recorder import FlightRecorderSettings, HermesFlightRecorder
from hermes_flight_recorder.integrations.openai import instrument_openai


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    def model_dump(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


class _FakeSyncCompletions:
    def __init__(self, response=None, stream_chunks=None, error=None):
        self._response = response
        self._stream_chunks = stream_chunks
        self._error = error

    def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        if kwargs.get("stream"):
            return iter(self._stream_chunks)
        return self._response


class _FakeAsyncCompletions:
    def __init__(self, response=None, stream_chunks=None, error=None):
        self._response = response
        self._stream_chunks = stream_chunks
        self._error = error

    async def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        if kwargs.get("stream"):
            async def agen():
                for chunk in self._stream_chunks:
                    yield chunk

            return agen()
        return self._response


def _sdk_dispatch_wrapper(func):
    """Mimics how the real openai/anthropic SDKs wrap `create`: a plain `def`
    dispatcher (their `@required_args`-style overload resolution) around the
    actual `async def`. Calling the dispatcher returns the inner coroutine
    without awaiting it, so `inspect.iscoroutinefunction` on the *bound
    method* is False even though the real underlying implementation (and
    `__wrapped__`) is a coroutine function -- verified against the real,
    installed `openai`/`anthropic` packages, not just their type stubs.
    """

    @functools.wraps(func)
    def dispatcher(*args, **kwargs):
        return func(*args, **kwargs)

    return dispatcher


class _FakeAsyncCompletionsSDKWrapped:
    """Same behavior as _FakeAsyncCompletions, but `create` is wrapped the way
    the real SDKs wrap theirs -- regression coverage for the `inspect.unwrap`
    fix (without it, `instrument_openai` misdetects this as sync and the
    caller's `await` on the wrapped method raises TypeError)."""

    def __init__(self, response=None, stream_chunks=None, error=None):
        self._response = response
        self._stream_chunks = stream_chunks
        self._error = error

    @_sdk_dispatch_wrapper
    async def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        if kwargs.get("stream"):
            async def agen():
                for chunk in self._stream_chunks:
                    yield chunk

            return agen()
        return self._response


def _make_recorder(path: Path) -> HermesFlightRecorder:
    return HermesFlightRecorder(FlightRecorderSettings(enabled=True, path=str(path), capture_mode="metadata"))


def _load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class OpenAIInstrumentationSyncTest(unittest.TestCase):
    def test_non_streaming_call_emits_llm_call_span_with_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(model="gpt-4o-mini", usage=_FakeUsage(10, 5, 15))
            client = _FakeClient(_FakeSyncCompletions(response=response))

            instrument_openai(client, recorder, run_id="run-1")
            result = client.chat.completions.create(model="gpt-4o-mini", messages=[])

            self.assertIs(result, response)
            events = _load_events(path)
            self.assertEqual(len(events), 2)
            start, end = events
            self.assertEqual(start["event_type"], "llm.call")
            self.assertEqual(start["phase"], "start")
            self.assertEqual(end["phase"], "end")
            self.assertEqual(end["status"], "ok")
            self.assertEqual(end["model"]["input_tokens"], 10)
            self.assertEqual(end["model"]["output_tokens"], 5)
            self.assertEqual(end["model"]["name"], "gpt-4o-mini")

    def test_exception_marks_span_error_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            client = _FakeClient(_FakeSyncCompletions(error=RuntimeError("boom")))
            instrument_openai(client, recorder, run_id="run-1")

            with self.assertRaises(RuntimeError):
                client.chat.completions.create(model="gpt-4o-mini", messages=[])

            events = _load_events(path)
            self.assertEqual(events[-1]["phase"], "end")
            self.assertEqual(events[-1]["status"], "error")

    def test_streaming_call_closes_span_only_after_exhaustion_and_captures_final_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            chunks = [
                SimpleNamespace(model="gpt-4o-mini", usage=None),
                SimpleNamespace(model="gpt-4o-mini", usage=None),
                SimpleNamespace(model="gpt-4o-mini", usage=_FakeUsage(20, 8, 28)),
            ]
            client = _FakeClient(_FakeSyncCompletions(stream_chunks=chunks))
            instrument_openai(client, recorder, run_id="run-1")

            stream = client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True)
            # create() returned but the stream hasn't been consumed -- only the
            # start event should exist so far (this is the whole point: the span
            # must not close before generation actually finishes).
            self.assertEqual(len(_load_events(path)), 1)

            collected = list(stream)

            self.assertEqual(len(collected), 3)
            events = _load_events(path)
            self.assertEqual(len(events), 2)
            end = events[-1]
            self.assertEqual(end["status"], "ok")
            self.assertEqual(end["model"]["output_tokens"], 8)

    def test_streaming_call_abandoned_early_closes_span_as_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            chunks = [SimpleNamespace(model="gpt-4o-mini", usage=None) for _ in range(5)]
            client = _FakeClient(_FakeSyncCompletions(stream_chunks=chunks))
            instrument_openai(client, recorder, run_id="run-1")

            stream = client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True)
            next(stream)
            stream.close()

            events = _load_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["status"], "cancelled")

    def test_double_instrumentation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(model="gpt-4o-mini", usage=None)
            client = _FakeClient(_FakeSyncCompletions(response=response))

            instrument_openai(client, recorder, run_id="run-1")
            wrapped_once = client.chat.completions.create
            instrument_openai(client, recorder, run_id="run-1")

            self.assertIs(client.chat.completions.create, wrapped_once)

    def test_cost_fn_populates_cost_usd(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(model="gpt-4o-mini", usage=_FakeUsage(1000, 500, 1500))
            client = _FakeClient(_FakeSyncCompletions(response=response))

            def cost_fn(model_name, usage):
                return usage.get("prompt_tokens", 0) * 0.001 + usage.get("completion_tokens", 0) * 0.002

            instrument_openai(client, recorder, run_id="run-1", cost_fn=cost_fn)
            client.chat.completions.create(model="gpt-4o-mini", messages=[])

            events = _load_events(path)
            self.assertAlmostEqual(events[-1]["model"]["cost_usd"], 2.0)


class OpenAIInstrumentationAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_async_non_streaming_call_emits_llm_call_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(model="gpt-4o-mini", usage=_FakeUsage(4, 2, 6))
            client = _FakeClient(_FakeAsyncCompletions(response=response))

            instrument_openai(client, recorder, run_id="run-async")
            result = await client.chat.completions.create(model="gpt-4o-mini", messages=[])

            self.assertIs(result, response)
            events = _load_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["model"]["input_tokens"], 4)

    async def test_async_create_wrapped_like_real_sdk_is_still_detected_as_async(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            response = SimpleNamespace(model="gpt-4o-mini", usage=_FakeUsage(7, 3, 10))
            client = _FakeClient(_FakeAsyncCompletionsSDKWrapped(response=response))

            instrument_openai(client, recorder, run_id="run-async-wrapped")
            # Before the inspect.unwrap fix, this silently produced a span
            # with no usage: the adapter misdetected the SDK-dispatcher
            # wrapped coroutine as sync, so it called the sync wrapper, which
            # finalized and closed the span using the *unawaited* coroutine
            # object (before the real call ever ran) as if it were the
            # response -- no exception, just silently missing telemetry.
            result = await client.chat.completions.create(model="gpt-4o-mini", messages=[])

            self.assertIs(result, response)
            events = _load_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["model"]["input_tokens"], 7)

    async def test_async_exception_marks_span_error_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            client = _FakeClient(_FakeAsyncCompletions(error=ValueError("nope")))
            instrument_openai(client, recorder, run_id="run-async")

            with self.assertRaises(ValueError):
                await client.chat.completions.create(model="gpt-4o-mini", messages=[])

            events = _load_events(path)
            self.assertEqual(events[-1]["phase"], "end")
            self.assertEqual(events[-1]["status"], "error")

    async def test_async_streaming_call_closes_span_after_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = _make_recorder(path)
            chunks = [
                SimpleNamespace(model="gpt-4o-mini", usage=None),
                SimpleNamespace(model="gpt-4o-mini", usage=_FakeUsage(1, 1, 2)),
            ]
            client = _FakeClient(_FakeAsyncCompletions(stream_chunks=chunks))
            instrument_openai(client, recorder, run_id="run-async")

            stream = await client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True)
            self.assertEqual(len(_load_events(path)), 1)

            collected = [chunk async for chunk in stream]

            self.assertEqual(len(collected), 2)
            events = _load_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["model"]["output_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
