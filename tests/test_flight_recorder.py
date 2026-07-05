import asyncio
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hermes_flight_recorder.flight_recorder import (
    DEFAULT_DEV_HASH_SECRET,
    FlightRecorderSettings,
    HermesFlightRecorder,
    event_to_otlp_span,
    redact_value,
    resolve_hash_secret,
    trace_context_payload,
)
from hermes_flight_recorder.hermes_fr import explain_events, main as hermes_fr_main
from hermes_flight_recorder.flight_recorder_schema import validate_event_schema
from hermes_flight_recorder.flight_recorder_replay import load_fixture_events, replay_fixture
from hermes_flight_recorder.flight_recorder_query import query_index
from hermes_flight_recorder.flight_recorder_timeline import (
    load_jsonl,
    redaction_report,
    render_timeline,
    structural_report,
    timeline_summary,
)
from hermes_flight_recorder.flight_recorder_wrappers import FlightRecorderContext, file_read, file_write, process_exec


class FlightRecorderTest(unittest.TestCase):
    def test_redacts_sensitive_keys_and_bearer_tokens(self):
        value = {
            "apiKey": "sk-secret",
            "Authorization": "Bearer abc.def.ghi",
            "nested": {"password": "clear", "safe": "hello"},
        }

        redacted = redact_value(value)

        self.assertEqual(redacted["apiKey"], "[REDACTED]")
        self.assertEqual(redacted["Authorization"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["password"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["safe"], "hello")

    def test_numeric_token_counts_survive_redaction(self):
        # Integer usage counts match "token" by substring but are not sensitive.
        redacted = redact_value({
            "input_tokens": 12,
            "output_tokens": 34,
            "total_tokens": 46,
            "reasoning_tokens": 0,
            "max_tokens": 1024,
        })
        self.assertEqual(redacted["input_tokens"], 12)
        self.assertEqual(redacted["total_tokens"], 46)
        self.assertEqual(redacted["max_tokens"], 1024)

    def test_string_token_keys_still_redacted(self):
        # Fail-closed: a token-like key carrying a string is still redacted,
        # and other sensitive keys are unaffected by the count allowlist.
        redacted = redact_value({"access_token": "sk-leak", "total_tokens": "42"})
        self.assertEqual(redacted["access_token"], "[REDACTED]")
        self.assertEqual(redacted["total_tokens"], "[REDACTED]")  # string value, not a count

    def test_metadata_mode_writes_hashes_without_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(
                    enabled=True,
                    path=str(path),
                    capture_mode="metadata",
                    preview_chars=80,
                )
            )

            recorder.record(
                event_type="tool.call",
                phase="end",
                trace_id=recorder.trace_id("run-1"),
                span_id=recorder.span_id("run-1", "tool", "document_search"),
                session_id="run-1",
                turn_id="run-1",
                run_id="run-1",
                task_id="run-1",
                tool=recorder.tool_payload(
                    "document_search",
                    arguments={"question": "secret token=abc"},
                    result={"answer": "private"},
                ),
            )

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            event = json.loads(lines[0])
            self.assertEqual(event["schema_version"], "0.3.0")
            self.assertEqual(event["recorder_version"], "0.1.1")
            self.assertEqual(event["otel_mapping_version"], "0.2.0")
            # parent_event_id / openinference_mapping_version stay out of the
            # canonical JSONL envelope; OpenInference is an OTLP projection.
            self.assertNotIn("openinference_mapping_version", event)
            self.assertNotIn("parent_event_id", event)
            self.assertIn("semconv_version", event)
            self.assertIn("event_id", event)
            self.assertIn("timestamp", event)
            self.assertIn("monotonic_ns", event)
            self.assertEqual(event["event_type"], "tool.call")
            self.assertEqual(event["privacy"]["capture_mode"], "metadata")
            self.assertEqual(event["privacy"]["hash_strategy"], "hmac")
            self.assertIn("args_hmac", event["tool"])
            self.assertIn("result_hmac", event["tool"])
            self.assertTrue(event["tool"]["args_hmac"].startswith("hmac-sha256:"))
            self.assertNotIn("args_preview", event["tool"])
            self.assertNotIn("result_preview", event["tool"])
            self.assertNotIn("args_hash", event["tool"])
            self.assertNotIn("secret token=abc", lines[0])

    def test_from_env_and_record_tool_call_are_public_onboarding_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            env = {
                "FLIGHT_RECORDER_ENABLED": "true",
                "FLIGHT_RECORDER_PATH": str(path),
                "FLIGHT_RECORDER_CAPTURE_MODE": "metadata",
                "FLIGHT_RECORDER_HASH_SECRET_ENV": "FR_TEST_HMAC_KEY",
                "FR_TEST_HMAC_KEY": "x" * 32,
            }
            with patch.dict(os.environ, env, clear=False):
                recorder = HermesFlightRecorder.from_env()
                event = recorder.record_tool_call(
                    tool_name="knowledge_answer",
                    arguments={"question": "Who owns this entity?", "token": "secret"},
                    result={"answer": "redacted by metadata mode"},
                    status="ok",
                    run_id="demo",
                )

            self.assertEqual(event["event_type"], "tool.call")
            self.assertEqual(event["run_id"], "demo")
            self.assertEqual(recorder.status()["hash_secret_source"], "env")
            line = path.read_text(encoding="utf-8")
            self.assertIn("args_hmac", line)
            self.assertIn("result_hmac", line)
            self.assertNotIn("Who owns this entity?", line)
            self.assertNotIn("redacted by metadata mode", line)

    def test_span_context_manager_records_start_end_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), capture_mode="metadata")
            )

            with recorder.span(
                "tool.call",
                tool_name="get_current_weather",
                arguments={"city": "Lisbon", "api_key": "secret"},
                run_id="span-demo",
            ) as span:
                span.set_result({"temperature_c": 21})

            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["phase"] for event in events], ["start", "end"])
            self.assertEqual({event["span_id"] for event in events}, {span.span_id})
            self.assertEqual(events[1]["status"], "ok")
            self.assertIn("args_hmac", events[0]["tool"])
            self.assertIn("result_hmac", events[1]["tool"])
            self.assertNotIn("secret", path.read_text(encoding="utf-8"))

    def test_span_context_manager_records_exception_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), capture_mode="metadata")
            )

            with self.assertRaises(ValueError):
                with recorder.span("tool.call", tool_name="failing_tool", run_id="span-error"):
                    raise ValueError("boom token=secret")

            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["phase"], "end")
            self.assertEqual(events[-1]["status"], "error")
            self.assertIn("error_hmac", events[-1]["tool"])
            self.assertNotIn("token=secret", path.read_text(encoding="utf-8"))

    def test_trace_tool_call_decorator_records_sync_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), capture_mode="metadata")
            )

            @recorder.trace_tool_call("lookup_account", run_id="decorator-demo")
            def lookup_account(account_id: str) -> dict[str, str]:
                return {"account_id": account_id, "tier": "gold"}

            result = lookup_account("acct-secret")

            self.assertEqual(result["tier"], "gold")
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["phase"] for event in events], ["start", "end"])
            self.assertEqual(events[0]["tool"]["name"], "lookup_account")
            self.assertIn("args_hmac", events[0]["tool"])
            self.assertIn("result_hmac", events[1]["tool"])
            self.assertNotIn("acct-secret", path.read_text(encoding="utf-8"))

    def test_async_span_and_decorator_record_without_blocking_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), capture_mode="metadata")
            )

            @recorder.trace_tool_call("async_lookup", run_id="async-demo")
            async def async_lookup(value: str) -> dict[str, str]:
                await asyncio.sleep(0)
                return {"value": value}

            async def drive():
                async with recorder.aspan(
                    "llm.call",
                    run_id="async-demo",
                    model=recorder.model_payload(provider="example", name="example-model"),
                ) as span:
                    span.set_model(
                        recorder.model_payload(
                            provider="example",
                            name="example-model",
                            usage={"input_tokens": 3, "output_tokens": 5},
                        )
                    )
                return await async_lookup("private-value")

            result = asyncio.run(drive())

            self.assertEqual(result["value"], "private-value")
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event_type"] for event in events], ["llm.call", "llm.call", "tool.call", "tool.call"])
            self.assertEqual(events[1]["model"]["input_tokens"], 3)
            self.assertIn("result_hmac", events[-1]["tool"])
            self.assertNotIn("private-value", path.read_text(encoding="utf-8"))

    def test_package_ships_pep561_marker(self):
        import hermes_flight_recorder

        marker = Path(hermes_flight_recorder.__file__).with_name("py.typed")
        self.assertTrue(marker.exists())

    def test_preview_mode_redacts_and_truncates_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(
                    enabled=True,
                    path=str(path),
                    capture_mode="preview",
                    preview_chars=90,
                )
            )

            recorder.record(
                event_type="tool.call",
                phase="start",
                trace_id=recorder.trace_id("run-2"),
                span_id=recorder.span_id("run-2", "tool"),
                tool=recorder.tool_payload(
                    "knowledge_answer",
                    arguments={"aaa_token": "abc", "question": "Contact", "long": "x" * 200},
                ),
            )

            event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("args_preview", event["tool"])
            self.assertIn("[REDACTED]", event["tool"]["args_preview"])
            self.assertIn("[truncated]", event["tool"]["args_preview"])
            self.assertNotIn("abc", event["tool"]["args_preview"])

    def test_disabled_recorder_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=False, path=str(path))
            )

            recorder.record(
                event_type="hermes.session",
                phase="start",
                trace_id=recorder.trace_id("run-3"),
                span_id=recorder.span_id("run-3", "session"),
            )

            self.assertFalse(path.exists())

    def test_end_event_timestamp_uses_end_ts_for_timeline_ordering(self):
        recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=False, capture_mode="metadata"))

        event = recorder.record(
            event_type="hermes.turn",
            phase="end",
            trace_id=recorder.trace_id("timeline-order"),
            span_id=recorder.span_id("timeline-order", "turn"),
            session_id="timeline-order",
            start_ts="2026-06-20T09:14:25.052Z",
            end_ts="2026-06-20T09:14:27.044Z",
            status="ok",
        )

        self.assertEqual(event["timestamp"], "2026-06-20T09:14:27.044Z")

    def test_full_mode_keeps_redacted_full_payload_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), capture_mode="full")
            )

            recorder.record(
                event_type="tool.call",
                phase="end",
                trace_id=recorder.trace_id("run-full"),
                span_id=recorder.span_id("run-full", "tool"),
                tool=recorder.tool_payload(
                    "document_search",
                    arguments={"question": "Contact", "apiKey": "secret"},
                    result={"answer": "complete local answer"},
                ),
            )

            event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(event["tool"]["args_full"]["question"], "Contact")
            self.assertEqual(event["tool"]["args_full"]["apiKey"], "[REDACTED]")
            self.assertEqual(event["tool"]["result_full"]["answer"], "complete local answer")

    def test_forensic_mode_adds_hash_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), capture_mode="forensic")
            )

            recorder.record(
                event_type="hermes.session",
                phase="start",
                trace_id=recorder.trace_id("run-4"),
                span_id=recorder.span_id("run-4", "session"),
            )
            recorder.record(
                event_type="hermes.session",
                phase="end",
                trace_id=recorder.trace_id("run-4"),
                span_id=recorder.span_id("run-4", "session"),
            )

            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertIsNone(events[0]["integrity"]["previous_event_hash"])
            self.assertEqual(events[0]["integrity"]["hash_algorithm"], "hmac-sha256")
            self.assertEqual(events[0]["integrity"]["hash_scope"], "canonical_redacted_event_v1")
            self.assertEqual(events[1]["integrity"]["previous_event_hash"], events[0]["integrity"]["event_hash"])
            self.assertTrue(events[0]["integrity"]["event_hash"].startswith("hmac-sha256:"))

    def test_otlp_span_excludes_previews_and_full_payload_by_default(self):
        recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=False, capture_mode="full"))
        event = recorder.record(
            event_type="tool.call",
            phase="end",
            trace_id=recorder.trace_id("run-otlp"),
            span_id=recorder.span_id("run-otlp", "tool"),
            parent_span_id=recorder.span_id("run-otlp", "turn"),
            session_id="run-otlp",
            run_id="run-otlp",
            task_id="run-otlp",
            start_ts="2026-06-20T08:00:00.000Z",
            end_ts="2026-06-20T08:00:00.100Z",
            tool=recorder.tool_payload(
                "document_search",
                arguments={"question": "Contact"},
                result={"answer": "private answer"},
                category="mcp",
            ),
        )

        span = event_to_otlp_span(event)
        serialized = json.dumps(span)

        self.assertEqual(span["name"], "tool.call document_search")
        self.assertEqual(span["traceId"], recorder.trace_id("run-otlp"))
        self.assertEqual(span["parentSpanId"], recorder.span_id("run-otlp", "turn"))
        self.assertIn("hermes.tool.args_hmac", serialized)
        self.assertIn("hermes.tool.result_hmac", serialized)
        self.assertIn("gen_ai.operation.name", serialized)
        self.assertIn("gen_ai.tool.name", serialized)
        self.assertIn("mcp.method.name", serialized)
        self.assertIn("openinference.span.kind", serialized)
        self.assertIn("TOOL", serialized)
        self.assertIn("w3c.traceparent", serialized)
        self.assertNotIn("private answer", serialized)
        self.assertNotIn("args_full", serialized)
        self.assertNotIn("result_full", serialized)
        self.assertNotIn("args_preview", serialized)

    def test_trace_context_payload_builds_w3c_traceparent(self):
        payload = trace_context_payload("a" * 32, "b" * 16, tracestate="vendor=1")

        self.assertEqual(payload["traceparent"], f"00-{'a' * 32}-{'b' * 16}-01")
        self.assertEqual(payload["tracestate"], "vendor=1")
        self.assertNotIn("baggage", payload)

    def test_otlp_span_can_include_redacted_preview_when_explicit(self):
        recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=False, capture_mode="preview"))
        event = recorder.record(
            event_type="tool.call",
            phase="start",
            trace_id=recorder.trace_id("run-preview"),
            span_id=recorder.span_id("run-preview", "tool"),
            tool=recorder.tool_payload("document_search", arguments={"question": "Contact", "token": "secret"}),
        )

        span = event_to_otlp_span(event, include_previews=True)
        serialized = json.dumps(span)

        self.assertIn("hermes.tool.args_preview", serialized)
        self.assertIn("[REDACTED]", serialized)
        self.assertNotIn("secret", serialized)

    def test_flush_otlp_posts_payload_and_clears_buffer(self):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, headers, json):
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return FakeResponse()

        recorder = HermesFlightRecorder(
            FlightRecorderSettings(
                enabled=True,
                otlp_enabled=True,
                otlp_endpoint="http://collector:4318/v1/traces",
                otlp_headers="Authorization=Basic test,x-test=yes",
            )
        )
        recorder.record(
            event_type="hermes.session",
            phase="start",
            trace_id=recorder.trace_id("run-flush"),
            span_id=recorder.span_id("run-flush", "session"),
            session_id="run-flush",
        )

        result = asyncio.run(recorder.flush_otlp(http_client_factory=lambda: FakeClient()))

        self.assertEqual(result["exported"], 1)
        self.assertEqual(captured["url"], "http://collector:4318/v1/traces")
        self.assertEqual(captured["headers"]["Authorization"], "Basic test")
        self.assertEqual(captured["headers"]["x-test"], "yes")
        self.assertEqual(recorder.status()["otlp_buffered_events"], 0)
        self.assertIn("resourceSpans", captured["json"])

    def test_otlp_flush_active_gates_on_enabled_and_endpoint(self):
        # Disabled recorder must never spawn the background flush loop.
        off = HermesFlightRecorder(
            FlightRecorderSettings(enabled=False, otlp_enabled=True, otlp_endpoint="http://c:4318/v1/traces")
        )
        self.assertFalse(off.otlp_flush_active())
        # Enabled but no endpoint configured â†’ nothing to export, no loop.
        no_endpoint = HermesFlightRecorder(
            FlightRecorderSettings(enabled=True, otlp_enabled=True, otlp_endpoint="")
        )
        self.assertFalse(no_endpoint.otlp_flush_active())
        # Enabled + OTLP + endpoint â†’ loop should run.
        active = HermesFlightRecorder(
            FlightRecorderSettings(enabled=True, otlp_enabled=True, otlp_endpoint="http://c:4318/v1/traces")
        )
        self.assertTrue(active.otlp_flush_active())

    def test_otlp_flush_loop_drains_buffer_until_cancelled(self):
        # The loop must export buffered spans on its cadence (not only at task
        # end) and stop cleanly on cancellation. This is the fix for buffer
        # overflow drops under long/concurrent runs.
        posts = {"count": 0}

        class FakeResponse:
            def raise_for_status(self):
                return None

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, headers, json):
                posts["count"] += 1
                return FakeResponse()

        recorder = HermesFlightRecorder(
            FlightRecorderSettings(
                enabled=True,
                otlp_enabled=True,
                otlp_endpoint="http://collector:4318/v1/traces",
                otlp_flush_interval_seconds=0.1,
            )
        )
        recorder.record(
            event_type="hermes.session",
            phase="start",
            trace_id=recorder.trace_id("run-loop"),
            span_id=recorder.span_id("run-loop", "session"),
            session_id="run-loop",
        )

        async def drive():
            task = asyncio.create_task(
                recorder.otlp_flush_loop(http_client_factory=lambda: FakeClient())
            )
            await asyncio.sleep(0.3)  # allow a couple of flush ticks
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(drive())

        self.assertGreaterEqual(posts["count"], 1)
        self.assertEqual(recorder.status()["otlp_buffered_events"], 0)
        self.assertEqual(recorder.status()["otlp_flush_interval_seconds"], 0.1)

    def test_replay_loads_json_and_jsonl_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "fixture.json"
            jsonl_path = Path(tmp) / "fixture.jsonl"
            json_path.write_text(
                json.dumps(
                    {
                        "run_id": "fixture-json",
                        "events": [
                            {"event_type": "hermes.session"},
                            {"event_type": "tool.call", "tool_name": "document_search"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            jsonl_path.write_text(
                "\n".join([
                    json.dumps({"event_type": "hermes.turn"}),
                    json.dumps({"event_type": "llm.call", "model_name": "test-model"}),
                ]),
                encoding="utf-8",
            )

            self.assertEqual([event["event_type"] for event in load_fixture_events(json_path)], ["hermes.session", "tool.call"])
            self.assertEqual([event["event_type"] for event in load_fixture_events(jsonl_path)], ["hermes.turn", "llm.call"])

    def test_replay_fixture_writes_metadata_jsonl_without_raw_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            output_path = Path(tmp) / "events.jsonl"
            fixture_path.write_text(
                json.dumps(
                    {
                        "run_id": "fixture-secret",
                        "events": [
                            {
                                "event_type": "tool.call",
                                "tool_name": "document_search",
                                "arguments": {"question": "Contact", "api_key": "secret-value"},
                                "result": {"answer": "ok token=abc"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = replay_fixture(input_path=fixture_path, output_path=output_path, capture_mode="metadata")

            self.assertEqual(summary["events"], 1)
            line = output_path.read_text(encoding="utf-8")
            event = json.loads(line)
            self.assertEqual(event["event_type"], "tool.call")
            self.assertIn("args_hmac", event["tool"])
            self.assertIn("result_hmac", event["tool"])
            self.assertNotIn("args_preview", event["tool"])
            self.assertNotIn("result_preview", event["tool"])
            self.assertNotIn("secret-value", line)
            self.assertNotIn("token=abc", line)

    def test_replay_fixture_writes_otlp_payload_without_full_content_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            output_path = Path(tmp) / "events.jsonl"
            otlp_path = Path(tmp) / "otlp.json"
            fixture_path.write_text(
                json.dumps(
                    {
                        "run_id": "fixture-otlp",
                        "events": [
                            {
                                "event_type": "tool.call",
                                "phase": "end",
                                "tool_name": "document_search",
                                "arguments": {"question": "Contact"},
                                "result": {"answer": "private local answer"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            replay_fixture(
                input_path=fixture_path,
                output_path=output_path,
                capture_mode="full",
                otlp_payload_path=otlp_path,
            )

            local_jsonl = output_path.read_text(encoding="utf-8")
            otlp_json = otlp_path.read_text(encoding="utf-8")

            self.assertIn("private local answer", local_jsonl)
            self.assertIn("hermes.tool.result_hmac", otlp_json)
            self.assertNotIn("private local answer", otlp_json)
            self.assertNotIn("result_full", otlp_json)

    def test_side_effect_payload_metadata_mode_hashes_content_without_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), capture_mode="metadata")
            )

            recorder.record(
                event_type="tool.call",
                phase="end",
                trace_id=recorder.trace_id("run-side-effect"),
                span_id=recorder.span_id("run-side-effect", "tool"),
                side_effects=[
                    recorder.side_effect_payload(
                        "file.write",
                        target="/workspace/private/report.md",
                        before="old secret=abc",
                        after="new content",
                        metadata={"bytes": 11},
                    )
                ],
            )

            serialized = path.read_text(encoding="utf-8")
            event = json.loads(serialized)
            effect = event["side_effects"][0]

            self.assertEqual(effect["type"], "file.write")
            self.assertIn("target_hmac", effect)
            self.assertIn("before_hmac", effect)
            self.assertIn("after_hmac", effect)
            self.assertEqual(effect["metadata"]["bytes"], 11)
            self.assertNotIn("target_preview", effect)
            self.assertNotIn("/workspace/private/report.md", serialized)
            self.assertNotIn("secret=abc", serialized)

    def test_side_effects_export_to_otlp_as_metadata_only(self):
        recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=False, capture_mode="full"))
        event = recorder.record(
            event_type="tool.call",
            phase="end",
            trace_id=recorder.trace_id("run-side-effect-otlp"),
            span_id=recorder.span_id("run-side-effect-otlp", "tool"),
            side_effects=[
                recorder.side_effect_payload(
                    "process.exec",
                    command="curl https://example.test?token=secret",
                    stdout="private output",
                    metadata={"exit_code": 0},
                )
            ],
        )

        span = event_to_otlp_span(event)
        serialized = json.dumps(span)

        self.assertIn("hermes.side_effect.count", serialized)
        self.assertIn("hermes.side_effect.0.command_hmac", serialized)
        self.assertIn("hermes.side_effect.0.metadata.exit_code", serialized)
        self.assertNotIn("private output", serialized)
        self.assertNotIn("curl https://example.test", serialized)
        self.assertNotIn("stdout_full", serialized)

    def test_replay_fixture_accepts_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            output_path = Path(tmp) / "events.jsonl"
            otlp_path = Path(tmp) / "otlp.json"
            fixture_path.write_text(
                json.dumps(
                    {
                        "run_id": "fixture-side-effect",
                        "events": [
                            {
                                "event_type": "tool.call",
                                "tool_name": "document_search",
                                "side_effects": [
                                    {
                                        "type": "file.write",
                                        "path": "/workspace/private/report.md",
                                        "after": "confidential content",
                                        "bytes": 20,
                                    },
                                    {
                                        "type": "process.exec",
                                        "command": "echo token=secret",
                                        "stdout": "token=secret",
                                        "exit_code": 0,
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            replay_fixture(
                input_path=fixture_path,
                output_path=output_path,
                capture_mode="metadata",
                otlp_payload_path=otlp_path,
            )

            local_jsonl = output_path.read_text(encoding="utf-8")
            otlp_json = otlp_path.read_text(encoding="utf-8")
            event = json.loads(local_jsonl)

            self.assertEqual(len(event["side_effects"]), 2)
            self.assertIn("target_hmac", event["side_effects"][0])
            self.assertIn("command_hmac", event["side_effects"][1])
            self.assertNotIn("/workspace/private/report.md", local_jsonl)
            self.assertNotIn("token=secret", local_jsonl)
            self.assertIn("hermes.side_effect.count", otlp_json)
            self.assertNotIn("confidential content", otlp_json)
            self.assertNotIn("echo token=secret", otlp_json)

    def test_runtime_policy_payload_metadata_mode_hashes_sensitive_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), capture_mode="metadata")
            )

            recorder.record(
                event_type="runtime.policy.deny",
                phase="instant",
                trace_id=recorder.trace_id("run-policy"),
                span_id=recorder.span_id("run-policy", "policy"),
                status="blocked",
                actor="runtime",
                runtime=recorder.runtime_policy_payload(
                    decision="denied",
                    policy_type="egress",
                    target="https://unknown.example.test/path?token=secret",
                    sandbox_id="sandbox-private-1",
                    policy_decision_id="policy-decision-1",
                    policy_version="policy-version-private",
                    secret_withheld=True,
                    correlation="matched",
                    metadata={"reason": "deny_by_default", "api_key": "secret"},
                ),
            )

            serialized = path.read_text(encoding="utf-8")
            event = json.loads(serialized)
            runtime = event["runtime"]

            self.assertEqual(runtime["decision"], "denied")
            self.assertEqual(runtime["egress_decision"], "denied")
            self.assertTrue(runtime["secret_withheld"])
            self.assertEqual(runtime["metadata"]["api_key"], "[REDACTED]")
            self.assertIn("target_hmac", runtime)
            self.assertIn("sandbox_id_hmac", runtime)
            self.assertIn("policy_decision_id_hmac", runtime)
            self.assertNotIn("unknown.example.test", serialized)
            self.assertNotIn("sandbox-private-1", serialized)
            self.assertNotIn("policy-decision-1", serialized)
            self.assertNotIn("token=secret", serialized)

    def test_runtime_policy_exports_to_otlp_as_metadata_only(self):
        recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=False, capture_mode="full"))
        event = recorder.record(
            event_type="runtime.policy.allow",
            phase="instant",
            trace_id=recorder.trace_id("run-policy-otlp"),
            span_id=recorder.span_id("run-policy-otlp", "policy"),
            actor="runtime",
            runtime=recorder.runtime_policy_payload(
                decision="allowed",
                policy_type="egress",
                target="https://backend.internal/api?token=secret",
                sandbox_id="sandbox-private-2",
                metadata={"reason": "explicit_allow"},
            ),
        )

        span = event_to_otlp_span(event)
        serialized = json.dumps(span)

        self.assertIn("hermes.runtime.decision", serialized)
        self.assertIn("hermes.runtime.egress_decision", serialized)
        self.assertIn("hermes.runtime.target_hmac", serialized)
        self.assertIn("hermes.runtime.metadata.reason", serialized)
        self.assertNotIn("backend.internal", serialized)
        self.assertNotIn("sandbox-private-2", serialized)
        self.assertNotIn("target_full", serialized)

    def test_replay_fixture_accepts_runtime_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            output_path = Path(tmp) / "events.jsonl"
            otlp_path = Path(tmp) / "otlp.json"
            fixture_path.write_text(
                json.dumps(
                    {
                        "run_id": "fixture-policy",
                        "events": [
                            {
                                "event_type": "runtime.policy.deny",
                                "status": "blocked",
                                "actor": "runtime",
                                "runtime_policy": {
                                    "decision": "denied",
                                    "policy_type": "egress",
                                    "target": "https://unknown-tracker.example?token=secret",
                                    "sandbox_id": "sandbox-private-3",
                                    "secret_withheld": True,
                                    "correlation": "matched",
                                    "metadata": {"reason": "deny_by_default"},
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            replay_fixture(
                input_path=fixture_path,
                output_path=output_path,
                capture_mode="metadata",
                otlp_payload_path=otlp_path,
            )

            local_jsonl = output_path.read_text(encoding="utf-8")
            otlp_json = otlp_path.read_text(encoding="utf-8")
            event = json.loads(local_jsonl)

            self.assertEqual(event["runtime"]["decision"], "denied")
            self.assertEqual(event["runtime"]["correlation"], "matched")
            self.assertIn("target_hmac", event["runtime"])
            self.assertNotIn("unknown-tracker.example", local_jsonl)
            self.assertNotIn("token=secret", local_jsonl)
            self.assertIn("hermes.runtime.target_hmac", otlp_json)
            self.assertNotIn("unknown-tracker.example", otlp_json)

    def test_timeline_redaction_report_passes_metadata_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            output_path = Path(tmp) / "events.jsonl"
            fixture_path.write_text(
                json.dumps(
                    {
                        "run_id": "fixture-redaction",
                        "events": [
                            {
                                "event_type": "tool.call",
                                "tool_name": "document_search",
                                "arguments": {
                                    "question": "my password is hunter2",
                                    "url": "https://example.com?token=abc123",
                                    "headers": {"Authorization": "Bearer eyJhbGciOi"},
                                    "path": "/home/alice/.ssh/id_rsa",
                                },
                                "result": {
                                    "email": "alice@example.com",
                                    "cookie": "sessionid=secret",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            replay_fixture(input_path=fixture_path, output_path=output_path, capture_mode="metadata")
            events = load_jsonl(output_path)
            report = redaction_report(events)

            self.assertEqual(report["raw_payload_fields"], 0)
            self.assertEqual(report["preview_fields"], 0)
            self.assertEqual(report["possible_secret_patterns"], 0)
            serialized = json.dumps(events)
            self.assertIn("args_hmac", serialized)
            self.assertNotIn("hunter2", serialized)
            self.assertNotIn("alice@example.com", serialized)
            self.assertNotIn("/home/alice", serialized)

    def test_timeline_renders_tool_status(self):
        recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=False, capture_mode="metadata"))
        event = recorder.record(
            event_type="tool.call",
            phase="end",
            trace_id=recorder.trace_id("timeline"),
            span_id=recorder.span_id("timeline", "tool"),
            session_id="timeline",
            status="blocked",
            duration_ms=2,
            tool=recorder.tool_payload("mutating_write"),
        )
        args = type("Args", (), {
            "session": None,
            "turn": None,
            "show_errors": True,
            "show_policy": True,
            "show_side_effects": True,
            "show_hashes": False,
            "show_previews": False,
        })()

        output = render_timeline([event], args)

        self.assertIn("tool.call end blocked 2ms", output)
        self.assertIn("tool=mutating_write", output)

    def test_schema_validation_and_live_index_are_status_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            index_path = Path(tmp) / "events.sqlite3"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(
                    enabled=True,
                    path=str(path),
                    index_enabled=True,
                    index_path=str(index_path),
                    schema_validation_enabled=True,
                )
            )

            recorder.record(
                event_type="tool.call",
                phase="end",
                trace_id=recorder.trace_id("indexed"),
                span_id=recorder.span_id("indexed", "tool"),
                run_id="indexed",
                status="ok",
                duration_ms=7,
                tool=recorder.tool_payload("document_search", arguments={"question": "Contact"}),
            )

            status = recorder.status()
            self.assertTrue(status["index_enabled"])
            self.assertFalse(status["index_failed"])
            self.assertEqual(status["schema_invalid_events"], 0)
            self.assertTrue(index_path.exists())
            indexed_events = query_index(index_path, {"tool_name": "document_search"}, limit=10)
            self.assertEqual(len(indexed_events), 1)
            self.assertEqual(indexed_events[0]["event_type"], "tool.call")

    def test_schema_validation_fail_open_records_invalid_count(self):
        recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=False, schema_validation_enabled=True))

        recorder.record(
            event_type="unknown.event",
            phase="instant",
            trace_id=recorder.trace_id("invalid"),
            span_id=recorder.span_id("invalid", "event"),
        )

        status = recorder.status()
        self.assertEqual(status["schema_invalid_events"], 1)
        self.assertIn("event_type_unknown", status["last_schema_errors"])

    def test_sampling_preserves_error_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), sample_rate=0.0)
            )

            recorder.record(
                event_type="hermes.session",
                phase="start",
                trace_id=recorder.trace_id("sample"),
                span_id=recorder.span_id("sample", "ok"),
            )
            recorder.record(
                event_type="hermes.session",
                phase="end",
                trace_id=recorder.trace_id("sample"),
                span_id=recorder.span_id("sample", "error"),
                status="error",
            )

            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["status"], "error")
            self.assertEqual(recorder.status()["sampled_events"], 1)

    def test_async_writer_flushes_and_reports_queue_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(
                    enabled=True,
                    path=str(path),
                    async_writes_enabled=True,
                    write_queue_max_events=10,
                )
            )

            recorder.record(
                event_type="hermes.session",
                phase="start",
                trace_id=recorder.trace_id("async"),
                span_id=recorder.span_id("async", "session"),
            )
            flushed = recorder.flush_writes(timeout_seconds=2)

            self.assertTrue(flushed["settled"])
            self.assertEqual(recorder.status()["write_queue_depth"], 0)
            self.assertEqual(recorder.status()["write_queue_dropped_events"], 0)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_rotation_and_retention_keep_bounded_rotated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(
                    enabled=True,
                    path=str(path),
                    rotate_bytes=450,
                    retention_files=1,
                )
            )

            for index in range(4):
                recorder.record(
                    event_type="tool.call",
                    phase="end",
                    trace_id=recorder.trace_id("rotate"),
                    span_id=recorder.span_id("rotate", "tool", index),
                    run_id="rotate",
                    tool=recorder.tool_payload("document_search", result={"answer": "x" * 200}),
                )

            status = recorder.status()
            rotated_files = list(Path(tmp).glob("events.*.jsonl"))
            self.assertGreaterEqual(status["rotations"], 1)
            self.assertGreaterEqual(status["current_file_bytes"], 0)
            self.assertEqual(status["rotated_files"], len(rotated_files))
            self.assertLessEqual(len(rotated_files), 1)
            self.assertFalse(status["retention_failed"])

    def test_max_event_bytes_writes_compact_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), capture_mode="full", max_event_bytes=1200)
            )

            recorder.record(
                event_type="tool.call",
                phase="end",
                trace_id=recorder.trace_id("oversize"),
                span_id=recorder.span_id("oversize", "tool"),
                tool=recorder.tool_payload("document_search", result={"answer": "x" * 2000}),
            )

            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("tool", event)
            self.assertIn("oversize_original_bytes", event["attributes"])
            self.assertEqual(recorder.status()["oversize_events"], 1)

    def test_timeline_summary_and_structural_report_are_available(self):
        recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=False))
        events = [
            recorder.record(
                event_type="hermes.session",
                phase="start",
                trace_id=recorder.trace_id("summary"),
                span_id=recorder.span_id("summary", "session"),
                status="ok",
            ),
            recorder.record(
                event_type="tool.call",
                phase="end",
                trace_id=recorder.trace_id("summary"),
                span_id=recorder.span_id("summary", "tool"),
                status="blocked",
                tool=recorder.tool_payload("plan_action"),
            ),
        ]

        summary = timeline_summary(events)
        structure = structural_report(events)

        self.assertEqual(summary["events"], 2)
        self.assertEqual(summary["statuses"]["blocked"], 1)
        self.assertEqual(summary["tools"]["plan_action"], 1)
        self.assertEqual(structure["invalid_events"], 0)
        self.assertEqual(structure["unknown_event_type_events"], 0)

    def test_explain_events_reports_blocked_tools_privacy_and_usage(self):
        recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=False))
        events = [
            recorder.record_tool_call(
                tool_name="mutating_write",
                status="blocked",
                run_id="explain",
                duration_ms=5,
                attributes={"retry_count": 2},
            ),
            recorder.record(
                event_type="llm.call",
                phase="end",
                trace_id=recorder.trace_id("explain"),
                span_id=recorder.span_id("explain", "llm"),
                run_id="explain",
                model={"name": "test-model", "provider": "local", "input_tokens": 10, "output_tokens": 3, "cost_usd": 0.01},
            ),
        ]

        report = explain_events(events)

        self.assertEqual(report["likely_cause"], "policy or capability block recorded")
        self.assertEqual(report["blocked_tools"], ["mutating_write"])
        self.assertEqual(report["retry_count"], 2)
        self.assertEqual(report["llm"]["cost_usd"], 0.01)
        self.assertEqual(report["privacy"]["raw_payload_fields"], 0)

    def test_hermes_fr_cli_replay_timeline_explain_redact_and_doctor(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            output_path = Path(tmp) / "events.jsonl"
            otlp_path = Path(tmp) / "otlp.json"
            fixture_path.write_text(
                json.dumps(
                    {
                        "run_id": "cli-demo",
                        "events": [
                            {
                                "event_type": "tool.call",
                                "tool_name": "knowledge_answer",
                                "arguments": {"question": "Contact", "token": "secret"},
                                "result": {"answer": "private"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                replay_code = hermes_fr_main(
                    ["replay", "--input", str(fixture_path), "--output", str(output_path), "--capture-mode", "metadata"]
                )
            self.assertEqual(replay_code, 0)
            self.assertTrue(output_path.exists())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                timeline_code = hermes_fr_main(["timeline", str(output_path), "--show-errors"])
            self.assertEqual(timeline_code, 0)
            self.assertIn("tool=knowledge_answer", stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                explain_code = hermes_fr_main(["explain", str(output_path)])
            self.assertEqual(explain_code, 0)
            self.assertIn("Runs:", stdout.getvalue())
            self.assertIn("cli-demo", stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                explain_run_code = hermes_fr_main(["explain", str(output_path), "--run", "cli-demo"])
            self.assertEqual(explain_run_code, 0)
            self.assertIn("Status: completed", stdout.getvalue())
            self.assertIn("Run: cli-demo", stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                redact_code = hermes_fr_main(["redact-check", str(output_path)])
            self.assertEqual(redact_code, 0)
            self.assertIn("possible secret patterns: 0", stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                doctor_code = hermes_fr_main(
                    ["doctor", "--check-privacy", str(output_path), "--check-otlp", "--events", str(output_path), "--payload-out", str(otlp_path)]
                )
            self.assertEqual(doctor_code, 0)
            self.assertTrue(otlp_path.exists())
            self.assertNotIn("_full", otlp_path.read_text(encoding="utf-8"))

    def test_hermes_fr_doctor_secret_uses_from_env(self):
        with patch.dict(
            os.environ,
            {"FLIGHT_RECORDER_HASH_SECRET_ENV": "FR_TEST_HMAC_KEY", "FR_TEST_HMAC_KEY": "x" * 32},
            clear=False,
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = hermes_fr_main(["doctor", "--check-secret"])

        self.assertEqual(code, 0)
        self.assertIn("hash_secret_source=env", stdout.getvalue())

    def test_hermes_fr_version_reports_recorder_version(self):
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            hermes_fr_main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("hermes-flight-recorder 0.1.1", stdout.getvalue())

    def test_flush_otlp_without_httpx_is_fail_open_with_install_hint(self):
        recorder = HermesFlightRecorder(
            FlightRecorderSettings(enabled=True, otlp_enabled=True, otlp_endpoint="http://collector:4318/v1/traces")
        )
        recorder.record(
            event_type="hermes.session",
            phase="start",
            trace_id=recorder.trace_id("otlp-missing-httpx"),
            span_id=recorder.span_id("otlp-missing-httpx", "session"),
            session_id="otlp-missing-httpx",
        )

        with patch.dict(sys.modules, {"httpx": None}):
            result = asyncio.run(recorder.flush_otlp())

        self.assertEqual(result["exported"], 0)
        self.assertEqual(result["error_type"], "ModuleNotFoundError")
        self.assertIn("pip install hermes-flight-recorder[otlp]", result["error"])
        self.assertTrue(recorder.status()["otlp_failed"])
        self.assertEqual(recorder.status()["otlp_buffered_events"], 1)

    def test_structural_report_is_fail_open_on_unknown_event_type(self):
        # Reproduces the P0 canary failure class: events produced by a newer
        # recorder image carry an event_type not yet in this checkout's allowlist
        # (e.g. memory.read seen by a stale offline schema). The offline analyzer
        # must stay fail-open like the live recorder: the unknown type is a
        # forward-compat warning, not a hard violation that fails the gate.
        recorder = HermesFlightRecorder(FlightRecorderSettings(enabled=False))
        valid = recorder.record(
            event_type="hermes.session",
            phase="start",
            trace_id=recorder.trace_id("unknown"),
            span_id=recorder.span_id("unknown", "session"),
            status="ok",
        )
        forward = dict(valid)
        forward["event_type"] = "memory.future_kind"

        report = structural_report([valid, forward])

        self.assertEqual(report["invalid_events"], 0)
        self.assertEqual(report["schema_violations"], [])
        self.assertEqual(report["unknown_event_type_events"], 1)
        self.assertEqual(report["unknown_event_types"][0]["event_type"], "memory.future_kind")

        # A genuinely malformed event (bad trace_id) is still a hard violation.
        malformed = dict(valid)
        malformed["trace_id"] = "not-hex"
        mixed = structural_report([forward, malformed])
        self.assertEqual(mixed["invalid_events"], 1)
        self.assertEqual(mixed["unknown_event_type_events"], 1)

    def test_sqlite_index_self_heals_when_db_file_removed(self):
        # The SQLite file is local/disposable and can be removed under a
        # long-lived recorder (disk cleanup, rotation/retention, a canary clear
        # step). Indexing must self-heal instead of failing "no such table" for
        # the rest of the process lifetime (root cause of a P0 index canary gate
        # failure on staging-7d09f7efaa63).
        from hermes_flight_recorder.flight_recorder_index import FlightRecorderIndex

        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "events.sqlite3"
            index = FlightRecorderIndex(index_path)  # _ensure_schema at construction
            event = {
                "event_id": "e1",
                "event_type": "tool.call",
                "phase": "end",
                "status": "ok",
                "trace_id": "t",
                "span_id": "s",
            }
            index.index_event(event, jsonl_path="x.jsonl", jsonl_offset=0, jsonl_bytes=10)

            index_path.unlink()  # DB removed out from under the recorder
            # Must not raise: schema re-ensured + write retried once.
            event["event_id"] = "e2"
            index.index_event(event, jsonl_path="x.jsonl", jsonl_offset=10, jsonl_bytes=10)

            self.assertTrue(index_path.exists())

    def test_file_wrappers_record_metadata_without_raw_paths_or_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "events.jsonl"
            target_path = Path(tmp) / "private" / "report.txt"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(log_path), capture_mode="metadata")
            )
            context = FlightRecorderContext(run_id="wrapper-file", trace_id=recorder.trace_id("wrapper-file"))

            write_event = file_write(recorder, target_path, "token=secret", context=context)
            content = file_read(recorder, target_path, context=context)

            serialized = log_path.read_text(encoding="utf-8")
            events = [json.loads(line) for line in serialized.splitlines()]
            self.assertEqual(write_event["event_type"], "file.write")
            self.assertEqual(content, "token=secret")
            self.assertEqual([event["event_type"] for event in events], ["file.write", "file.read"])
            self.assertIn("target_hmac", events[0]["side_effects"][0])
            self.assertIn("after_hmac", events[1]["side_effects"][0])
            self.assertNotIn(str(target_path), serialized)
            self.assertNotIn("token=secret", serialized)

    def test_process_wrapper_records_metadata_without_raw_command_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(log_path), capture_mode="metadata")
            )

            result = process_exec(
                recorder,
                [sys.executable, "-c", "print('token=secret')"],
                context={"run_id": "wrapper-process", "trace_id": recorder.trace_id("wrapper-process")},
            )

            serialized = log_path.read_text(encoding="utf-8")
            event = json.loads(serialized)
            effect = event["side_effects"][0]
            self.assertEqual(result.returncode, 0)
            self.assertEqual(event["event_type"], "process.exec")
            self.assertIn("command_hmac", effect)
            self.assertIn("stdout_hmac", effect)
            self.assertEqual(effect["metadata"]["exit_code"], 0)
            self.assertNotIn("token=secret", serialized)
            self.assertNotIn(sys.executable, serialized)

    def _concurrent_write(self, *, async_writes: bool) -> None:
        threads_count = 8
        events_per_thread = 40
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(
                    enabled=True,
                    path=str(path),
                    capture_mode="metadata",
                    async_writes_enabled=async_writes,
                    write_queue_max_events=threads_count * events_per_thread,
                )
            )
            barrier = threading.Barrier(threads_count)

            def worker(worker_id: int) -> None:
                barrier.wait()
                for index in range(events_per_thread):
                    run_id = f"concurrent-{worker_id}"
                    recorder.record(
                        event_type="tool.call",
                        phase="end",
                        trace_id=recorder.trace_id(run_id),
                        span_id=recorder.span_id(run_id, "tool", worker_id, index),
                        run_id=run_id,
                        tool=recorder.tool_payload(
                            "document_search",
                            arguments={"worker": worker_id, "index": index},
                            result={"answer": "x" * 64},
                        ),
                    )

            threads = [threading.Thread(target=worker, args=(worker_id,)) for worker_id in range(threads_count)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            recorder.flush_writes(timeout_seconds=5)

            lines = path.read_text(encoding="utf-8").splitlines()
            # No torn or interleaved lines: every line parses as standalone JSON.
            events = [json.loads(line) for line in lines]
            self.assertEqual(len(events), threads_count * events_per_thread)
            # No duplicated or lost writes.
            event_ids = {event["event_id"] for event in events}
            self.assertEqual(len(event_ids), threads_count * events_per_thread)
            self.assertFalse(recorder.status()["write_failed"])
            self.assertEqual(recorder.status()["write_queue_dropped_events"], 0)

    def test_concurrent_sync_writes_do_not_corrupt_jsonl(self):
        self._concurrent_write(async_writes=False)

    def test_concurrent_async_writes_do_not_corrupt_jsonl(self):
        self._concurrent_write(async_writes=True)

    def test_unknown_fields_are_additive_and_fail_open(self):
        # Schema evolution contract: a newer recorder may add fields; an older
        # validator/loader must accept them without flagging or dropping events.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(
                    enabled=True,
                    path=str(path),
                    capture_mode="metadata",
                    schema_validation_enabled=True,
                    schema_validation_strict=True,
                )
            )

            # A known event carrying unknown extra attributes records cleanly
            # even under strict validation, and the extra fields survive to JSONL.
            event = recorder.record(
                event_type="tool.call",
                phase="end",
                trace_id=recorder.trace_id("compat"),
                span_id=recorder.span_id("compat", "tool"),
                run_id="compat",
                tool=recorder.tool_payload("document_search"),
                attributes={"future_field": "preserved", "nested": {"future_nested": 7}},
            )

            self.assertEqual(validate_event_schema(event), [])
            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written["attributes"]["future_field"], "preserved")
            self.assertEqual(written["attributes"]["nested"]["future_nested"], 7)
            self.assertEqual(recorder.status()["schema_invalid_events"], 0)

            # An event with unknown top-level fields produced by a newer schema
            # is still accepted by the current validator (additive within minor).
            forward_event = dict(event)
            forward_event["future_top_level"] = {"shape": "tbd"}
            self.assertEqual(validate_event_schema(forward_event), [])


class FlightRecorderRobustnessTest(unittest.TestCase):
    """Fail-open guarantees and degradation paths that must never break traffic."""

    def test_record_is_fail_open_and_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path))
            )

            def boom(**kwargs):
                raise RuntimeError("synthetic recorder failure")

            recorder._record_impl = boom

            # A recorder bug must be swallowed, not propagated into the live turn.
            result = recorder.record(
                event_type="tool.call",
                phase="end",
                trace_id=recorder.trace_id("fail-open"),
                span_id=recorder.span_id("fail-open", "tool"),
            )

            self.assertEqual(result, {})
            status = recorder.status()
            self.assertTrue(status["record_failed"])
            self.assertEqual(status["record_dropped_events"], 1)
            # Nothing partial written.
            self.assertFalse(path.exists())

    def test_resolve_hash_secret_precedence_and_weak_fallback(self):
        import os

        secret, source = resolve_hash_secret(FlightRecorderSettings())
        self.assertEqual(source, "default-dev")
        self.assertEqual(secret, DEFAULT_DEV_HASH_SECRET)

        var = "HERMES_FR_TEST_SECRET_UNSET"
        os.environ.pop(var, None)
        secret, source = resolve_hash_secret(FlightRecorderSettings(hash_secret_env=var))
        self.assertEqual(source, "weak-env-fallback")
        # Never use the env var NAME as the secret.
        self.assertNotEqual(secret, var)
        self.assertEqual(secret, DEFAULT_DEV_HASH_SECRET)

        os.environ[var] = "real-strong-secret-value"
        try:
            secret, source = resolve_hash_secret(FlightRecorderSettings(hash_secret_env=var))
            self.assertEqual(source, "env")
            self.assertEqual(secret, "real-strong-secret-value")
        finally:
            os.environ.pop(var, None)

        with tempfile.TemporaryDirectory() as tmp:
            secret_file = Path(tmp) / "secret"
            secret_file.write_text("file-secret", encoding="utf-8")
            os.environ[var] = "env-secret"
            try:
                secret, source = resolve_hash_secret(
                    FlightRecorderSettings(hash_secret_file=str(secret_file), hash_secret_env=var)
                )
                # File beats env.
                self.assertEqual(source, "file")
                self.assertEqual(secret, "file-secret")
            finally:
                os.environ.pop(var, None)

    def test_require_strong_secret_blocks_enable_when_weak(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path), require_strong_secret=True)
            )

            status = recorder.status()
            # Config-time fail-closed: enabled is neutralized, reason is visible.
            self.assertFalse(status["enabled"])
            self.assertTrue(status["strong_secret_required"])
            self.assertTrue(status["weak_secret_blocked"])

            recorder.record(
                event_type="hermes.session",
                phase="start",
                trace_id=recorder.trace_id("blocked"),
                span_id=recorder.span_id("blocked", "session"),
            )
            self.assertFalse(path.exists())

    def test_require_strong_secret_allows_enable_with_file_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret_file = Path(tmp) / "secret"
            secret_file.write_text("strong-secret", encoding="utf-8")
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(
                    enabled=True,
                    path=str(path),
                    require_strong_secret=True,
                    hash_secret_file=str(secret_file),
                )
            )

            status = recorder.status()
            self.assertTrue(status["enabled"])
            self.assertFalse(status["weak_secret_blocked"])
            self.assertEqual(status["hash_secret_source"], "file")

    def test_flush_otlp_failure_sets_flag_and_retains_buffer(self):
        class FailingClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, headers, json):
                raise RuntimeError("collector unreachable")

        recorder = HermesFlightRecorder(
            FlightRecorderSettings(
                enabled=True,
                otlp_enabled=True,
                otlp_endpoint="http://collector:4318/v1/traces",
            )
        )
        recorder.record(
            event_type="hermes.session",
            phase="start",
            trace_id=recorder.trace_id("otlp-fail"),
            span_id=recorder.span_id("otlp-fail", "session"),
            session_id="otlp-fail",
        )

        result = asyncio.run(recorder.flush_otlp(http_client_factory=lambda: FailingClient()))

        self.assertEqual(result["exported"], 0)
        status = recorder.status()
        self.assertTrue(status["otlp_failed"])
        # Buffer retained so a later flush can retry â€” no silent loss.
        self.assertEqual(status["otlp_buffered_events"], 1)

    def test_index_failure_keeps_jsonl_source_of_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            index_path = Path(tmp) / "events.sqlite3"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(
                    enabled=True,
                    path=str(path),
                    index_enabled=True,
                    index_path=str(index_path),
                )
            )

            def boom(*args, **kwargs):
                raise RuntimeError("index write failed")

            recorder._index.index_event = boom

            recorder.record(
                event_type="tool.call",
                phase="end",
                trace_id=recorder.trace_id("index-fail"),
                span_id=recorder.span_id("index-fail", "tool"),
            )

            status = recorder.status()
            self.assertTrue(status["index_failed"])
            # JSONL is the source of truth and must still be written.
            self.assertTrue(path.exists())
            self.assertEqual(len(path.read_text(encoding="utf-8").strip().splitlines()), 1)

    def test_write_queue_full_increments_dropped(self):
        import queue as _queue

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = HermesFlightRecorder(
                FlightRecorderSettings(enabled=True, path=str(path))
            )
            # Install a saturated queue with no consumer to force the drop path.
            recorder._write_queue = _queue.Queue(maxsize=1)
            recorder._write_queue.put_nowait(({}, b"{}\n"))

            recorder.record(
                event_type="tool.call",
                phase="end",
                trace_id=recorder.trace_id("queue-full"),
                span_id=recorder.span_id("queue-full", "tool"),
            )

            self.assertEqual(recorder.status()["write_queue_dropped_events"], 1)


class EventTypeSplitContractTest(unittest.TestCase):
    """Locks the package-owned vocabulary invariants.

    The Hermes service owns the LIVE call-site contract against `main.py`; this
    package intentionally does not ship that service module. Package tests keep
    the schema split honest and verify the VM-SANDBOX-BETA wrapper emitters that
    are included in the distribution.
    """

    def _literals(self, filename):
        import re
        src = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "hermes_flight_recorder"
            / filename
        ).read_text(encoding="utf-8")
        return set(re.findall(r'event_type="([^"]+)"', src))

    def test_sets_are_disjoint_and_union_to_allowed(self):
        from hermes_flight_recorder.flight_recorder_schema import (
            ALLOWED_EVENT_TYPES,
            GATED_EVENT_TYPES,
            LIVE_EVENT_TYPES,
            RESERVED_EVENT_TYPES,
            VM_SANDBOX_BETA_EVENT_TYPES,
        )
        self.assertTrue(LIVE_EVENT_TYPES.isdisjoint(RESERVED_EVENT_TYPES))
        self.assertTrue(LIVE_EVENT_TYPES.isdisjoint(GATED_EVENT_TYPES))
        self.assertTrue(LIVE_EVENT_TYPES.isdisjoint(VM_SANDBOX_BETA_EVENT_TYPES))
        self.assertTrue(VM_SANDBOX_BETA_EVENT_TYPES.isdisjoint(GATED_EVENT_TYPES))
        self.assertTrue(VM_SANDBOX_BETA_EVENT_TYPES.isdisjoint(RESERVED_EVENT_TYPES))
        self.assertTrue(GATED_EVENT_TYPES.isdisjoint(RESERVED_EVENT_TYPES))
        self.assertEqual(
            LIVE_EVENT_TYPES | VM_SANDBOX_BETA_EVENT_TYPES | GATED_EVENT_TYPES | RESERVED_EVENT_TYPES,
            ALLOWED_EVENT_TYPES,
        )

    def test_vm_sandbox_beta_types_have_a_wrapper_emitter(self):
        from hermes_flight_recorder.flight_recorder_schema import VM_SANDBOX_BETA_EVENT_TYPES
        wrapper_literals = self._literals("flight_recorder_wrappers.py")
        for event_type in sorted(VM_SANDBOX_BETA_EVENT_TYPES):
            self.assertIn(
                event_type, wrapper_literals,
                f"VM-SANDBOX-BETA type {event_type} has no wrapper emitter",
            )


class RedactionReportFalsePositiveTest(unittest.TestCase):
    """`sk-` is anchored so internal ids containing the substring don't false-flag."""

    def _event(self, run_id):
        return {
            "schema_version": "0.2.0", "recorder_version": "0.2.0", "semconv_version": "x",
            "event_id": "e", "trace_id": "a" * 32, "span_id": "b" * 16,
            "session_id": run_id, "turn_id": run_id, "run_id": run_id, "task_id": run_id,
            "event_type": "memory.search", "phase": "instant", "timestamp": "2026-06-24T00:00:00.000Z",
            "status": "ok", "actor": "system",
            "privacy": {"capture_mode": "metadata"},
        }

    def test_internal_id_with_sk_substring_is_not_flagged(self):
        from hermes_flight_recorder.flight_recorder_timeline import redaction_report
        # "task-store-scan" contains "sk-" mid-word â€” must NOT count as a secret.
        report = redaction_report([self._event("task-store-scan")])
        self.assertEqual(report["possible_secret_patterns"], 0)
        self.assertEqual(report["pattern_hits"], {})

    def test_real_openai_key_is_still_flagged(self):
        from hermes_flight_recorder.flight_recorder_timeline import redaction_report
        ev = self._event("run-1")
        # Built from parts so the source contains no literal key (secret scanner),
        # while the runtime value is a realistic OpenAI-style key the detector flags.
        ev["attributes"] = {"leaked": "sk-" + "proj-" + "ABCDEF0123456789"}
        report = redaction_report([ev])
        self.assertGreaterEqual(report["possible_secret_patterns"], 1)
        self.assertIn("api_key_like", report["pattern_hits"])

    def test_nested_dict_pattern_hit_is_not_double_counted(self):
        # Regression for a redaction_report() bug: walk_items() emits both a
        # container entry (e.g. "runtime": {...}) and each of its leaf entries
        # separately, so scanning scalar_values() on the container re-matched
        # the same string once per ancestor level. A single URL occurrence
        # nested one level deep must count exactly once, not (depth + 1) times.
        from hermes_flight_recorder.flight_recorder_timeline import redaction_report
        ev = self._event("run-nested")
        ev["runtime"] = {
            "target_preview": "https://example.com?[REDACTED]",
            "sandbox_id_preview": "sandbox-1",
        }
        report = redaction_report([ev])
        self.assertEqual(report["urls_raw"], 1)
        self.assertEqual(report["possible_secret_patterns"], 1)
        self.assertEqual(report["pattern_hits"], {"url": 1})


if __name__ == "__main__":
    unittest.main()

