"""Optional auto-instrumentation adapters for popular LLM/agent SDKs.

Each adapter wraps a client in place so calls emit canonical Flight Recorder
events with no per-call-site ``span()``/``record()`` calls. Adapters are
duck-typed and do not import the SDK they instrument, so importing a
submodule here never requires that SDK to be installed -- only calling the
adapter function does (and even then, only if you pass it an actual client).

Import the specific adapter you need, e.g.::

    from hermes_flight_recorder.integrations.openai import instrument_openai
"""

from __future__ import annotations
