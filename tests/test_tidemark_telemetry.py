"""Backend timing sanitization, the cache formulas, the cache-semantics probe, and buffered-stream TTFT."""

import json
import math
import time
from pathlib import Path

import pytest

from steadyhand import SteadyhandClient
from steadyhand.types import AttemptMetrics, TurnKind, TurnMetrics
from tidemark import telemetry
from tidemark.telemetry import (
    backend_timing_fields,
    cache_reuse_fraction,
    derive_cache,
    interpret_cache_probe,
    prefill_savings_fraction,
    probe_cache_semantics,
    reused_tokens,
    slot_completion,
)
from toolsmith.api import APIError, ChatCompletionRequest, ToolsmithService
from toolsmith.backends import CompletionTruncatedError, LlamaCppClient
from toolsmith.parsing import Qwen3ToolCallParser
from toolsmith.repair import RepairPolicy
from toolsmith.timings import AttemptTiming, finite_number, sanitize_backend_fields

from tidemark_support import FakeService, attempt, completion

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "tidemark" / "backend_responses.json").read_text())
CONFIRMED = {"confirmed": True, "new_tokens_field": "prompt_n", "total_tokens_field": "tokens_evaluated"}
CONFIRMED_NO_TOTAL = {"confirmed": True, "new_tokens_field": "prompt_n", "total_tokens_field": None}
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather", "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False},
    },
}
CALL = '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
BAD_CALL = '<tool_call>{"name":"get_weather","arguments":{"town":"Paris"}}</tool_call>'


def metrics(*attempts_, elapsed=50.0):
    return TurnMetrics.from_toolsmith(list(attempts_), response_elapsed_ms=elapsed, generation_path="non_streaming")


# Sanitization


def test_only_whitelisted_finite_non_negative_numbers_survive():
    clean = sanitize_backend_fields(FIXTURES["hostile"])
    # prompt_n 12.5 is not an integer count; NaN strings, infinities, negatives, booleans, and text are dropped.
    assert clean == {"predicted_n": 7, "cache_n": 3}
    assert "authorization" not in json.dumps(clean) and "headers" not in json.dumps(clean)


def test_finite_number_rules():
    assert finite_number(3) == 3 and finite_number(2.5) == 2.5 and finite_number(0) == 0
    assert finite_number(True) is None and finite_number("4") is None and finite_number(-1) is None
    assert finite_number(float("nan")) is None and finite_number(float("inf")) is None
    assert finite_number(12.0, integer=True) == 12 and finite_number(12.5, integer=True) is None


def test_a_sanitized_cold_response_keeps_counts_and_timings():
    clean = sanitize_backend_fields(FIXTURES["cold"])
    assert clean["prompt_n"] == 412 and clean["tokens_evaluated"] == 412 and clean["id_slot"] == 0
    assert clean["prompt_ms"] == 91.25 and "content" not in clean and "truncated" not in clean


def test_a_non_streaming_attempt_cannot_claim_a_ttft():
    with pytest.raises(ValueError):
        AttemptTiming(0, "success", "unavailable", 10.0, ttft_ms=5.0)
    with pytest.raises(ValueError):
        AttemptTiming(0, "success", "streamed", 10.0)


def test_attempt_metrics_keep_unknown_values_null_not_zero():
    parsed = AttemptMetrics.from_toolsmith({"attempt": 0, "outcome": "success", "measurement_path": "unavailable",
                                            "response_elapsed_ms": 9.0, "ttft_ms": None, "backend": {}})
    assert parsed.prompt_evaluated_tokens is None and parsed.completion_tokens is None and parsed.prompt_ms is None
    assert parsed.counts()["raw_counts"] == {} and parsed.timings()["raw_timings"] == {}
    # A TTFT is ignored unless the attempt really streamed.
    forged = AttemptMetrics.from_toolsmith({"attempt": 0, "measurement_path": "unavailable", "ttft_ms": 3.0, "backend": {}})
    assert forged.ttft_ms is None


# Formulas, by hand


@pytest.mark.parametrize(
    ("rendered", "evaluated", "reused", "fraction"),
    [
        (1000, 1, 999, 0.999),  # full hit: only the last token re-evaluated
        (1000, 1000, 0, 0.0),  # full miss
        (1000, 250, 750, 0.75),  # partial reuse
        (0, 0, 0, None),  # zero-token prompt: no denominator
        (None, 10, None, None),  # unknown rendered count
        (100, None, None, None),  # unknown server count
        (100, 120, 0, 0.0),  # never negative (the derivation marks this unreconciled)
    ],
)
def test_cache_formulas(rendered, evaluated, reused, fraction):
    assert reused_tokens(rendered, evaluated) == reused
    result = cache_reuse_fraction(rendered, evaluated)
    assert (result is None and fraction is None) or math.isclose(result, fraction)


def test_prefill_savings_needs_a_paired_naive_count():
    assert math.isclose(prefill_savings_fraction(200, 800), 0.75)
    assert prefill_savings_fraction(800, 800) == 0.0
    assert prefill_savings_fraction(None, 800) is None and prefill_savings_fraction(100, 0) is None
    assert prefill_savings_fraction(100, None) is None


def test_cache_metrics_stay_null_until_the_semantics_are_confirmed():
    turn = metrics(attempt(prompt_n=1, tokens_evaluated=412))
    assert derive_cache(412, turn, {"confirmed": False}).cache_status == telemetry.UNCONFIRMED
    assert derive_cache(412, turn, {"confirmed": False}).reused_tokens is None
    confirmed = derive_cache(412, turn, CONFIRMED)
    assert confirmed.cache_status == telemetry.AVAILABLE
    assert confirmed.reused_tokens == 411 and math.isclose(confirmed.cache_reuse_fraction, 411 / 412)


@pytest.mark.parametrize(
    ("rendered", "attempts", "status"),
    [
        (None, [attempt(prompt_n=5)], telemetry.MISSING_RENDERED),
        (100, [attempt()], telemetry.MISSING_SERVER),
        (100, [attempt(prompt_n=101)], telemetry.UNRECONCILED),
        (100, [attempt(prompt_n=5, tokens_evaluated=99)], telemetry.UNRECONCILED),  # the server counted another prompt
        (100, [], telemetry.MISSING_SERVER),
    ],
)
def test_unavailable_cache_metrics_say_why(rendered, attempts, status):
    derived = derive_cache(rendered, metrics(*attempts), CONFIRMED)
    assert derived.cache_status == status and derived.reused_tokens is None and derived.cache_reuse_fraction is None


def test_repairs_do_not_fold_into_the_first_attempt_cache_metrics():
    turn = metrics(attempt(prompt_n=40, outcome="retry"), attempt(prompt_n=300, outcome="repaired", index=1))
    fields = backend_timing_fields(turn, 400, CONFIRMED_NO_TOTAL)
    assert fields["prompt_evaluated_tokens"] == 40 and fields["reused_tokens"] == 360
    assert [item["outcome"] for item in fields["attempt_metrics"]] == ["retry", "repaired"]
    assert [item["prompt_evaluated_tokens"] for item in fields["attempt_metrics"]] == [40, 300]
    assert turn.repair_attempts == 1


def test_backend_timing_fields_without_metrics_are_null():
    fields = backend_timing_fields(None, 120, CONFIRMED)
    assert fields["attempt_metrics"] == [] and fields["prompt_ms"] is None and fields["ttft_ms"] is None
    assert fields["measurement_path"] == "unavailable" and fields["cache_status"] == telemetry.MISSING_SERVER
    assert fields["rendered_prompt_tokens"] == 120


# The repeated-prompt probe


def test_a_cold_then_warm_probe_confirms_prompt_n_means_new_tokens():
    semantics = interpret_cache_probe(FIXTURES["cold"], FIXTURES["warm"], 412)
    assert semantics.confirmed and semantics.new_tokens_field == "prompt_n"
    assert semantics.total_tokens_field == "tokens_evaluated"
    assert semantics.config() == CONFIRMED and semantics.evidence["warm"]["prompt_n"] == 1


def test_a_probe_that_does_not_reconcile_stays_unconfirmed():
    assert not interpret_cache_probe(FIXTURES["cold"], FIXTURES["warm"], 400).confirmed  # tokenizer disagrees
    assert not interpret_cache_probe(FIXTURES["cold"], FIXTURES["cold"], 412).confirmed  # no reuse seen
    semantics = interpret_cache_probe({"content": ""}, {"content": ""}, 10)
    assert not semantics.confirmed and "reason" in semantics.evidence


def test_probe_cache_semantics_marks_the_prompt_unique_and_sends_it_twice():
    sent = []

    def complete(prompt):
        sent.append(prompt)
        return FIXTURES["cold"] if len(sent) == 1 else FIXTURES["warm"]

    semantics = probe_cache_semantics(complete, "hello", lambda text: 412)
    assert sent[0] == sent[1] and sent[0].startswith("[cache-probe ") and sent[0].endswith("hello")
    assert semantics.confirmed


def test_slot_completion_pins_the_slot_and_accepts_the_one_token_stop():
    class Client:
        def __init__(self):
            self.calls = []

        def complete(self, prompt, **settings):
            self.calls.append(settings)
            raise CompletionTruncatedError(FIXTURES["cold"])

    client = Client()
    assert slot_completion(client, id_slot=1)("p") == FIXTURES["cold"]
    assert client.calls == [{"max_tokens": 1, "temperature": 0.0, "seed": 0, "id_slot": 1}]


def test_the_llama_payload_carries_id_slot_only_when_given():
    base = dict(max_tokens=4, temperature=0.0, grammar=None, seed=None, stream=False)
    assert "id_slot" not in LlamaCppClient._completion_payload("p", **base)
    assert LlamaCppClient._completion_payload("p", **base, id_slot=2)["id_slot"] == 2
    with pytest.raises(ValueError):
        LlamaCppClient._completion_payload("p", **base, id_slot=-1)
    with pytest.raises(ValueError):
        LlamaCppClient._completion_payload("p", **base, id_slot=True)


# Buffered-stream TTFT inside ToolsmithService


class Renderer:
    def render(self, messages, tools=(), *, enable_thinking=None):
        return "PROMPT:" + json.dumps(messages, sort_keys=True)


class StreamingBackend:
    """Streams scripted event lists; ``delay`` sleeps before each non-empty piece."""

    def __init__(self, *scripts, delay=0.02, truncate=False):
        self.scripts = list(scripts)
        self.delay = delay
        self.truncate = truncate
        self.calls = []
        self.closed = 0

    def stream_complete(self, prompt, **settings):
        self.calls.append((prompt, settings))
        events = self.scripts.pop(0)
        backend = self

        def generate():
            try:
                yield {"content": "", "stop": False}  # an empty first chunk never counts as first content
                for piece in events:
                    time.sleep(backend.delay)
                    yield {"content": piece, "stop": False}
                terminal = {"content": "", "stop": True, "id_slot": settings.get("id_slot", 0), "tokens_evaluated": 30,
                            "timings": {"prompt_n": 30, "prompt_ms": 7.5, "predicted_n": len(events), "predicted_ms": 11.0}}
                if backend.truncate:
                    raise CompletionTruncatedError({**terminal, "content": "".join(events), "truncated": True})
                yield terminal
            finally:
                backend.closed += 1

        return generate()

    def complete(self, prompt, **settings):  # pragma: no cover - the buffered path must not call this
        raise AssertionError("non-streaming complete() used on the buffered-stream path")


def service(backend, *, repairs=1, path="buffered_stream"):
    return ToolsmithService(
        backend, Renderer(), parser=Qwen3ToolCallParser(lambda: "call_test"),
        repair_policy=RepairPolicy(max_repairs=repairs), generation_path=path,
    )


def request(**overrides):
    values = {
        "messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER_TOOL], "max_tokens": 64,
        "temperature": 0.0, "seed": 1, "chat_template_kwargs": {"enable_thinking": False},
    }
    return ChatCompletionRequest(**{**values, **overrides})


def test_buffered_stream_records_first_content_ttft_and_returns_one_response():
    backend = StreamingBackend([CALL[:12], CALL[12:]], delay=0.03)
    response = service(backend).chat(request(), id_slot=1)
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    meta = response["toolsmith"]
    assert meta["generation_path"] == "buffered_stream" and len(meta["attempts"]) == 1
    first = meta["attempts"][0]
    assert first["measurement_path"] == "buffered_stream" and first["outcome"] == "success"
    assert first["ttft_ms"] >= 25  # timed at the first non-empty piece, after the empty chunk
    assert first["response_elapsed_ms"] >= first["ttft_ms"]
    assert first["backend"]["prompt_n"] == 30 and first["backend"]["id_slot"] == 1
    assert backend.calls[0][1]["id_slot"] == 1 and backend.closed == 1
    assert response["usage"]["prompt_tokens"] == 30


def test_buffered_stream_runs_the_same_validation_and_at_most_one_repair():
    backend = StreamingBackend([BAD_CALL], [CALL], delay=0.0)
    response = service(backend).chat(request())
    outcomes = [item["outcome"] for item in response["toolsmith"]["attempts"]]
    assert outcomes == ["retry", "repaired"] and response["toolsmith"]["repair_attempts"] == 1
    assert len(backend.calls) == 2 and "town" in backend.calls[1][0]  # the repair quotes the bad output

    failing = StreamingBackend([BAD_CALL], [BAD_CALL], [CALL], delay=0.0)
    with pytest.raises(APIError) as caught:
        service(failing).chat(request())
    assert len(failing.calls) == 2  # never a third attempt
    assert [item["outcome"] for item in caught.value.attempt_metrics] == ["retry", "error"]


def test_buffered_stream_parse_errors_match_the_non_streaming_path():
    for path in ("buffered_stream", "non_streaming"):
        if path == "buffered_stream":
            backend = StreamingBackend(["<tool_call>{broken"], delay=0.0)
        else:
            class Backend:
                def complete(self, prompt, **settings):
                    return {"content": "<tool_call>{broken", "timings": {"prompt_n": 30}}
            backend = Backend()
        with pytest.raises(APIError) as caught:
            service(backend, repairs=0, path=path).chat(request())
        assert caught.value.status_code == 500 and caught.value.raw_output == "<tool_call>{broken"
        assert caught.value.attempt_metrics[0]["outcome"] == "error"


def test_buffered_stream_truncation_is_not_repaired_and_keeps_its_ttft():
    backend = StreamingBackend([CALL[:10]], delay=0.01, truncate=True)
    with pytest.raises(APIError) as caught:
        service(backend).chat(request())
    assert caught.value.code == "completion_truncated" and len(backend.calls) == 1
    (only,) = caught.value.attempt_metrics
    assert only["outcome"] == "error" and only["ttft_ms"] is not None and backend.closed == 1


def test_a_stream_without_content_has_no_ttft():
    backend = StreamingBackend([], delay=0.0)
    with pytest.raises(APIError) as caught:
        service(backend, repairs=0).chat(request(tools=[]))
    (only,) = caught.value.attempt_metrics
    assert only["ttft_ms"] is None and only["measurement_path"] == "unavailable"


def test_non_streaming_attempts_never_report_a_ttft():
    class Backend:
        def __init__(self):
            self.settings = []

        def complete(self, prompt, **settings):
            self.settings.append(settings)
            return {"content": "Sunny.", "timings": {"prompt_n": 9, "prompt_ms": 2.0, "predicted_n": 2}}

    backend = Backend()
    response = service(backend, path="non_streaming").chat(request(tools=[]))
    (only,) = response["toolsmith"]["attempts"]
    assert only["ttft_ms"] is None and only["measurement_path"] == "unavailable" and only["backend"]["prompt_n"] == 9
    assert backend.settings == [{"max_tokens": 64, "temperature": 0.0, "seed": 1}]


def test_an_unknown_generation_path_is_refused():
    with pytest.raises(ValueError):
        service(StreamingBackend(), path="streaming")


def test_render_prompt_is_the_first_attempt_prompt():
    backend = StreamingBackend([CALL], delay=0.0)
    svc = service(backend)
    prompt = svc.render_prompt(request())
    svc.chat(request())
    assert backend.calls[0][0] == prompt


# SteadyhandClient


def test_the_client_attaches_typed_metrics_and_passes_the_slot():
    svc = FakeService(completion("Done.", attempts=[attempt(prompt_n=20, tokens_evaluated=20, id_slot=3, ttft_ms=4.0)],
                                 generation_path="buffered_stream"), generation_path="buffered_stream")
    client = SteadyhandClient(svc)
    turn = client.complete([{"role": "user", "content": "hi"}], [], id_slot=3)
    assert turn.kind is TurnKind.TEXT and svc.slots == [3]
    assert turn.metrics.prompt_evaluated_tokens == 20 and turn.metrics.slot_id == 3
    assert turn.metrics.ttft_ms == 4.0 and turn.metrics.measurement_path == "buffered_stream"
    assert turn.metrics.response_elapsed_ms is not None and client.settings()["generation_path"] == "buffered_stream"


def test_legacy_settings_and_failures_keep_their_shape():
    svc = FakeService(completion("ok"))
    client = SteadyhandClient(svc)
    assert "generation_path" not in client.settings()
    turn = client.complete([{"role": "user", "content": "hi"}], [])
    assert svc.slots == [None] and turn.metrics.attempts == () and turn.metrics.ttft_ms is None

    error = APIError("bad", status_code=500, code="invalid_tool_call_json")
    error.attempt_metrics = [attempt(prompt_n=10, outcome="error")]
    failing = SteadyhandClient(FakeService(error))
    failed = failing.complete([{"role": "user", "content": "hi"}], [])
    assert failed.kind is TurnKind.FATAL_ERROR and failed.metrics.prompt_evaluated_tokens == 10
