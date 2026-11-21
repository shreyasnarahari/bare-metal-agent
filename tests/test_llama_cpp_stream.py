import json

import httpx
import pytest

from toolsmith.backends import (
    CompletionTruncatedError,
    LlamaCppClient,
    LlamaCppResponseError,
)


def sse_body(events, *, done=False):
    body = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\r\n\r\n" for event in events)
    if done:
        body += "data: [DONE]\r\n\r\n"
    return body.encode()


def stream_client(body, requests):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=body)

    return LlamaCppClient(
        "http://localhost:8080",
        "test-key",
        transport=httpx.MockTransport(handler),
    )


def test_stream_complete_yields_events_and_matches_non_stream_payload():
    requests = []
    events = [
        {"content": "caf", "stop": False},
        {"content": "é", "stop": False},
        {"content": "", "stop": True, "truncated": False, "timings": {"predicted_n": 2}},
    ]
    client = stream_client(sse_body(events, done=True), requests)

    with client:
        streamed = list(
            client.stream_complete(
                "prompt", max_tokens=32, temperature=0.0, grammar="root ::= x", seed=7
            )
        )

    assert streamed == events
    assert requests[0] == {
        "prompt": "prompt",
        "n_predict": 32,
        "temperature": 0.0,
        "grammar": "root ::= x",
        "seed": 7,
        "stream": True,
    }


def test_generation_payloads_match_between_modes():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if requests[-1]["stream"]:
            return httpx.Response(
                200,
                content=sse_body([{"content": "done", "stop": True, "truncated": False}]),
            )
        return httpx.Response(200, json={"content": "done", "truncated": False})

    with LlamaCppClient(
        "http://localhost:8080", "key", transport=httpx.MockTransport(handler)
    ) as client:
        client.complete("prompt", max_tokens=12, temperature=0.2, grammar="g", seed=3)
        list(client.stream_complete("prompt", max_tokens=12, temperature=0.2, grammar="g", seed=3))

    assert requests[0] == {**requests[1], "stream": False}
    assert requests[1]["stream"] is True


@pytest.mark.parametrize(
    "terminal",
    [
        {"content": "partial", "stop": True, "truncated": True},
        {"content": "partial", "stop": True, "stopped_limit": True},
        {"content": "partial", "stop": True, "stop_type": "limit"},
    ],
)
def test_stream_truncation_is_an_explicit_error(terminal):
    client = stream_client(sse_body([terminal]), [])

    with client:
        with pytest.raises(CompletionTruncatedError) as caught:
            list(client.stream_complete("prompt"))

    assert caught.value.response["content"] == "partial"


def test_explicit_non_truncated_limit_from_pinned_backend_is_accepted():
    terminal = {"content": "done", "stop": True, "stop_type": "limit", "truncated": False}
    client = stream_client(sse_body([terminal]), [])

    with client:
        assert list(client.stream_complete("prompt")) == [terminal]


def test_non_stream_truncated_field_is_checked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": "partial", "truncated": True})

    with LlamaCppClient(
        "http://localhost:8080", "key", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(CompletionTruncatedError):
            client.complete("prompt")


def test_missing_terminal_event_is_an_error():
    client = stream_client(sse_body([{"content": "partial", "stop": False}]), [])

    with client:
        with pytest.raises(LlamaCppResponseError, match="terminal"):
            list(client.stream_complete("prompt"))


def test_invalid_stream_json_is_an_error():
    client = stream_client(b"data: not-json\n\n", [])

    with client:
        with pytest.raises(LlamaCppResponseError, match="invalid JSON"):
            list(client.stream_complete("prompt"))


class TrackingStream(httpx.SyncByteStream):
    def __init__(self):
        self.closed = False

    def __iter__(self):
        yield sse_body([{"content": "first", "stop": False}])
        yield sse_body([{"content": "second", "stop": True, "truncated": False}])

    def close(self):
        self.closed = True


def test_closing_stream_iterator_early_closes_http_response():
    stream = TrackingStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    with LlamaCppClient(
        "http://localhost:8080", "key", transport=httpx.MockTransport(handler)
    ) as client:
        iterator = client.stream_complete("prompt")
        assert next(iterator)["content"] == "first"
        iterator.close()

    assert stream.closed


def test_stream_http_error_preserves_status_and_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    with LlamaCppClient(
        "http://localhost:8080", "key", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(LlamaCppResponseError) as caught:
            list(client.stream_complete("prompt"))

    assert caught.value.status_code == 401
    assert json.loads(caught.value.response_body) == {"error": "unauthorized"}
