import json

import httpx
import pytest

from toolsmith.backends import (
    CompletionTruncatedError,
    LlamaCppClient,
    LlamaCppConnectionError,
    LlamaCppResponseError,
    LlamaCppTimeoutError,
)


def make_client(handler):
    return LlamaCppClient(
        "http://localhost:8080/",
        "test-key",
        transport=httpx.MockTransport(handler),
    )


def test_health_uses_health_endpoint_and_auth_header():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"status": "ok"})

    with make_client(handler) as client:
        assert client.health() == {"status": "ok"}


def test_complete_sends_raw_completion_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/completion"
        assert json.loads(request.content) == {
            "prompt": "Name one primary color.",
            "n_predict": 64,
            "temperature": 0.0,
            "stream": False,
        }
        return httpx.Response(200, json={"content": "Blue", "stopped_limit": False})

    with make_client(handler) as client:
        response = client.complete(
            "Name one primary color.", max_tokens=64, temperature=0.0
        )

    assert response["content"] == "Blue"


def test_complete_includes_grammar_only_when_provided():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"content": "Blue"})

    with make_client(handler) as client:
        client.complete("without grammar")
        client.complete("with grammar", grammar='root ::= "Blue"')

    assert "grammar" not in payloads[0]
    assert payloads[1]["grammar"] == 'root ::= "Blue"'


def test_http_error_preserves_status_and_response_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "model is loading"})

    with make_client(handler) as client:
        with pytest.raises(LlamaCppResponseError) as caught:
            client.health()

    assert caught.value.status_code == 503
    assert json.loads(caught.value.response_body) == {"error": "model is loading"}


def test_non_json_response_has_a_specific_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with make_client(handler) as client:
        with pytest.raises(LlamaCppResponseError, match="non-JSON"):
            client.health()


def test_timeout_is_not_reported_as_a_connection_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with make_client(handler) as client:
        with pytest.raises(LlamaCppTimeoutError):
            client.health()


def test_connection_error_is_not_silently_ignored():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with make_client(handler) as client:
        with pytest.raises(LlamaCppConnectionError):
            client.health()


@pytest.mark.parametrize(
    "response",
    [
        {"content": "partial", "stopped_limit": True},
        {"content": "partial", "stop_type": "limit"},
    ],
)
def test_token_limit_is_an_explicit_truncation_error(response):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with make_client(handler) as client:
        with pytest.raises(CompletionTruncatedError) as caught:
            client.complete("prompt")

    assert caught.value.response == response


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_tokens": 0}, "max_tokens"),
        ({"temperature": -0.1}, "temperature"),
    ],
)
def test_complete_rejects_invalid_generation_settings(kwargs, message):
    with make_client(lambda request: httpx.Response(500)) as client:
        with pytest.raises(ValueError, match=message):
            client.complete("prompt", **kwargs)
