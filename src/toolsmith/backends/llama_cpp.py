"""HTTP client for llama-server's raw ``/completion`` endpoint.

Toolsmith renders prompts itself and sends llama.cpp a plain prompt string (plus
an optional GBNF grammar), instead of using llama-server's own chat endpoint.
That keeps prompt rendering, tool parsing, and validation in Python where they
can be tested. Every failure maps to one ``LlamaCppError`` subclass, so callers
can tell a dead server (connection), a slow one (timeout), a bad reply
(response), and a generation that hit ``max_tokens`` (truncated) apart.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from ..sse import SSEDecodeError, SSEJSONDecoder


class LlamaCppError(RuntimeError):
    """Base class for every failure talking to llama-server."""


class LlamaCppConnectionError(LlamaCppError):
    """The server could not be reached (not started, crashed, wrong port)."""


class LlamaCppTimeoutError(LlamaCppError):
    """The server took longer than the client timeout."""


class LlamaCppResponseError(LlamaCppError):
    """An HTTP error status, or a body that is not the JSON object expected."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class CompletionTruncatedError(LlamaCppError):
    """Generation stopped at ``max_tokens``; ``response`` keeps what was produced."""

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__("Completion stopped at the requested token limit")
        self.response = response


class LlamaCppClient:
    """A thread-safe, connection-pooling client; one instance can serve every worker."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not api_key:
            raise ValueError("api_key must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> LlamaCppClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        """``GET /health``; llama-server answers 503 while the model is still loading."""
        return self._request("GET", "/health")

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.6,
        grammar: str | None = None,
        seed: int | None = None,
        id_slot: int | None = None,
    ) -> dict[str, Any]:
        """One non-streaming completion. Raises ``CompletionTruncatedError`` at the token limit.

        ``id_slot`` pins the request to one llama-server slot (sent only when given).
        """
        response = self._request(
            "POST",
            "/completion",
            json=self._completion_payload(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                grammar=grammar,
                seed=seed,
                stream=False,
                id_slot=id_slot,
            ),
        )
        if self._is_truncated(response):
            raise CompletionTruncatedError(response)
        return response

    def stream_complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.6,
        grammar: str | None = None,
        seed: int | None = None,
        id_slot: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield each streamed event as it arrives.

        Raises ``CompletionTruncatedError`` after the last event if the token
        limit was hit; the error's response holds all the streamed content.
        """
        payload = self._completion_payload(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            grammar=grammar,
            seed=seed,
            stream=True,
            id_slot=id_slot,
        )
        # The terminal event (stop=true) carries timings and the stop reason; content arrives
        # in pieces before it, so it is collected for a truncation error.
        terminal: dict[str, Any] | None = None
        content_parts: list[str] = []
        try:
            with self._client.stream("POST", "/completion", json=payload) as response:
                self._raise_for_status(response)
                decoder = SSEJSONDecoder()
                for chunk in response.iter_bytes():
                    for event in decoder.feed(chunk):
                        terminal = self._process_stream_event(event, content_parts)
                        yield event
                for event in decoder.finish():
                    terminal = self._process_stream_event(event, content_parts)
                    yield event
        except httpx.TimeoutException as exc:
            raise LlamaCppTimeoutError("llama.cpp streaming completion timed out") from exc
        except httpx.RequestError as exc:
            raise LlamaCppConnectionError("llama.cpp streaming connection failed") from exc
        except SSEDecodeError as exc:
            raise LlamaCppResponseError(str(exc)) from exc

        if terminal is None:
            raise LlamaCppResponseError("llama.cpp stream ended without a terminal event")
        if self._is_truncated(terminal):
            response = dict(terminal)
            response["content"] = "".join(content_parts)
            raise CompletionTruncatedError(response)

    # Every event must have string content and a boolean stop flag; returns the event if it is the last.
    def _process_stream_event(
        self,
        event: dict[str, Any],
        content_parts: list[str],
    ) -> dict[str, Any] | None:
        if not isinstance(event.get("content"), str):
            raise LlamaCppResponseError("Streaming event is missing string content")
        if not isinstance(event.get("stop"), bool):
            raise LlamaCppResponseError("Streaming event is missing boolean stop")
        content_parts.append(event["content"])
        return event if event["stop"] else None

    @staticmethod
    def _completion_payload(
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        grammar: str | None,
        seed: int | None,
        stream: bool,
        id_slot: int | None = None,
    ) -> dict[str, Any]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must not be negative")

        # llama.cpp's names: n_predict is the token limit. A grammar constrains sampling to GBNF.
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if grammar is not None:
            payload["grammar"] = grammar
        if seed is not None:
            payload["seed"] = seed
        if id_slot is not None:
            if isinstance(id_slot, bool) or not isinstance(id_slot, int) or id_slot < 0:
                raise ValueError("id_slot must be a non-negative integer")
            payload["id_slot"] = id_slot
        return payload

    @staticmethod
    def _is_truncated(response: dict[str, Any]) -> bool:
        # llama.cpp has reported this three ways across releases; newer ones send "truncated".
        if "truncated" in response:
            return response["truncated"] is True
        return response.get("stopped_limit") is True or response.get("stop_type") == "limit"

    # One JSON request, with transport errors mapped to the LlamaCppError subclasses above.
    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise LlamaCppTimeoutError(f"llama.cpp {method} {path} timed out") from exc
        except httpx.RequestError as exc:
            raise LlamaCppConnectionError(
                f"Could not connect to llama.cpp for {method} {path}"
            ) from exc

        self._raise_for_status(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise LlamaCppResponseError(
                "llama.cpp returned a non-JSON response",
                status_code=response.status_code,
                response_body=response.text,
            ) from exc

        if not isinstance(payload, dict):
            raise LlamaCppResponseError(
                "llama.cpp returned a JSON value instead of an object",
                status_code=response.status_code,
                response_body=response.text,
            )
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # A streamed response has not read its body yet; read it so the error can quote it.
            if not response.is_closed and not hasattr(response, "_content"):
                response.read()
            body = response.text
            raise LlamaCppResponseError(
                f"llama.cpp returned HTTP {response.status_code}: {body[:500]}",
                status_code=response.status_code,
                response_body=body,
            ) from exc
