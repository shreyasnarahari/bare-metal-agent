from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from ..sse import SSEDecodeError, SSEJSONDecoder


class LlamaCppError(RuntimeError):
    pass


class LlamaCppConnectionError(LlamaCppError):
    pass


class LlamaCppTimeoutError(LlamaCppError):
    pass


class LlamaCppResponseError(LlamaCppError):
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
    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__("Completion stopped at the requested token limit")
        self.response = response


class LlamaCppClient:
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
        return self._request("GET", "/health")

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.6,
        grammar: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
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
    ) -> Iterator[dict[str, Any]]:
        payload = self._completion_payload(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            grammar=grammar,
            seed=seed,
            stream=True,
        )
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
    ) -> dict[str, Any]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must not be negative")

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
        return payload

    @staticmethod
    def _is_truncated(response: dict[str, Any]) -> bool:
        if "truncated" in response:
            return response["truncated"] is True
        return response.get("stopped_limit") is True or response.get("stop_type") == "limit"

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
            if not response.is_closed and not hasattr(response, "_content"):
                response.read()
            body = response.text
            raise LlamaCppResponseError(
                f"llama.cpp returned HTTP {response.status_code}: {body[:500]}",
                status_code=response.status_code,
                response_body=body,
            ) from exc
