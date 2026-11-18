from __future__ import annotations

from typing import Any

import httpx


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
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must not be negative")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if grammar is not None:
            payload["grammar"] = grammar
        if seed is not None:
            payload["seed"] = seed

        response = self._request("POST", "/completion", json=payload)
        if response.get("stopped_limit") is True or response.get("stop_type") == "limit":
            raise CompletionTruncatedError(response)
        return response

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise LlamaCppTimeoutError(f"llama.cpp {method} {path} timed out") from exc
        except httpx.RequestError as exc:
            raise LlamaCppConnectionError(
                f"Could not connect to llama.cpp for {method} {path}"
            ) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text
            raise LlamaCppResponseError(
                f"llama.cpp returned HTTP {response.status_code}: {body[:500]}",
                status_code=response.status_code,
                response_body=body,
            ) from exc

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
