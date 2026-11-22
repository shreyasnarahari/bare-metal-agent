"""Model-turn adapter over ``ToolsmithService.chat``, plus live-server wiring.

Prompt rendering, native tool-call parsing, grammars, and the shim's one-shot
model-output repair all stay inside ``toolsmith``. This module only builds the
request, calls ``chat()`` (never ``chat_stream()``), and classifies the result.

Concurrency: ``SteadyhandClient`` keeps no per-call state. ``ToolsmithService``
recompiles a grammar in ``_prepare`` without a lock, so each worker thread gets
its own service (``worker_service``) and all of them share one ``LlamaCppClient``
whose ``httpx`` pool is sized to the server's ``--parallel``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from toolsmith.api import APIError, ChatCompletionRequest, ToolsmithService
from toolsmith.backends import (
    CompletionTruncatedError,
    LlamaCppClient,
    LlamaCppConnectionError,
    LlamaCppError,
    LlamaCppTimeoutError,
)
from toolsmith.parsing import ToolCallParseError
from toolsmith.prompting import ChatPromptRenderer
from toolsmith.protocols import ModelProfile
from toolsmith.validation import StructuredOutputValidationError, ToolCallValidationError

from .types import ModelTurn, ToolCall, TurnKind

DEFAULT_MAX_TOKENS = 256
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_MODEL_OUTPUT_EXCEPTIONS = (
    ToolCallParseError,
    ToolCallValidationError,
    StructuredOutputValidationError,
)


class SteadyhandClient:
    """One model turn per ``complete()`` call: text, tool calls, a parse error, or a fatal error."""

    def __init__(
        self,
        service: ToolsmithService,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        seed: int | None = 0,
        enable_thinking: bool | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must not be negative")
        use_grammars = bool(service.use_grammars)
        if enable_thinking is None:
            # The grammar path needs thinking off; prompt-only mode is the thinking ablation.
            enable_thinking = not use_grammars
        if use_grammars and enable_thinking:
            raise ValueError(
                "Fixed tool grammars require enable_thinking=False. "
                "Use a service with use_grammars=False for prompt-only thinking."
            )
        self.service = service
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.enable_thinking = enable_thinking

    @property
    def profile(self) -> ModelProfile:
        return self.service.profile

    def settings(self) -> dict[str, Any]:
        """What the journal records about this model configuration."""
        return {
            "profile": self.profile.name,
            "model": self.profile.model,
            "protocol": self.profile.protocol.value,
            "use_grammars": bool(self.service.use_grammars),
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
        }

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ModelTurn:
        request = ChatCompletionRequest(
            model=self.profile.model,
            messages=deepcopy([dict(message) for message in messages]),
            tools=deepcopy([dict(tool) for tool in tools]),
            tool_choice=tool_choice,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        started = time.perf_counter()
        try:
            response = self.service.chat(request)
        except APIError as exc:
            return ModelTurn.failure(
                TurnKind.PARSE_ERROR if is_model_output_error(exc) else TurnKind.FATAL_ERROR,
                exc.code,
                str(exc),
                raw_output=getattr(exc, "raw_output", None),
                repair_attempts=exc.repair_attempts,
                elapsed_s=time.perf_counter() - started,
            )
        except LlamaCppError as exc:
            return ModelTurn.failure(
                TurnKind.FATAL_ERROR,
                backend_error_code(exc),
                f"{type(exc).__name__}: {exc}",
                elapsed_s=time.perf_counter() - started,
            )
        return classify_completion(response, elapsed_s=time.perf_counter() - started)


def classify_completion(response: Mapping[str, Any], *, elapsed_s: float = 0.0) -> ModelTurn:
    """Turn an OpenAI-shaped chat completion from the shim into a ``ModelTurn``."""
    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    metadata = response.get("toolsmith") or {}
    repair_attempts = int(metadata.get("repair_attempts") or 0)
    try:
        message = deepcopy(dict(response["choices"][0]["message"]))
        raw_calls = message.get("tool_calls") or []
        calls = tuple(ToolCall.from_openai(call) for call in raw_calls)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        return ModelTurn.failure(
            TurnKind.FATAL_ERROR,
            "invalid_backend_response",
            f"Chat completion has no usable assistant message: {exc!r}",
            repair_attempts=repair_attempts,
            elapsed_s=elapsed_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if calls:
        kind = TurnKind.TOOL_CALLS
    elif isinstance(content, str) and content.strip():
        kind = TurnKind.TEXT
    else:
        return ModelTurn.failure(
            TurnKind.PARSE_ERROR,
            "empty_model_response",
            "The assistant message has neither visible content nor tool calls",
            raw_output=content if isinstance(content, str) else None,
            repair_attempts=repair_attempts,
            elapsed_s=elapsed_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    return ModelTurn(
        kind=kind,
        assistant_message=message,
        tool_calls=calls,
        content=content if isinstance(content, str) else None,
        reasoning_content=reasoning if isinstance(reasoning, str) else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error_code=None,
        repair_attempts=repair_attempts,
        elapsed_s=elapsed_s,
    )


def is_model_output_error(exc: APIError) -> bool:
    """True when the model's own output failed to parse or validate, so the model can repair it.

    The shim raises these ``from`` a parser or validator exception, and raises
    ``empty_model_response`` directly. Request errors, backend failures, and
    ``completion_truncated`` are not model-output errors.
    """
    return isinstance(exc.__cause__, _MODEL_OUTPUT_EXCEPTIONS) or exc.code == "empty_model_response"


def backend_error_code(exc: LlamaCppError) -> str:
    if isinstance(exc, CompletionTruncatedError):
        return "completion_truncated"
    if isinstance(exc, LlamaCppTimeoutError):
        return "backend_timeout"
    if isinstance(exc, LlamaCppConnectionError):
        return "backend_connection_error"
    return "backend_error"


class LiveConfigError(RuntimeError):
    """The environment does not describe a usable localhost model server."""


@dataclass(frozen=True)
class LiveSettings:
    """Where the already-running llama-server is, read from the environment."""

    base_url: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        api_key: str | None = None,
    ) -> LiveSettings:
        env = os.environ if environ is None else environ
        base_url = (env.get("TOOLSMITH_BASE_URL") or "").strip()
        if not base_url:
            raise LiveConfigError("TOOLSMITH_BASE_URL is not set")
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or parts.hostname not in LOCAL_HOSTS:
            raise LiveConfigError(
                "TOOLSMITH_BASE_URL must be an http(s) URL on localhost; "
                f"got scheme {parts.scheme!r} and host {parts.hostname!r}"
            )
        key = (api_key or env.get("LOCAL_KEY") or "").strip()
        if not key:
            raise LiveConfigError("No API key: set LOCAL_KEY or pass api_key")
        return cls(base_url=base_url.rstrip("/"), api_key=key)


def build_llama_client(
    settings: LiveSettings,
    *,
    connections: int,
    timeout: float = 600.0,
) -> LlamaCppClient:
    """A shared ``LlamaCppClient`` whose pool admits ``connections`` concurrent requests.

    Pass at least the server's ``--parallel`` so requests never queue inside the
    client while llama-server slots sit idle.
    """
    if connections < 1:
        raise ValueError("connections must be positive")
    limits = httpx.Limits(
        max_connections=connections,
        max_keepalive_connections=connections,
    )
    return LlamaCppClient(
        settings.base_url,
        settings.api_key,
        timeout=timeout,
        transport=httpx.HTTPTransport(limits=limits),
    )


def worker_service(
    llama_client: LlamaCppClient,
    renderer: ChatPromptRenderer,
    profile: ModelProfile,
    *,
    use_grammars: bool = True,
) -> ToolsmithService:
    """One ``ToolsmithService`` per worker thread. The recorder stays unset: the journal is the record."""
    return ToolsmithService(
        llama_client,
        renderer,
        profile=profile,
        use_grammars=use_grammars,
    )


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "LOCAL_HOSTS",
    "LiveConfigError",
    "LiveSettings",
    "SteadyhandClient",
    "backend_error_code",
    "build_llama_client",
    "classify_completion",
    "is_model_output_error",
    "worker_service",
]
