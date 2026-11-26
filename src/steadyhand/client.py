"""Model-turn adapter over ``ToolsmithService.chat``, plus live-server wiring.

Prompt rendering, native tool-call parsing, grammars, and the shim's one-shot
model-output repair all stay inside ``toolsmith``. This module only builds the
request, calls ``chat()`` (never ``chat_stream()``), and classifies the result.
A service built with ``generation_path="buffered_stream"`` measures TTFT inside
``chat()`` and still returns one complete response, so nothing here streams.
Each turn carries the service's per-attempt evidence as ``ModelTurn.metrics``.

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
from dataclasses import dataclass, field, replace
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

from .controller import has_visible_character
from .types import ModelTurn, ToolCall, TurnKind, TurnMetrics

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
        slot_context: int | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if slot_context is not None and slot_context <= max_tokens:
            raise ValueError("slot_context must be larger than max_tokens")
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
        # Tokens one llama-server slot holds (-c divided by --parallel), when known.
        self.slot_context = slot_context

    @property
    def profile(self) -> ModelProfile:
        """The model profile of the wrapped service."""
        return self.service.profile

    @property
    def generation_path(self) -> str:
        """``non_streaming`` or ``buffered_stream`` (the service's TTFT path)."""
        return getattr(self.service, "generation_path", "non_streaming")

    def settings(self) -> dict[str, Any]:
        """What the journal records about this model configuration.

        ``generation_path`` appears only when it is not the default, so legacy
        episodes keep their journals and ids.
        """
        settings = {
            "profile": self.profile.name,
            "model": self.profile.model,
            "protocol": self.profile.protocol.value,
            "use_grammars": bool(self.service.use_grammars),
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "slot_context": self.slot_context,
        }
        if self.generation_path != "non_streaming":
            settings["generation_path"] = self.generation_path
        return settings

    def request(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ChatCompletionRequest:
        """The shim request ``complete()`` sends for these messages."""
        return ChatCompletionRequest(
            model=self.profile.model,
            messages=deepcopy([dict(message) for message in messages]),
            tools=deepcopy([dict(tool) for tool in tools]),
            tool_choice=tool_choice,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )

    def render_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> str:
        """The exact prompt text ``complete()`` would send first (the model's own chat template)."""
        return self.service.render_prompt(self.request(messages, tools, tool_choice=tool_choice))

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
        id_slot: int | None = None,
    ) -> ModelTurn:
        """One chat completion through the shim, as a ``ModelTurn`` (failures become failure turns).

        ``id_slot`` pins the request to a llama-server slot (a scheduler lease).
        """
        request = self.request(messages, tools, tool_choice=tool_choice)
        started = time.perf_counter()

        def metrics(attempts: Sequence[Mapping[str, Any]] = ()) -> TurnMetrics:
            return TurnMetrics.from_toolsmith(
                attempts,
                response_elapsed_ms=(time.perf_counter() - started) * 1000.0,
                generation_path=self.generation_path,
            )

        try:
            response = self.service.chat(request) if id_slot is None else self.service.chat(request, id_slot=id_slot)
        except APIError as exc:
            message = str(exc)
            if exc.code == "completion_truncated" and self.slot_context is not None:
                message += f" (server slot holds {self.slot_context} tokens; max_tokens {self.max_tokens})"
            return ModelTurn.failure(
                TurnKind.PARSE_ERROR if is_model_output_error(exc) else TurnKind.FATAL_ERROR,
                exc.code,
                message,
                raw_output=getattr(exc, "raw_output", None),
                repair_attempts=exc.repair_attempts,
                elapsed_s=time.perf_counter() - started,
                metrics=metrics(getattr(exc, "attempt_metrics", ()) or ()),
            )
        except LlamaCppError as exc:
            return ModelTurn.failure(
                TurnKind.FATAL_ERROR,
                backend_error_code(exc),
                f"{type(exc).__name__}: {exc}",
                elapsed_s=time.perf_counter() - started,
                metrics=metrics(),
            )
        turn = classify_completion(response, elapsed_s=time.perf_counter() - started)
        attempts = (response.get("toolsmith") or {}).get("attempts") or ()
        return replace(turn, metrics=metrics(attempts if isinstance(attempts, list) else ()))


def classify_completion(response: Mapping[str, Any], *, elapsed_s: float = 0.0) -> ModelTurn:
    """Turn an OpenAI-shaped chat completion from the shim into a ``ModelTurn`` (metrics left unset)."""
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
    elif isinstance(content, str) and has_visible_character(content):
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
    """The abort code for a backend failure (a sweep reruns the timeout and connection ones)."""
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
        """Read ``TOOLSMITH_BASE_URL`` (must be localhost) and ``LOCAL_KEY`` (or ``api_key``)."""
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
    generation_path: str = "non_streaming",
) -> ToolsmithService:
    """One ``ToolsmithService`` per worker thread. The recorder stays unset: the journal is the record.

    ``generation_path="buffered_stream"`` measures TTFT (a manifest identity field for Tidemark runs).
    """
    return ToolsmithService(
        llama_client,
        renderer,
        profile=profile,
        use_grammars=use_grammars,
        generation_path=generation_path,
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
