"""A JSONSchemaBench engine backed by the Toolsmith shim.

``JSONSchemaEngineCore`` sends one ``response_format=json_schema`` request per
schema and maps the outcome to JSONSchemaBench's compile and decoding statuses.
``create_engine_class`` wraps it in JSONSchemaBench's ``Engine`` interface; it
is built lazily because the benchmark package is installed only on Colab. A
serving failure raises instead of being scored, so a dead server cannot count
as failed generations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..api import APIError
from .models import Mode
from .policies import exception_error_code, is_infrastructure_error
from .runner import InfrastructureError


@dataclass(frozen=True)
class JSONGenerationResult:
    """One generation and its compile/decoding status, in JSONSchemaBench's terms."""

    generation: str
    compile_status: str
    decoding_status: str
    input_tokens: int
    output_tokens: int
    error: str | None = None
    ttft: None = None
    gct: None = None


class JSONSchemaEngineCore:
    """Ask the shim for a JSON answer matching each schema, in the configured mode."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        mode: Mode = Mode.GRAMMAR,
        max_tokens: int = 1024,
        seed: int = 0,
    ) -> None:
        self.client = client
        self.model = model
        self.mode = mode
        self.max_tokens = max_tokens
        self.seed = seed

    def generate(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> JSONGenerationResult:
        """One structured-output request. Unsupported schemas count as compile failures."""
        thinking = self.mode is Mode.THINKING
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "jsonschemabench",
                        "schema": schema,
                        "strict": True,
                    },
                },
                temperature=0.0,
                seed=self.seed,
                max_tokens=self.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
            ).model_dump()
        except APIError as exc:
            if is_infrastructure_error(exc):
                raise InfrastructureError("jsonschema", f"{type(exc).__name__}: {exc}") from exc
            compile_failure = exc.code in {"unsupported_schema", "incompatible_response_format"}
            return JSONGenerationResult(
                "",
                "unsupported_schema" if compile_failure else "runtime_grammar_error",
                "bad_api_response" if not compile_failure else "not_run",
                0,
                0,
                str(exc),
            )
        except Exception as exc:
            # A dead or hung server is not a failed generation: the split is redone after a restart.
            if is_infrastructure_error(exc):
                raise InfrastructureError("jsonschema", f"{type(exc).__name__}: {exc}") from exc
            code = exception_error_code(exc)
            compile_failure = code in {
                "unsupported_schema",
                "incompatible_response_format",
            }
            return JSONGenerationResult(
                "",
                "unsupported_schema" if compile_failure else "runtime_grammar_error",
                "not_run" if compile_failure else "bad_api_response",
                0,
                0,
                f"{type(exc).__name__}: {exc}",
            )
        message = response["choices"][0]["message"]
        usage = response.get("usage", {})
        return JSONGenerationResult(
            message.get("content") or "",
            "ok",
            "ok",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )


def create_engine_class() -> type:
    """JSONSchemaBench's ``Engine`` subclass for Toolsmith (its ``Config`` is attached as ``.Config``)."""
    try:
        from core.engine import Engine, EngineConfig
        from core.types import (
            CompileStatus,
            CompileStatusCode,
            DecodingStatus,
            DecodingStatusCode,
        )
    except ImportError as exc:
        raise ImportError("Install the pinned JSONSchemaBench core package") from exc

    @dataclass
    class ToolsmithEngineConfig(EngineConfig):
        client: Any
        model: str
        tokenizer: Any
        mode: Mode = Mode.GRAMMAR
        context_size: int = 16384
        max_tokens: int = 1024
        seed: int = 0

    class ToolsmithJSONSchemaEngine(Engine[ToolsmithEngineConfig]):
        name = "toolsmith"

        def __init__(self, config):
            super().__init__(config)
            self.core = JSONSchemaEngineCore(
                config.client,
                config.model,
                mode=config.mode,
                max_tokens=config.max_tokens,
                seed=config.seed,
            )

        def _generate(self, output) -> None:
            result = self.core.generate(output.messages, output.schema)
            output.generation = result.generation
            output.token_usage.input_tokens = result.input_tokens
            output.token_usage.output_tokens = result.output_tokens
            compile_code = {
                "ok": CompileStatusCode.OK,
                "unsupported_schema": CompileStatusCode.UNSUPPORTED_SCHEMA,
            }.get(result.compile_status, CompileStatusCode.RUNTIME_GRAMMAR_ERROR)
            decoding_code = {
                "ok": DecodingStatusCode.OK,
                "not_run": DecodingStatusCode.TBD,
            }.get(result.decoding_status, DecodingStatusCode.BAD_API_RESPONSE)
            output.metadata.compile_status = CompileStatus(
                code=compile_code, message=result.error
            )
            output.metadata.decoding_status = DecodingStatus(
                code=decoding_code, message=result.error
            )

        def encode(self, text):
            return self.config.tokenizer.encode(text, add_special_tokens=False)

        def decode(self, ids):
            return self.config.tokenizer.decode(ids, skip_special_tokens=True)

        @property
        def max_context_length(self):
            return self.config.context_size

        def close(self):
            self.config.client.close()

    ToolsmithJSONSchemaEngine.Config = ToolsmithEngineConfig
    return ToolsmithJSONSchemaEngine


__all__ = ["JSONGenerationResult", "JSONSchemaEngineCore", "create_engine_class"]
