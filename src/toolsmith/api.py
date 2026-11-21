from __future__ import annotations

import hmac
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .backends import CompletionTruncatedError, LlamaCppClient, LlamaCppError
from .deltas import DONE, OpenAIStreamAdapter, sse_error, sse_json
from .grammar import (
    CompiledGrammar,
    GBNFCompiler,
    GrammarConfigurationError,
    SchemaNotSupportedError,
    ToolChoice,
    resolve_tool_choice,
)
from .parsing import (
    ParsedResponse,
    Qwen3IncrementalParser,
    Qwen3ToolCallParser,
    ToolCallParseError,
)
from .prompting import ChatPromptRenderer, normalize_history, with_response_schema
from .protocols import ModelProfile, QWEN3_PROFILE, ToolProtocol
from .qwen35 import Qwen35IncrementalParser, Qwen35ToolCallParser
from .recording import JsonlRecorder
from .registry import Tool
from .repair import RepairPolicy, build_repair_messages
from .validation import (
    StrictJSONError,
    StructuredOutputValidationError,
    StructuredOutputValidator,
    ToolCallValidationError,
    ToolCallValidator,
    strict_json_loads,
)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "Qwen/Qwen3-8B"
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = "auto"
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    max_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.6, ge=0)
    seed: int | None = None
    response_format: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class APIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        code: str = "invalid_request",
        repair_attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.repair_attempts = repair_attempts

    def body(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "message": str(self),
            "type": self.error_type,
            "code": self.code,
        }
        if self.repair_attempts:
            error["repair_attempts"] = self.repair_attempts
        return {"error": error}


@dataclass(frozen=True)
class _PreparedRequest:
    tools: tuple[Tool, ...]
    messages: list[dict[str, Any]]
    prompt: str
    choice: ToolChoice
    grammar: CompiledGrammar | None
    validator: ToolCallValidator
    structured_validator: StructuredOutputValidator | None


_ModelOutputError = (
    ToolCallParseError,
    ToolCallValidationError,
    StructuredOutputValidationError,
)


class ToolsmithService:
    def __init__(
        self,
        client: LlamaCppClient,
        renderer: ChatPromptRenderer,
        *,
        parser: Qwen3ToolCallParser | None = None,
        parser_factory: Callable[[bool | None], Qwen3IncrementalParser] | None = None,
        recorder: JsonlRecorder | None = None,
        profile: ModelProfile = QWEN3_PROFILE,
        repair_policy: RepairPolicy | None = None,
        use_grammars: bool = True,
        validator_factory: Callable[[tuple[Tool, ...], ToolChoice], Any] | None = None,
    ) -> None:
        self.client = client
        self.renderer = renderer
        self.profile = profile
        self.parser = parser
        self.parser_factory = parser_factory
        self.recorder = recorder
        self.repair_policy = repair_policy or RepairPolicy()
        self.use_grammars = use_grammars
        self.validator_factory = validator_factory or ToolCallValidator
        self.grammar_compiler = GBNFCompiler(protocol=profile.protocol)

    def chat(self, request: ChatCompletionRequest) -> dict[str, Any]:
        prepared = self._prepare(request)
        messages = prepared.messages
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for attempt in range(self.repair_policy.max_repairs + 1):
            prompt = (
                prepared.prompt
                if attempt == 0
                else self._render(request, prepared, messages)
            )
            try:
                raw_response = self.client.complete(
                    prompt,
                    **self._generation_settings(request, prepared.grammar),
                )
            except CompletionTruncatedError as exc:
                error = APIError(
                    "The model response was truncated before it could be parsed",
                    status_code=500,
                    error_type="server_error",
                    code="completion_truncated",
                    repair_attempts=attempt,
                )
                self._record_non_stream(
                    request,
                    prepared,
                    prompt,
                    exc.response,
                    None,
                    "not_run",
                    "error",
                    attempt,
                    None,
                    total_usage,
                    error,
                )
                raise error from exc

            content = raw_response.get("content")
            if not isinstance(content, str):
                raise APIError(
                    "llama.cpp response did not contain string content",
                    status_code=502,
                    error_type="server_error",
                    code="invalid_backend_response",
                    repair_attempts=attempt,
                )
            usage = self._usage(raw_response)
            for key in total_usage:
                total_usage[key] += usage[key]
            validation = "not_run"
            try:
                parsed = self._parse(request, prepared, content)
                self._validate_parsed(prepared, parsed)
                validation = "passed"
                if not parsed.tool_calls and not parsed.content:
                    raise APIError(
                        "Model returned an empty response",
                        status_code=500,
                        error_type="server_error",
                        code="empty_model_response",
                    )
            except _ModelOutputError as exc:
                error = self._model_output_error(exc, attempt)
                can_repair = attempt < self.repair_policy.max_repairs
                self._record_non_stream(
                    request,
                    prepared,
                    prompt,
                    raw_response,
                    None,
                    "failed",
                    "retry" if can_repair else "error",
                    attempt,
                    usage,
                    total_usage,
                    error,
                )
                if can_repair:
                    messages = build_repair_messages(
                        messages,
                        content,
                        str(exc),
                        self.repair_policy,
                    )
                    continue
                raise error from exc
            except APIError as exc:
                exc.repair_attempts = attempt
                can_repair = (
                    exc.code == "empty_model_response"
                    and attempt < self.repair_policy.max_repairs
                )
                self._record_non_stream(
                    request,
                    prepared,
                    prompt,
                    raw_response,
                    None,
                    validation,
                    "retry" if can_repair else "error",
                    attempt,
                    usage,
                    total_usage,
                    exc,
                )
                if can_repair:
                    messages = build_repair_messages(
                        messages,
                        content,
                        str(exc),
                        self.repair_policy,
                    )
                    continue
                raise

            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": self._message(parsed),
                        "finish_reason": "tool_calls" if parsed.tool_calls else "stop",
                    }
                ],
                "usage": total_usage.copy(),
                "toolsmith": {
                    "repair_attempts": attempt,
                    "grammar_sha256": (
                        prepared.grammar.sha256 if prepared.grammar is not None else None
                    ),
                    "protocol": self.profile.protocol.value,
                },
            }
            self._record_non_stream(
                request,
                prepared,
                prompt,
                raw_response,
                response,
                validation,
                "repaired" if attempt else "success",
                attempt,
                usage,
                total_usage,
            )
            return response

        raise AssertionError("Repair loop ended without a response")

    def chat_stream(self, request: ChatCompletionRequest) -> Iterator[str]:
        prepared = self._prepare(request)
        return self._stream_completion(request, prepared)

    def _stream_completion(
        self,
        request: ChatCompletionRequest,
        prepared: _PreparedRequest,
    ) -> Iterator[str]:
        include_usage = bool(
            request.stream_options and request.stream_options.get("include_usage")
        )
        adapter = OpenAIStreamAdapter(request.model, include_usage=include_usage)
        expect_thinking = request.chat_template_kwargs.get("enable_thinking")
        parser = self._incremental_parser(prepared, expect_thinking)
        raw_events: list[dict[str, Any]] = []
        upstream: Iterator[dict[str, Any]] | None = None
        outcome = "cancelled"
        validation_outcome = "not_run"
        stream_error: APIError | None = None
        response_message: dict[str, Any] | None = None
        usage: dict[str, int] | None = None

        try:
            upstream = self.client.stream_complete(
                prepared.prompt,
                **self._generation_settings(request, prepared.grammar),
            )
            yield sse_json(adapter.role_chunk())
            for raw_event in upstream:
                raw_events.append(raw_event)
                for event in parser.feed(raw_event["content"]):
                    chunk = adapter.convert(event)
                    if chunk is not None:
                        yield sse_json(chunk)

            for event in parser.finish():
                chunk = adapter.convert(event)
                if chunk is not None:
                    yield sse_json(chunk)
            parsed = parser.response
            try:
                self._validate_parsed(prepared, parsed)
            except (ToolCallValidationError, StructuredOutputValidationError):
                validation_outcome = "failed"
                raise
            validation_outcome = "passed"
            if not parsed.tool_calls and not parsed.content:
                raise APIError(
                    "Model returned an empty response",
                    status_code=500,
                    error_type="server_error",
                    code="empty_model_response",
                )

            response_message = self._message(parsed)
            terminal = raw_events[-1] if raw_events else {}
            usage = self._usage(terminal)
            yield sse_json(adapter.finish_chunk(parsed))
            usage_chunk = adapter.usage_chunk(usage)
            if usage_chunk is not None:
                yield sse_json(usage_chunk)
            outcome = "success"
            yield DONE
        except CompletionTruncatedError as exc:
            stream_error = APIError(
                "The model response was truncated before it could be parsed",
                status_code=500,
                error_type="server_error",
                code="completion_truncated",
            )
        except ToolCallParseError as exc:
            stream_error = APIError(
                f"Could not parse model output: {exc}",
                status_code=500,
                error_type="server_error",
                code=exc.code,
            )
        except (ToolCallValidationError, StructuredOutputValidationError) as exc:
            stream_error = APIError(
                f"Model output failed validation: {exc}",
                status_code=500,
                error_type="server_error",
                code=exc.code,
            )
        except LlamaCppError as exc:
            stream_error = APIError(
                f"Model backend failed: {exc}",
                status_code=502,
                error_type="server_error",
                code="backend_error",
            )
        except APIError as exc:
            stream_error = exc
        finally:
            if stream_error is not None:
                outcome = "error"
                yield sse_error(
                    str(stream_error),
                    error_type=stream_error.error_type,
                    code=stream_error.code,
                )
                yield DONE
            if upstream is not None:
                close = getattr(upstream, "close", None)
                if close is not None:
                    close()
            if self.recorder is not None:
                self.recorder.append(
                    {
                        "event": "chat_completion_stream",
                        "request": request.model_dump(mode="json"),
                        "profile": self.profile.name,
                        "protocol": self.profile.protocol.value,
                        "raw_prompt": prepared.prompt,
                        "raw_events": raw_events,
                        "tool_choice": prepared.choice.as_dict(),
                        **self._grammar_details(prepared.grammar),
                        "validation": validation_outcome,
                        "repair_attempts": 0,
                        "outcome": outcome,
                        "response": response_message,
                        "usage": usage,
                        "error": stream_error.body() if stream_error is not None else None,
                    }
                )

    def _prepare(self, request: ChatCompletionRequest) -> _PreparedRequest:
        self._validate_options(request)
        if request.model != self.profile.model:
            raise APIError(
                f"This service exposes {self.profile.model!r}, not {request.model!r}",
                code="invalid_model",
            )
        try:
            tools = tuple(Tool.from_openai(definition) for definition in request.tools)
        except (TypeError, ValueError) as exc:
            raise APIError(f"Invalid tool definition: {exc}") from exc

        self._validate_messages(request.messages)
        try:
            choice = resolve_tool_choice(tools, request.tool_choice)
        except GrammarConfigurationError as exc:
            raise APIError(str(exc), code="invalid_tool_choice") from exc

        response_format = request.response_format
        structured_schema = self._structured_schema(response_format)
        if structured_schema is not None and tools:
            raise APIError(
                "response_format=json_schema cannot be combined with tools in this milestone",
                code="incompatible_response_format",
            )
        enable_thinking = request.chat_template_kwargs.get("enable_thinking")
        constrained = self.use_grammars and (bool(tools) or structured_schema is not None)
        if constrained and enable_thinking is not False:
            raise APIError(
                "Constrained tool or schema requests require "
                "chat_template_kwargs.enable_thinking=false",
                code="thinking_with_grammar_unsupported",
            )
        try:
            if not self.use_grammars:
                compiled = None
            elif structured_schema is not None:
                compiled = self.grammar_compiler.compile_schema(structured_schema)
            else:
                compiled = self.grammar_compiler.compile_tools(tools, choice)
        except SchemaNotSupportedError as exc:
            raise APIError(
                f"Unsupported tool or response schema: {exc}",
                code="unsupported_schema",
            ) from exc
        except GrammarConfigurationError as exc:
            raise APIError(str(exc), code="grammar_compilation_failed") from exc
        grammar = compiled if self.use_grammars else None

        try:
            messages = normalize_history(request.messages, self.profile.protocol)
        except ValueError as exc:
            raise APIError(str(exc), code="invalid_message_history") from exc
        if response_format is not None and structured_schema is not None:
            messages = with_response_schema(messages, response_format)
        prompt = self.renderer.render(
            messages,
            [tool.to_openai() for tool in tools],
            enable_thinking=enable_thinking,
        )
        structured_validator = (
            StructuredOutputValidator(structured_schema)
            if structured_schema is not None
            else None
        )
        return _PreparedRequest(
            tools,
            messages,
            prompt,
            choice,
            grammar,
            self.validator_factory(tools, choice),
            structured_validator,
        )

    def _render(
        self,
        request: ChatCompletionRequest,
        prepared: _PreparedRequest,
        messages: list[dict[str, Any]],
    ) -> str:
        return self.renderer.render(
            messages,
            [tool.to_openai() for tool in prepared.tools],
            enable_thinking=request.chat_template_kwargs.get("enable_thinking"),
        )

    def _parse(
        self,
        request: ChatCompletionRequest,
        prepared: _PreparedRequest,
        content: str,
    ) -> ParsedResponse:
        expect_thinking = request.chat_template_kwargs.get("enable_thinking")
        if self.profile.protocol is ToolProtocol.QWEN35:
            return Qwen35ToolCallParser(
                prepared.tools,
                expect_thinking=expect_thinking,
            ).parse(content)
        parser = self.parser or Qwen3ToolCallParser(expect_thinking=expect_thinking)
        return parser.parse(content)

    def _incremental_parser(
        self,
        prepared: _PreparedRequest,
        expect_thinking: bool | None,
    ) -> Qwen3IncrementalParser | Qwen35IncrementalParser:
        if self.profile.protocol is ToolProtocol.QWEN35:
            return Qwen35IncrementalParser(
                prepared.tools,
                expect_thinking=expect_thinking,
            )
        if self.parser_factory is not None:
            return self.parser_factory(expect_thinking)
        return Qwen3IncrementalParser(expect_thinking=expect_thinking)

    @staticmethod
    def _validate_parsed(prepared: _PreparedRequest, parsed: ParsedResponse) -> None:
        if prepared.structured_validator is None:
            prepared.validator.validate(parsed)
            return
        if parsed.tool_calls:
            raise StructuredOutputValidationError(
                "A structured response cannot contain tool calls",
                code="unexpected_tool_call",
            )
        if parsed.content is None:
            raise StructuredOutputValidationError(
                "The structured response is empty",
                code="invalid_structured_output_json",
            )
        prepared.structured_validator.validate(parsed.content)

    @staticmethod
    def _model_output_error(exc: Exception, attempt: int) -> APIError:
        if isinstance(exc, ToolCallParseError):
            message = f"Could not parse model output: {exc}"
        elif isinstance(exc, ToolCallValidationError):
            message = f"Model output failed validation: {exc}"
        elif isinstance(exc, StructuredOutputValidationError):
            message = f"Structured output failed validation: {exc}"
        else:
            raise TypeError(type(exc))
        return APIError(
            message[:4000],
            status_code=500,
            error_type="server_error",
            code=exc.code,
            repair_attempts=attempt,
        )

    @staticmethod
    def _generation_settings(
        request: ChatCompletionRequest,
        grammar: CompiledGrammar | None,
    ) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "seed": request.seed,
        }
        if grammar is not None:
            settings["grammar"] = grammar.text
        return settings

    @staticmethod
    def _grammar_details(grammar: CompiledGrammar | None) -> dict[str, Any]:
        return {
            "grammar": grammar.text if grammar is not None else None,
            "grammar_sha256": grammar.sha256 if grammar is not None else None,
            "grammar_mode": grammar.mode if grammar is not None else None,
            "grammar_tool_names": grammar.tool_names if grammar is not None else (),
            "grammar_protocol": grammar.protocol if grammar is not None else None,
            "grammar_limitations": grammar.limitations if grammar is not None else (),
        }

    def _record_non_stream(
        self,
        request: ChatCompletionRequest,
        prepared: _PreparedRequest,
        prompt: str,
        raw_response: dict[str, Any],
        response: dict[str, Any] | None,
        validation: str,
        outcome: str,
        attempt: int,
        usage: dict[str, int] | None,
        total_usage: dict[str, int],
        error: APIError | None = None,
    ) -> None:
        if self.recorder is None:
            return
        self.recorder.append(
            {
                "event": "chat_completion",
                "request": request.model_dump(mode="json"),
                "profile": self.profile.name,
                "protocol": self.profile.protocol.value,
                "raw_prompt": prompt,
                "raw_response": raw_response,
                "tool_choice": prepared.choice.as_dict(),
                **self._grammar_details(prepared.grammar),
                "validation": validation,
                "attempt": attempt,
                "repair_attempts": attempt,
                "outcome": outcome,
                "response": response,
                "usage": usage,
                "total_usage": total_usage.copy(),
                "error": error.body() if error is not None else None,
            }
        )

    @staticmethod
    def _message(parsed: ParsedResponse) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
        if parsed.reasoning_content is not None:
            message["reasoning_content"] = parsed.reasoning_content
        if parsed.tool_calls:
            message["tool_calls"] = [call.to_openai() for call in parsed.tool_calls]
        return message

    @classmethod
    def _usage(cls, response: dict[str, Any]) -> dict[str, int]:
        prompt_tokens = cls._token_count(response, "prompt_n", "tokens_evaluated")
        completion_tokens = cls._token_count(response, "predicted_n", "tokens_predicted")
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def _token_count(response: dict[str, Any], timing_key: str, response_key: str) -> int:
        timings = response.get("timings")
        value = timings.get(timing_key) if isinstance(timings, dict) else None
        if value is None:
            value = response.get(response_key, 0)
        return int(value or 0)

    @staticmethod
    def _structured_schema(
        response_format: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        if response_format is None:
            return None
        kind = response_format.get("type")
        if kind == "text":
            if set(response_format) != {"type"}:
                raise APIError("A text response_format contains unsupported fields")
            return None
        if kind != "json_schema" or set(response_format) != {"type", "json_schema"}:
            raise APIError(
                "Only response_format={'type': 'json_schema', ...} or text is supported",
                code="unsupported_response_format",
            )
        specification = response_format.get("json_schema")
        if not isinstance(specification, Mapping):
            raise APIError("response_format.json_schema must be an object")
        required = {"name", "schema", "strict"}
        if not required <= set(specification) or not set(specification) <= required | {"description"}:
            raise APIError(
                "json_schema requires name, schema, strict and optional description only",
                code="invalid_response_format",
            )
        if not isinstance(specification["name"], str) or not specification["name"].strip():
            raise APIError("json_schema.name must be a non-empty string")
        if specification["strict"] is not True:
            raise APIError("json_schema.strict must be true")
        if "description" in specification and not isinstance(specification["description"], str):
            raise APIError("json_schema.description must be a string")
        schema = specification["schema"]
        if not isinstance(schema, Mapping):
            raise APIError("json_schema.schema must be an object")
        return schema

    @staticmethod
    def _validate_options(request: ChatCompletionRequest) -> None:
        if request.stream_options is not None:
            if not request.stream:
                raise APIError("stream_options requires stream=true")
            unsupported_stream_options = set(request.stream_options) - {"include_usage"}
            if unsupported_stream_options:
                names = ", ".join(sorted(unsupported_stream_options))
                raise APIError(f"Unsupported stream options: {names}")
            include_usage = request.stream_options.get("include_usage")
            if not isinstance(include_usage, bool):
                raise APIError("stream_options.include_usage must be a boolean")
        unsupported_template_options = set(request.chat_template_kwargs) - {"enable_thinking"}
        if unsupported_template_options:
            names = ", ".join(sorted(unsupported_template_options))
            raise APIError(f"Unsupported chat template options: {names}")
        enable_thinking = request.chat_template_kwargs.get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise APIError("chat_template_kwargs.enable_thinking must be a boolean")

    @staticmethod
    def _validate_messages(messages: list[dict[str, Any]]) -> None:
        pending_tool_ids: set[str] = set()
        all_tool_ids: set[str] = set()
        for index, message in enumerate(messages):
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise APIError(f"Message {index} has an unsupported role: {role!r}")
            content = message.get("content")
            if not isinstance(content, str) and not (
                role == "assistant" and content is None
            ):
                raise APIError(
                    f"Message {index} needs string content; multimodal content is unsupported"
                )
            if role == "system" and index != 0:
                raise APIError("A system message is only allowed at the beginning")
            if role == "assistant":
                calls = message.get("tool_calls", [])
                if not isinstance(calls, list):
                    raise APIError(f"Assistant message {index} tool_calls must be a list")
                for call_index, call in enumerate(calls):
                    if not isinstance(call, Mapping) or call.get("type") != "function":
                        raise APIError(f"Assistant call {call_index} must be a function call")
                    call_id = call.get("id")
                    function = call.get("function")
                    if not isinstance(call_id, str) or not call_id or call_id in all_tool_ids:
                        raise APIError(f"Assistant call {call_index} needs a unique non-empty id")
                    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
                        raise APIError(f"Assistant call {call_index} needs a function name")
                    arguments = function.get("arguments")
                    if not isinstance(arguments, str):
                        raise APIError(f"Assistant call {call_index} needs string arguments")
                    try:
                        decoded = strict_json_loads(arguments)
                    except StrictJSONError as exc:
                        raise APIError(
                            f"Assistant call {call_index} has invalid JSON arguments: {exc}"
                        ) from exc
                    if not isinstance(decoded, dict):
                        raise APIError(f"Assistant call {call_index} arguments must be an object")
                    all_tool_ids.add(call_id)
                    pending_tool_ids.add(call_id)
            if role == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in pending_tool_ids:
                    raise APIError(
                        f"Tool message {index} needs an unanswered preceding tool_call_id"
                    )
                pending_tool_ids.remove(call_id)


def create_app(service: ToolsmithService, *, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="Toolsmith")
    app.state.service = service

    @app.middleware("http")
    async def authenticate(request: Request, call_next: Any) -> Any:
        if api_key is not None:
            expected = f"Bearer {api_key}"
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, expected):
                error = APIError(
                    "Invalid or missing API key",
                    status_code=401,
                    error_type="authentication_error",
                    code="invalid_api_key",
                )
                return JSONResponse(error.body(), status_code=401)
        return await call_next(request)

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status_code)

    @app.exception_handler(LlamaCppError)
    async def handle_backend_error(request: Request, exc: LlamaCppError) -> JSONResponse:
        error = APIError(
            f"Model backend failed: {exc}",
            status_code=502,
            error_type="server_error",
            code="backend_error",
        )
        return JSONResponse(error.body(), status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        error = APIError("The chat completion request is invalid")
        return JSONResponse(error.body(), status_code=400)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.client.health()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": service.profile.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "toolsmith",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> Any:
        if request.stream:
            return StreamingResponse(
                service.chat_stream(request),
                media_type="text/event-stream",
            )
        return service.chat(request)

    return app
