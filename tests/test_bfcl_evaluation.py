import pytest

from toolsmith.api import ChatCompletionRequest, ToolsmithService
from toolsmith.evaluation.bfcl_adapter import (
    constraint_support,
    decoded_calls,
    eligible_ids,
    openai_message_to_bfcl_result,
    result_entry,
)
from toolsmith.evaluation.bfcl_handler import decode_standardized_result
from toolsmith.evaluation.bfcl_runner import BFCLRunner
from toolsmith.evaluation.models import CaseResult, Mode, RunConfig
from toolsmith.evaluation.policies import (
    DeferredSchemaValidator,
    repair_eligible,
    reuse_grammar_result,
    service_options,
)
from toolsmith.protocols import ToolProtocol

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def response(content=None, calls=None):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


def test_bfcl_result_and_decoder_match_openai_fc_shape():
    raw = response(
        calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }]
    )
    entry = result_entry("simple_python_1", raw, latency=1.5)

    assert entry["id"] == "simple_python_1"
    assert entry["result"] == [{"get_weather": '{"city":"Paris"}'}]
    assert decode_standardized_result(entry["result"]) == [
        {"get_weather": {"city": "Paris"}}
    ]
    assert decoded_calls(raw) == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    assert entry["input_token_count"] == 10


def test_irrelevance_text_stays_text_and_non_list_decode_fails():
    assert openai_message_to_bfcl_result({"content": "No tool"}) == "No tool"
    with pytest.raises(TypeError):
        decode_standardized_result("No tool")


def test_unsupported_schema_has_pointer_and_is_not_eligible():
    bad = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "pattern": "^Paris$"}},
                "additionalProperties": False,
            },
        },
    }
    support = constraint_support([bad], ToolProtocol.QWEN3)

    assert not support.eligible
    assert support.error["pointer"] == "/properties/city/pattern"
    assert not eligible_ids([
        CaseResult("x", "simple", "m", "grammar", "constraint_unsupported")
    ])


class FakeCompletions:
    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("Response", (), {"model_dump": lambda self: response(
            calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }]
        )})()


class FakeOpenAI:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_bfcl_runner_writes_standard_result_metadata():
    runner = BFCLRunner(
        FakeOpenAI(),
        RunConfig("run", "bfcl", "Qwen/Qwen3-8B", Mode.GRAMMAR),
        tools_converter=lambda functions: [WEATHER],
    )
    entry = {"id": "simple_python_1", "question": [[{"role": "user", "content": "Paris?"}]], "function": []}

    result = runner.execute(entry)

    assert result.status == "success"
    assert result.metadata["bfcl_result"]["result"][0]["get_weather"]
    assert result.grammar_sha256


def test_prompt_policy_defers_schema_without_compiling_pattern():
    class Client:
        def complete(self, prompt, **settings):
            self.settings = settings
            return {"content": "Done"}

    class Renderer:
        def render(self, messages, tools=(), *, enable_thinking=None):
            return "prompt"

    client = Client()
    service = ToolsmithService(client, Renderer(), **service_options(Mode.PROMPT, defer_bfcl_schema=True))
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hi"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string", "pattern": "^x$"}},
                    "additionalProperties": False,
                },
            },
        }],
        chat_template_kwargs={"enable_thinking": False},
    )

    assert service.chat(request)["choices"][0]["message"]["content"] == "Done"
    assert "grammar" not in client.settings
    assert isinstance(DeferredSchemaValidator, type)


def test_repair_reuse_never_retries_backend_or_truncation():
    base = dict(case_id="x", category="simple", model="m", mode="grammar", status="error")
    assert not repair_eligible(CaseResult(**base, error={"code": "backend_error"}))
    assert not repair_eligible(CaseResult(**base, error={"code": "completion_truncated"}))
    assert repair_eligible(CaseResult(**base, error={"code": "invalid_tool_call"}))
    assert reuse_grammar_result(
        CaseResult(**{**base, "status": "success"}, parse_valid=True, schema_valid=True)
    )
