import httpx
from openai import InternalServerError

from toolsmith.api import APIError
from toolsmith.evaluation.models import AttemptRecord, CaseResult, Mode, RunConfig
from toolsmith.evaluation.own_runner import OwnRunner, run_repair_with_reuse
from toolsmith.evaluation.policies import repair_eligible, reuse_grammar_result

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


def case(case_id="simple-001"):
    return {
        "case_id": case_id,
        "category": "simple",
        "messages": [{"role": "user", "content": "Paris"}],
        "tool_choice": "required",
        "expected_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
        "comparison": "sequential",
    }


class FakeCompletions:
    def create(self, **kwargs):
        return type("Response", (), {"model_dump": lambda self: {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }],
                }
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "toolsmith": {"repair_attempts": 0, "grammar_sha256": "abc"},
        }})()


class FakeOpenAI:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_own_runner_scores_exact_call_and_usage():
    runner = OwnRunner(FakeOpenAI(), RunConfig("run", "own", "model", Mode.GRAMMAR), [WEATHER])
    result = runner.execute(case())

    assert result.semantic_valid
    assert result.schema_valid
    assert result.grammar_sha256 == "abc"
    assert result.attempts[0].completion_tokens == 3


def test_prompt_mode_measures_schema_instead_of_inferring_it():
    runner = OwnRunner(FakeOpenAI(), RunConfig("run", "own", "model", Mode.PROMPT), [WEATHER])
    valid = runner.execute(case())
    invalid = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":1}'},
                }],
            }
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "toolsmith": {"repair_attempts": 0},
    }

    class WrongType:
        def create(self, **kwargs):
            return type("Response", (), {"model_dump": lambda self: invalid})()

    invalid_result = OwnRunner(
        type("Client", (), {"chat": type("Chat", (), {"completions": WrongType()})()})(),
        RunConfig("run", "own", "model", Mode.PROMPT),
        [WEATHER],
    ).execute(case())

    assert valid.schema_valid
    assert valid.semantic_valid
    assert invalid_result.status == "success"
    assert not invalid_result.schema_valid


def test_openai_sdk_error_keeps_nested_toolsmith_code():
    body = {
        "error": {
            "message": "Model output failed validation",
            "type": "server_error",
            "code": "schema_validation_failed",
        }
    }
    request = httpx.Request("POST", "http://127.0.0.1/v1/chat/completions")
    response = httpx.Response(500, json=body, request=request)

    class Broken:
        def create(self, **kwargs):
            raise InternalServerError("failed", response=response, body=body)

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Broken()})()})()
    result = OwnRunner(client, RunConfig("run", "own", "m", Mode.GRAMMAR), [WEATHER]).execute(case())

    assert result.error["code"] == "schema_validation_failed"
    assert repair_eligible(result)


def test_malformed_tool_arguments_are_recorded():
    class BadJSON:
        def create(self, **kwargs):
            return type("Response", (), {"model_dump": lambda self: {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{"},
                        }],
                    }
                }],
            }})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": BadJSON()})()})()
    result = OwnRunner(client, RunConfig("run", "own", "m", Mode.PROMPT), [WEATHER]).execute(case())

    assert result.status == "error"
    assert result.error["code"] == "invalid_tool_call_json"
    assert repair_eligible(result)


def test_no_call_is_not_schema_valid_when_a_call_is_expected():
    class TextOnly:
        def create(self, **kwargs):
            return type("Response", (), {"model_dump": lambda self: {
                "choices": [{"message": {"content": "No tool", "tool_calls": None}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "toolsmith": {"repair_attempts": 0},
            }})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": TextOnly()})()})()
    result = OwnRunner(client, RunConfig("run", "own", "m", Mode.PROMPT), [WEATHER]).execute(case())

    assert result.status == "success"
    assert not result.schema_valid
    assert result.semantic_valid is False
    assert not reuse_grammar_result(result)
    assert repair_eligible(result)


def test_repair_reuse_skips_success_and_backend_but_repairs_parse_failure(tmp_path):
    grammar = [
        CaseResult(
            "a", "simple", "m", "grammar", "success",
            parse_valid=True, schema_valid=True, semantic_valid=True,
        ),
        CaseResult("b", "simple", "m", "grammar", "error", error={"code": "backend_error"}),
        CaseResult("c", "simple", "m", "grammar", "error", error={"code": "invalid_tool_call"}),
        CaseResult(
            "d", "simple", "m", "grammar", "success",
            parse_valid=True, schema_valid=False, semantic_valid=False,
        ),
    ]
    cases = [case("a"), case("b"), case("c"), case("d")]
    runner = OwnRunner(FakeOpenAI(), RunConfig("run", "own", "m", Mode.REPAIR), [WEATHER])

    results = run_repair_with_reuse(grammar, cases, runner, tmp_path / "repair.jsonl", "hash")

    assert [result.mode for result in results] == ["repair", "repair", "repair", "repair"]
    assert results[0].attempts == ()
    assert results[1].status == "error"
    assert results[2].status == "success"
    assert results[3].status == "success"
    assert results[3].semantic_valid
    assert not results[3].attempts[0].reused
    resumed = run_repair_with_reuse(grammar, cases, runner, tmp_path / "repair.jsonl", "hash")
    assert len(resumed) == 4


def test_backend_error_is_explicit_result():
    class Broken:
        def create(self, **kwargs):
            raise APIError("backend", code="backend_error")

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Broken()})()})()
    result = OwnRunner(client, RunConfig("run", "own", "m", Mode.PROMPT), []).execute(case())

    assert result.status == "error"
    assert result.error["code"] == "backend_error"
    assert not result.parse_valid
