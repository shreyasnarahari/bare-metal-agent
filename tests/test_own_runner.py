from toolsmith.api import APIError
from toolsmith.evaluation.models import AttemptRecord, CaseResult, Mode, RunConfig
from toolsmith.evaluation.own_runner import OwnRunner, run_repair_with_reuse

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


def test_repair_reuse_skips_success_and_backend_but_repairs_parse_failure(tmp_path):
    grammar = [
        CaseResult("a", "simple", "m", "grammar", "success", parse_valid=True, schema_valid=True),
        CaseResult("b", "simple", "m", "grammar", "error", error={"code": "backend_error"}),
        CaseResult("c", "simple", "m", "grammar", "error", error={"code": "invalid_tool_call"}),
    ]
    cases = [case("a"), case("b"), case("c")]
    runner = OwnRunner(FakeOpenAI(), RunConfig("run", "own", "m", Mode.REPAIR), [WEATHER])

    results = run_repair_with_reuse(grammar, cases, runner, tmp_path / "repair.jsonl", "hash")

    assert [result.mode for result in results] == ["repair", "repair", "repair"]
    assert results[0].attempts == ()
    assert results[1].status == "error"
    assert results[2].status == "success"
    resumed = run_repair_with_reuse(grammar, cases, runner, tmp_path / "repair.jsonl", "hash")
    assert len(resumed) == 3


def test_backend_error_is_explicit_result():
    class Broken:
        def create(self, **kwargs):
            raise APIError("backend", code="backend_error")

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Broken()})()})()
    result = OwnRunner(client, RunConfig("run", "own", "m", Mode.PROMPT), []).execute(case())

    assert result.status == "error"
    assert result.error["code"] == "backend_error"
    assert not result.parse_valid
