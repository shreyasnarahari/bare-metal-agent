from toolsmith.api import APIError
from toolsmith.evaluation.jsonschema_engine import JSONSchemaEngineCore
from toolsmith.evaluation.models import Mode
from toolsmith.evaluation.schema_coverage import (
    compile_coverage,
    isolated_compile_coverage,
    pending_file_is_stale,
    replay_schema,
)


def test_compile_and_replay_distinguish_valid_invalid_and_unsupported():
    compiled = compile_coverage({"type": "integer"}, "integer")
    unsupported = compile_coverage({"type": "string", "pattern": "^x$"}, "pattern")
    replay = replay_schema(
        {"type": "integer"},
        [
            {"data": 1, "valid": True},
            {"data": "1", "valid": False},
            {"data": 1.5, "valid": False},
        ],
        "integer",
    )

    assert compiled.status == "compiled"
    assert compiled.grammar_sha256
    assert unsupported.status == "explicit_unsupported"
    assert unsupported.pointer == "/pattern"
    assert replay["valid_passed"] == 1
    assert replay["invalid_passed"] == 2
    assert replay["validation_error"] is None
    assert replay["invalidation_error"] is None


def test_replay_counts_expected_invalid_instances():
    object_replay = replay_schema(
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        [
            {"data": {"x": "wrong"}, "valid": False},
            {"data": {}, "valid": False},
        ],
    )

    assert object_replay["invalidation_error"] is None
    assert object_replay["invalid_passed"] == 2


def test_isolated_compile_and_pending_ownership(tmp_path):
    result = isolated_compile_coverage({"type": "integer"}, "isolated", wall_seconds=10)
    assert result["status"] == "compiled"
    pending = tmp_path / "case.json"
    pending.write_text('{"pending_file":1,"run_id":"mine"}')
    other = tmp_path / "other.json"
    other.write_text('{"pending_file":1,"run_id":"other"}')
    assert pending_file_is_stale(pending, "mine")
    assert not pending_file_is_stale(other, "mine")


class FakeCompletions:
    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("Response", (), {"model_dump": lambda self: {
            "choices": [{"message": {"content": '{"x":1}', "tool_calls": None}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }})()


class FakeOpenAI:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_jsonschema_engine_returns_real_usage_without_fabricating_timing():
    engine = JSONSchemaEngineCore(FakeOpenAI(), "model", mode=Mode.GRAMMAR)
    result = engine.generate([], {"type": "object"})

    assert result.generation == '{"x":1}'
    assert result.input_tokens == 10 and result.output_tokens == 4
    assert result.ttft is None and result.gct is None


class UnsupportedCompletions:
    def create(self, **kwargs):
        raise APIError("unsupported", code="unsupported_schema")


def test_jsonschema_engine_maps_unsupported_schema():
    client = type("Client", (), {"chat": type("Chat", (), {"completions": UnsupportedCompletions()})()})()
    result = JSONSchemaEngineCore(client, "model").generate([], {"pattern": "x"})

    assert result.compile_status == "unsupported_schema"
    assert result.decoding_status == "not_run"
