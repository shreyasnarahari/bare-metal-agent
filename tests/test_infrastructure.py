import json
import threading

import httpx
import openai
import pytest

from toolsmith.api import APIError
from toolsmith.evaluation.models import CaseResult
from toolsmith.evaluation.policies import is_infrastructure_error
from toolsmith.evaluation.runner import (
    INFRASTRUCTURE_RETRIES,
    CheckpointRunner,
    InfrastructureError,
    InfrastructureFailure,
)

REQUEST = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")


def status_error(status, code):
    body = {"error": {"code": code, "message": "x"}}
    return openai.APIStatusError("failed", response=httpx.Response(status, json=body, request=REQUEST), body=body)


@pytest.mark.parametrize("error", [
    openai.APIConnectionError(request=REQUEST),
    openai.APITimeoutError(request=REQUEST),
    httpx.ConnectError("refused", request=REQUEST),
    httpx.ReadTimeout("slow", request=REQUEST),
    ConnectionRefusedError(),
    status_error(502, "backend_error"),
    status_error(503, "anything"),
    APIError("llama-server down", code="backend_error"),
])
def test_serving_failures_are_infrastructure(error):
    assert is_infrastructure_error(error)


@pytest.mark.parametrize("error", [
    status_error(500, "completion_truncated"),  # the model ran out of tokens: a result
    status_error(500, "invalid_tool_call_json"),
    status_error(400, "invalid_request"),
    APIError("truncated", code="completion_truncated"),
    KeyError("choices"),
    ValueError("bad json"),
])
def test_model_and_request_failures_are_not(error):
    assert not is_infrastructure_error(error)


def ok(case_id):
    return CaseResult(case_id, "simple", "m", "prompt", "success")


def infra(case_id):
    fallback = CaseResult(case_id, "simple", "m", "prompt", "error", error={"code": "infrastructure_retries_exhausted"})
    return InfrastructureError(case_id, "APIConnectionError: refused", fallback=fallback)


CASES = [{"case_id": f"c{index}"} for index in range(8)]


def test_an_infrastructure_error_stops_the_run_and_records_nothing_for_it(tmp_path):
    path = tmp_path / "own.jsonl"
    gate = threading.Barrier(2, timeout=5)

    def execute(case):
        if case["case_id"] in ("c0", "c1"):
            gate.wait()  # c0 and c1 are in flight together
        if case["case_id"] == "c1":
            raise infra("c1")
        return ok(case["case_id"])

    with pytest.raises(InfrastructureFailure, match="Nothing was recorded") as caught:
        CheckpointRunner(path, "sha").run(CASES, execute, workers=2)

    recorded = [json.loads(line)["result"]["case_id"] for line in path.read_text().splitlines()]
    assert "c0" in recorded and "c1" not in recorded  # the case in flight finished; the failed one did not record
    assert len(recorded) < len(CASES)  # no new cases started after the failure
    assert caught.value.failures[0][0] == "c1"
    assert json.loads((tmp_path / "own.jsonl.infra.jsonl").read_text())["case_id"] == "c1"

    calls = []
    resumed = CheckpointRunner(path, "sha").run(CASES, lambda case: calls.append(case["case_id"]) or ok(case["case_id"]), workers=2)
    assert "c1" in calls and "c0" not in calls
    assert [result.status for result in resumed] == ["success"] * len(CASES)


def test_a_case_that_keeps_failing_on_infrastructure_is_recorded_after_the_retries(tmp_path):
    path = tmp_path / "bfcl.jsonl"
    cases = [{"case_id": "always-times-out"}, {"case_id": "fine"}]

    def execute(case):
        if case["case_id"] == "always-times-out":
            raise infra(case["case_id"])
        return ok(case["case_id"])

    for _ in range(INFRASTRUCTURE_RETRIES - 1):
        with pytest.raises(InfrastructureFailure):
            CheckpointRunner(path, "sha").run(cases, execute, workers=1)
    results = CheckpointRunner(path, "sha").run(cases, execute, workers=1)

    assert [result.status for result in results] == ["error", "success"]
    assert results[0].error["code"] == "infrastructure_retries_exhausted"
    assert len((tmp_path / "bfcl.jsonl.infra.jsonl").read_text().splitlines()) == INFRASTRUCTURE_RETRIES


def test_completed_results_reads_a_finished_run_only(tmp_path):
    from toolsmith.evaluation.runner import completed_results

    path = tmp_path / "own.jsonl"
    CheckpointRunner(path, "sha").run([{"case_id": "a"}, {"case_id": "b"}], lambda case: ok(case["case_id"]), workers=1)

    assert [r.case_id for r in completed_results(path, "sha", ["b", "a"])] == ["b", "a"]
    assert completed_results(path, "sha", ["a", "b", "c"]) is None  # a partial run is not a result
