"""Run Toolsmith's own 200-case evaluation set through the shim, checkpointed and resumable.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from .bfcl_adapter import decoded_calls
from .metrics import calls_equal
from .models import AttemptRecord, CaseResult, Mode, RunConfig
from .policies import (
    calls_match_schema,
    exception_error_code,
    is_infrastructure_error,
    repair_eligible,
    reuse_grammar_result,
)
from .runner import EVAL_WORKERS, CheckpointRunner, InfrastructureError

def infrastructure_error(case_id: str, exc: BaseException, result: CaseResult) -> InfrastructureError:
    """The serving stack failed: raise instead of recording ``result``, which becomes the
    fallback once the case has failed this way ``INFRASTRUCTURE_RETRIES`` times."""
    message = f"{type(exc).__name__}: {exc}"
    fallback = dataclasses.replace(
        result, error={"code": "infrastructure_retries_exhausted", "message": message},
    )
    return InfrastructureError(case_id, message, fallback=fallback)


class OwnRunner:
    """One model and mode over the own set: ``execute`` runs and scores one case against its gold calls."""

    def __init__(self, client: Any, config: RunConfig, tools: list[dict[str, Any]]) -> None:
        self.client = client
        self.config = config
        self.tools = tools

    def execute(self, case: dict[str, Any]) -> CaseResult:
        """Send one case and score it. A model failure is a result; a serving failure raises."""
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=case["messages"],
                tools=self.tools,
                tool_choice=case["tool_choice"],
                temperature=self.config.temperature,
                seed=self.config.seed,
                max_tokens=self.config.max_tokens,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": self.config.mode is Mode.THINKING
                    }
                },
            ).model_dump()
            message = response["choices"][0]["message"]
            calls = decoded_calls(response)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            code = exception_error_code(exc)
            # A malformed response body is the model's problem (bad JSON), not the backend's.
            if code == "backend_error" and isinstance(
                exc, (json.JSONDecodeError, KeyError, TypeError, ValueError)
            ):
                code = "invalid_tool_call_json"
            result = CaseResult(
                case["case_id"],
                case["category"],
                self.config.model,
                self.config.mode.value,
                "error",
                attempts=(AttemptRecord(0, None, elapsed, error={"code": code, "message": str(exc)}),),
                error={"code": code, "message": str(exc)},
            )
            if is_infrastructure_error(exc):
                raise infrastructure_error(case["case_id"], exc, result) from exc
            return result
        elapsed = time.perf_counter() - started
        # Semantic: the calls equal the gold calls. A no-tool case also needs a non-empty text answer.
        semantic = calls_equal(calls, case["expected_calls"], case["comparison"])
        if case["comparison"] == "none":
            semantic = semantic and bool((message.get("content") or "").strip())
        details = response.get("toolsmith", {})
        repairs = details.get("repair_attempts", 0)
        return CaseResult(
            case["case_id"],
            case["category"],
            self.config.model,
            self.config.mode.value,
            "repaired" if repairs else "success",
            attempts=(
                AttemptRecord(
                    0,
                    response,
                    elapsed,
                    response.get("usage", {}).get("prompt_tokens", 0),
                    response.get("usage", {}).get("completion_tokens", 0),
                ),
            ),
            parsed_calls=calls,
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
            parse_valid=True,
            schema_valid=calls_match_schema(
                self.tools,
                calls,
                allow_empty=case["comparison"] == "none",
            ),
            semantic_valid=semantic,
            repair_attempts=repairs,
            grammar_sha256=details.get("grammar_sha256"),
        )

    def run(
        self,
        cases: list[dict[str, Any]],
        checkpoint: str | Path,
        manifest_sha256: str,
        *,
        manifest: dict[str, Any] | None = None,
        workers: int = EVAL_WORKERS,
    ) -> list[CaseResult]:
        """``workers`` should equal the llama-server slots so every slot stays busy."""
        return CheckpointRunner(checkpoint, manifest_sha256, manifest=manifest).run(cases, self.execute, workers=workers)


# Same reuse rule as BFCL: grammar-mode results that were already right are copied (marked
# reused), and only repair-eligible cases run again with repair enabled.
def run_repair_with_reuse(
    grammar_results: list[CaseResult],
    cases: list[dict[str, Any]],
    runner: OwnRunner,
    checkpoint: str | Path,
    manifest_sha256: str,
    *,
    manifest: dict[str, Any] | None = None,
    workers: int = EVAL_WORKERS,
) -> list[CaseResult]:
    case_by_id = {case["case_id"]: case for case in cases}
    checkpoint_runner = CheckpointRunner(checkpoint, manifest_sha256, manifest=manifest)
    remaining = []
    for result in grammar_results:
        if not reuse_grammar_result(result) and repair_eligible(result):
            remaining.append(case_by_id[result.case_id])
            continue
        reused = dataclasses.replace(
            result,
            mode=Mode.REPAIR.value,
            attempts=tuple(
                dataclasses.replace(attempt, reused=True) for attempt in result.attempts
            ),
            metadata={**result.metadata, "repair_reused": True},
        )
        if result.case_id not in checkpoint_runner.completed:
            checkpoint_runner.append(reused)
    checkpoint_runner.run(remaining, runner.execute, workers=workers)
    return [checkpoint_runner.completed[case["case_id"]] for case in cases]


__all__ = ["OwnRunner", "run_repair_with_reuse"]
