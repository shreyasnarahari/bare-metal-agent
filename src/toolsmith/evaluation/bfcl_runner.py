from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

from ..protocols import ToolProtocol
from .bfcl_adapter import constraint_support, decoded_calls, result_entry
from .models import AttemptRecord, CaseResult, Mode, RunConfig
from .policies import calls_match_schema, exception_error_code, repair_eligible, reuse_grammar_result
from .runner import CheckpointRunner


class BFCLRunner:
    def __init__(
        self,
        client: Any,
        config: RunConfig,
        *,
        protocol: ToolProtocol = ToolProtocol.QWEN3,
        tools_converter: Any,
    ) -> None:
        self.client = client
        self.config = config
        self.protocol = protocol
        self.tools_converter = tools_converter

    def execute(self, entry: dict[str, Any]) -> CaseResult:
        case_id = entry["id"]
        category = case_id.rsplit("_", 1)[0]
        tools = self.tools_converter(entry["function"])
        constrained = self.config.mode in {Mode.GRAMMAR, Mode.REPAIR}
        support = constraint_support(tools, self.protocol) if constrained else None
        if support is not None and not support.eligible:
            return CaseResult(
                case_id,
                category,
                self.config.model,
                self.config.mode.value,
                "constraint_unsupported",
                error=support.error,
            )
        thinking = self.config.mode is Mode.THINKING
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=entry["question"][0],
                tools=tools,
                tool_choice="auto",
                temperature=self.config.temperature,
                seed=self.config.seed,
                max_tokens=self.config.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
            ).model_dump()
            elapsed = time.perf_counter() - started
            serialized = result_entry(case_id, response, latency=elapsed)
            calls = decoded_calls(response)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            code = exception_error_code(exc)
            return CaseResult(
                case_id,
                category,
                self.config.model,
                self.config.mode.value,
                "error",
                attempts=(
                    AttemptRecord(0, None, elapsed, error={"code": code, "message": str(exc)}),
                ),
                error={"code": code, "message": str(exc)},
                grammar_sha256=support.grammar_sha256 if support else None,
            )
        details = response.get("toolsmith", {})
        repairs = details.get("repair_attempts", 0)
        return CaseResult(
            case_id,
            category,
            self.config.model,
            self.config.mode.value,
            "repaired" if repairs else "success",
            attempts=(AttemptRecord(0, response, elapsed),),
            parsed_calls=calls,
            content=response["choices"][0]["message"].get("content"),
            reasoning_content=response["choices"][0]["message"].get("reasoning_content"),
            parse_valid=True,
            schema_valid=calls_match_schema(tools, calls, allow_empty=True),
            repair_attempts=repairs,
            grammar_sha256=support.grammar_sha256 if support else None,
            metadata={
                "bfcl_result": serialized,
                "validation": "toolsmith" if constrained else "bfcl_ast",
            },
        )

    def run(self, entries: list[dict[str, Any]], checkpoint: str | Path, manifest_sha256: str) -> list[CaseResult]:
        return CheckpointRunner(checkpoint, manifest_sha256).run(entries, self.execute)


def evaluate_bfcl(
    *,
    registry_name: str,
    model_name: str,
    categories: list[str],
    result_dir: str,
    score_dir: str,
    partial: bool,
) -> None:
    from .bfcl_handler import register_handler
    from bfcl_eval.eval_checker.eval_runner import main

    register_handler(registry_name, model_name=model_name)
    main([registry_name], categories, result_dir, score_dir, partial_eval=partial)


def run_bfcl_repair_with_reuse(
    grammar_results,
    entries,
    runner,
    checkpoint,
    manifest_sha256,
):
    entry_by_id = {entry["id"]: entry for entry in entries}
    checkpoint_runner = CheckpointRunner(checkpoint, manifest_sha256)
    remaining = []
    for result in grammar_results:
        if not reuse_grammar_result(result) and repair_eligible(result):
            remaining.append(entry_by_id[result.case_id])
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
    checkpoint_runner.run(remaining, runner.execute)
    return [checkpoint_runner.completed[entry["id"]] for entry in entries]


__all__ = ["BFCLRunner", "evaluate_bfcl", "run_bfcl_repair_with_reuse"]
