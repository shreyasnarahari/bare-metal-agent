"""Regenerate the golden Steadyhand episodes from scripted model output.

Run by hand when an episode behavior change is intentional, review the diff,
and commit it. The test suite never rewrites these files.

    .venv/bin/python tests/fixtures/steadyhand_episodes/regenerate.py

Each scenario is recorded through ``run_claimed_episode`` with a fake
``ToolsmithService``, so the files hold no prompts, keys, or GPU output.
"""

from __future__ import annotations

import json
from pathlib import Path

from steadyhand import SteadyhandClient, build_registry, initial_messages, load_fixture
from steadyhand.replay import run_claimed_episode
from toolsmith.protocols import QWEN3_PROFILE

HERE = Path(__file__).resolve().parent
REFUND = {"order_id": "ord_1001", "amount_inr": 120, "reason": "Dented lid"}
BAD_REFUND = {**REFUND, "amount_inr": -120}


class ScriptedService:
    """Stands in for ToolsmithService: returns canned OpenAI-shaped completions."""

    profile = QWEN3_PROFILE
    use_grammars = True

    def __init__(self, completions):
        self.completions = list(completions)

    def chat(self, request):
        return self.completions.pop(0)


def completion(content=None, calls=(), prompt_tokens=120, completion_tokens=18):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            for call_id, name, arguments in calls
        ]
    return {
        "choices": [{"index": 0, "message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "toolsmith": {"repair_attempts": 0},
    }


SCENARIOS = {
    "happy_path": (
        "What is the current status of order ord_1001?",
        [
            completion(calls=[("call_status", "get_order", {"order_id": "ord_1001"})]),
            completion("Order ord_1001 (an electric kettle) was delivered."),
        ],
    ),
    "validation_retry": (
        "Refund 120 INR on order ord_1001 for the dented lid.",
        [
            completion(calls=[("call_bad", "issue_refund", BAD_REFUND)]),
            completion(calls=[("call_fixed", "issue_refund", REFUND)]),
            completion("I refunded 120 INR on order ord_1001 (refund ref_7002)."),
        ],
    ),
    "repeated_call_escalation": (
        "Refund 120 INR on order ord_1001 for the dented lid.",
        [completion(calls=[(f"call_{index}", "issue_refund", BAD_REFUND)]) for index in range(4)],
    ),
}


def main() -> None:
    for old in [*HERE.glob("ep_*.jsonl"), *HERE.glob("ep_*.manifest.json")]:
        old.unlink()
    for label, (goal, completions) in SCENARIOS.items():
        result = run_claimed_episode(
            HERE,
            SteadyhandClient(ScriptedService(completions)),
            build_registry(),
            load_fixture().clone(),
            initial_messages(goal),
            run_label=label,
        )
        assert result is not None
        print(f"{label}: {result.status.value} -> {result.journal_path.name}")


if __name__ == "__main__":
    main()
