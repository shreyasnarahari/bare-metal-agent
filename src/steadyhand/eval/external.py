"""Offline parts of the external benchmarks: BFCL multi-turn and a ToolMaze subset.

These adapters parse ids, check schema eligibility, label subsets, and define
metrics. Downloads and live runs stay in the notebook, which skips them when
the data is absent.

BFCL multi-turn is a new runner with its own label. It never touches the
single-turn ``BFCL_CATEGORIES`` from ``toolsmith.evaluation.bfcl_adapter``,
and a partial category run is labeled with the exact id list and its hash,
never as a leaderboard score. A schema the grammar compiler rejects makes its
entry ineligible: counted on its own, neither a success nor a failure.

The row shapes below are the fields these adapters read. The notebook's
loader maps each project's files into them; the tests use three-row fixtures.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from toolsmith.evaluation.bfcl_adapter import BFCL_CATEGORIES, constraint_support
from toolsmith.evaluation.metrics import wilson_interval
from toolsmith.evaluation.models import canonical_hash
from toolsmith.protocols import ToolProtocol

MULTI_TURN_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)
MULTI_TURN_CATEGORY_SIZE = 200
TOOLMAZE_FULL = {"base_tasks": 400, "instances": 2000, "tools": 270}
TOOLMAZE_RECOVERIES = ("retry", "switch", "stop")

if set(MULTI_TURN_CATEGORIES) & set(BFCL_CATEGORIES):  # pragma: no cover - guards a future edit
    raise RuntimeError("Multi-turn categories must stay separate from the single-turn BFCL categories")

# BFCL ids are "<category>_<index>", e.g. multi_turn_base_12.
_ID = re.compile(r"(?P<category>[a-z_]+?)_(?P<index>\d+)$")


def multi_turn_category(entry_id: str) -> str:
    """The multi-turn category of a BFCL id; any other id is refused."""
    match = _ID.fullmatch(entry_id)
    if match is None or match.group("category") not in MULTI_TURN_CATEGORIES:
        raise ValueError(f"Not a BFCL multi-turn id: {entry_id!r}")
    return match.group("category")


@dataclass(frozen=True)
class MultiTurnEntry:
    """One BFCL multi-turn conversation: its user turns, tool classes, and withheld functions."""

    entry_id: str
    category: str
    turns: tuple[tuple[dict[str, Any], ...], ...]  # user turns, each a list of messages
    involved_classes: tuple[str, ...]
    missed_functions: dict[str, Any]  # turn index -> functions withheld until then (miss_func)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> MultiTurnEntry:
        """Build from a BFCL dataset row."""
        return cls(
            entry_id=row["id"],
            category=multi_turn_category(row["id"]),
            turns=tuple(tuple(turn) for turn in row["question"]),
            involved_classes=tuple(row.get("involved_classes", ())),
            missed_functions=dict(row.get("missed_function", {})),
        )


@dataclass(frozen=True)
class Eligibility:
    """Whether an entry's tools compile to a grammar, and if not, where (JSON Pointer) and why."""

    entry_id: str
    eligible: bool
    pointer: str | None = None
    message: str | None = None
    grammar_sha256: str | None = None


def entry_tools(
    entry: MultiTurnEntry,
    docs_by_class: Mapping[str, Sequence[Mapping[str, Any]]],
    convert: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """The OpenAI tool list for an entry. ``convert`` defaults to Toolsmith's BFCL converter."""
    functions = [dict(doc) for name in entry.involved_classes for doc in docs_by_class.get(name, ())]
    if convert is None:
        from toolsmith.evaluation.bfcl_adapter import convert_bfcl_tools as convert
    return convert(functions)


def check_eligibility(
    entries: Sequence[MultiTurnEntry],
    docs_by_class: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    protocol: ToolProtocol = ToolProtocol.QWEN35,
    convert: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> list[Eligibility]:
    """Check every entry; ineligible ones are reported, never run without the grammar."""
    results = []
    for entry in entries:
        support = constraint_support(entry_tools(entry, docs_by_class, convert), protocol)
        error = support.error or {}
        results.append(
            Eligibility(
                entry_id=entry.entry_id,
                eligible=support.eligible,
                pointer=error.get("pointer"),
                message=error.get("message"),
                grammar_sha256=support.grammar_sha256,
            )
        )
    return results


def subset_label(name: str, ids: Sequence[str], full_size: int) -> dict[str, Any]:
    """Name exactly what was scored. Anything short of the full id set is ``partial``."""
    ordered = sorted(set(ids))
    return {
        "name": name,
        "ids": ordered,
        "count": len(ordered),
        "sha256": canonical_hash(ordered),
        "label": "full" if len(ordered) >= full_size else "partial",
    }


@dataclass(frozen=True)
class MultiTurnOutcome:
    """How one multi-turn entry went, with or without Steadyhand's guards."""

    entry_id: str
    loop: str  # "guarded" (validation and guards on) or "bare" (execute calls only)
    eligible: bool
    success: bool | None  # None for ineligible entries and infrastructure aborts


def score_multi_turn(outcomes: Sequence[MultiTurnOutcome]) -> dict[str, Any]:
    """Ineligible entries are counted separately and appear in neither successes nor failures."""
    eligible = [outcome for outcome in outcomes if outcome.eligible]
    scored = [outcome for outcome in eligible if outcome.success is not None]
    successes = sum(1 for outcome in scored if outcome.success)
    return {
        "entries": len(outcomes),
        "eligible": len(eligible),
        "ineligible": len(outcomes) - len(eligible),
        "scored": len(scored),
        "successes": successes,
        "failures": len(scored) - successes,
        "success_rate": successes / len(scored) if scored else None,
        "success_interval": wilson_interval(successes, len(scored)) if scored else None,
    }


@dataclass(frozen=True)
class ToolMazeInstance:
    """One ToolMaze task: a topology, a failure mode, and the recovery a good agent shows."""

    instance_id: str
    base_task_id: str
    topology: str
    failure_mode: str  # e.g. explicit_transient, implicit_permanent
    gold_recovery: str  # retry, switch, or stop

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ToolMazeInstance:
        """Build from a ToolMaze row; an unknown gold recovery is refused."""
        recovery = row["gold_recovery"]
        if recovery not in TOOLMAZE_RECOVERIES:
            raise ValueError(f"Unknown ToolMaze recovery {recovery!r}")
        return cls(row["instance_id"], row["base_task_id"], row["topology"], row["failure_mode"], recovery)


def toolmaze_subset(instances: Sequence[ToolMazeInstance], topologies: Sequence[str]) -> list[ToolMazeInstance]:
    """The simpler-topology subset Steadyhand runs. Record its ids with ``subset_label``."""
    wanted = set(topologies)
    return [instance for instance in instances if instance.topology in wanted]


@dataclass(frozen=True)
class ToolMazeOutcome:
    """How one ToolMaze instance went, and whether the agent's recovery matched the gold."""

    instance_id: str
    success: bool
    perturbed: bool
    recovery_action: str | None  # what the agent did after the fault: retry, switch, stop, or None
    gold_recovery: str
    calls_after_fault: int
    effective_calls_after_fault: int  # of those, calls that changed the outcome


def score_toolmaze(outcomes: Sequence[ToolMazeOutcome]) -> dict[str, Any]:
    """Task success, perturbation recovery (action matches gold), and recovery cost (wasted calls)."""
    perturbed = [outcome for outcome in outcomes if outcome.perturbed]
    recovered = sum(1 for outcome in perturbed if outcome.recovery_action == outcome.gold_recovery)
    wasted = [outcome.calls_after_fault - outcome.effective_calls_after_fault for outcome in perturbed]
    successes = sum(1 for outcome in outcomes if outcome.success)
    return {
        "instances": len(outcomes),
        "task_success_rate": successes / len(outcomes) if outcomes else None,
        "perturbed": len(perturbed),
        "perturbation_recovery_rate": recovered / len(perturbed) if perturbed else None,
        "recovery_cost": sum(wasted) / len(wasted) if wasted else None,
    }


__all__ = [
    "Eligibility",
    "MULTI_TURN_CATEGORIES",
    "MULTI_TURN_CATEGORY_SIZE",
    "MultiTurnEntry",
    "MultiTurnOutcome",
    "TOOLMAZE_FULL",
    "ToolMazeInstance",
    "ToolMazeOutcome",
    "check_eligibility",
    "entry_tools",
    "multi_turn_category",
    "score_multi_turn",
    "score_toolmaze",
    "subset_label",
    "toolmaze_subset",
]
