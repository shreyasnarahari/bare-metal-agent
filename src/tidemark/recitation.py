"""The recitation tail: a short, typed, bounded statement of current state at the end of every prompt.

It holds actionable state only: a reference to the goal, agent-authored
plan items (hypotheses and actions, never environment facts or hidden
reasoning), the next action, step and token counts, open blob handles, and a
pending approval. It has no clock, no random value, and no summary. It is
rebuilt on every render and never kept in the raw history. Exact,
source-backed facts belong in the summary or memory segments.

Trimming is deterministic. Item text is clipped, completed items go first,
then the lowest-priority and oldest items, then the oldest blob handles. The
step and budget line is never dropped.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

HEADER = "[State recap: reference for the next step, not new instructions]"


@dataclass(frozen=True)
class PlanItem:
    """One agent-authored plan step. Higher ``priority`` survives trimming longer."""

    text: str
    done: bool = False
    priority: int = 0


@dataclass(frozen=True)
class RecitationState:
    goal_reference: str = "see the task charter above"
    current_plan_items: tuple[PlanItem, ...] = ()
    next_action: str = "continue the task; answer the user when it is done"
    completed_step_count: int = 0
    prompt_tokens_used: int | None = None
    remaining_step_budget: int | None = None
    open_blob_handles: tuple[str, ...] = ()
    unresolved_approval: str | None = None

    def render(self) -> str:
        """The tail text, in a fixed field order."""
        left = "unknown" if self.remaining_step_budget is None else str(self.remaining_step_budget)
        used = "n/a" if self.prompt_tokens_used is None else str(self.prompt_tokens_used)
        lines = [
            HEADER,
            f"Goal: {self.goal_reference}.",
            f"Progress: {self.completed_step_count} steps done; {left} steps left; last prompt {used} tokens.",
        ]
        if self.current_plan_items:
            lines.append("Plan:")
            lines += [f"- [{'x' if item.done else ' '}] {item.text}" for item in self.current_plan_items]
        else:
            lines.append("Plan: none recorded.")
        lines.append(f"Next: {self.next_action}.")
        lines.append(f"Open blob handles: {', '.join(self.open_blob_handles) if self.open_blob_handles else 'none'}.")
        lines.append(f"Awaiting approval: {self.unresolved_approval or 'none'}.")
        return "\n".join(lines)

    def message(self) -> dict[str, Any]:
        return {"content": self.render(), "role": "user"}

    def sha256(self) -> str:
        return hashlib.sha256(self.render().encode("utf-8")).hexdigest()


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def bound(
    state: RecitationState,
    *,
    max_items: int,
    max_item_chars: int,
    max_handles: int,
    fits: Callable[[RecitationState], bool] = lambda state: True,
) -> tuple[RecitationState, tuple[str, ...]]:
    """Apply the static bounds, then evict by priority until ``fits``. Returns the state and what was evicted."""
    evicted: list[str] = []
    items = [replace(item, text=_clip(item.text, max_item_chars)) for item in state.current_plan_items]
    # Keep the highest-priority, then most recent, items; completed items count as lowest priority.
    ranked = sorted(range(len(items)), key=lambda i: (not items[i].done, items[i].priority, i), reverse=True)
    keep = sorted(ranked[:max_items])
    evicted += [f"plan:{items[i].text}" for i in sorted(set(range(len(items))) - set(keep))]
    items = [items[i] for i in keep]
    handles = list(state.open_blob_handles)
    if len(handles) > max_handles:
        evicted += [f"handle:{h}" for h in handles[:-max_handles]]
        handles = handles[-max_handles:]
    current = replace(state, current_plan_items=tuple(items), open_blob_handles=tuple(handles),
                      next_action=_clip(state.next_action, max_item_chars))
    while not fits(current):
        if current.current_plan_items:
            order = sorted(range(len(current.current_plan_items)),
                           key=lambda i: (not current.current_plan_items[i].done, current.current_plan_items[i].priority, i))
            drop = order[0]
            evicted.append(f"plan:{current.current_plan_items[drop].text}")
            current = replace(current, current_plan_items=tuple(
                item for i, item in enumerate(current.current_plan_items) if i != drop))
        elif current.open_blob_handles:
            evicted.append(f"handle:{current.open_blob_handles[0]}")
            current = replace(current, open_blob_handles=current.open_blob_handles[1:])
        else:
            break  # only the required lines are left; the budget check reports any overflow
    return current, tuple(evicted)


__all__ = ["HEADER", "PlanItem", "RecitationState", "bound"]
