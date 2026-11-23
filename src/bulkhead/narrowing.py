"""The narrowing relation: whether one privilege set is no wider than another.

``task_widenings(candidate, current)`` lists every way ``candidate`` grants
more than ``current``. It is empty exactly when the candidate is narrower or
equal, so ``is_narrower`` is a total relation over task policies: reflexive,
and defined for every pair. The comparisons are exact; there is no fuzzy path
or hostname matching.

- Tool and destination sets: the candidate must be a subset.
- Read/write paths: every candidate path must lie beneath a current path of
  the same kind, and no current denied path may lose its coverage.
- Numeric bounds: a lower bound may rise and an upper bound may fall; moving
  either the other way, or dropping a bound, widens.
- Resource limits and egress byte ceilings may only decrease.
- Deny rules, ask rules, argument checks, and taint-sensitive sinks can never
  be removed, and the required sandbox tier can never drop.

Everything here is pure: no I/O, clocks, or randomness.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .manifest import Bounds, Egress, Limits, PathAccess, beneath_any

if TYPE_CHECKING:
    from .policy import TaskPolicy


@dataclass(frozen=True)
class Widening:
    """One way a candidate grants more than the current policy: a stable kind and an exact detail."""

    kind: str
    detail: Any

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail}


def added(candidate: Iterable[str], current: Iterable[str]) -> list[str]:
    """Members of ``candidate`` that ``current`` lacks (each one widens)."""
    return sorted(set(candidate) - set(current))


def removed(candidate: Iterable[str], current: Iterable[str]) -> list[str]:
    """Members of ``current`` that ``candidate`` drops (each dropped restriction widens)."""
    return sorted(set(current) - set(candidate))


def paths_outside(candidate: Iterable[str], current: Iterable[str]) -> list[str]:
    """Candidate paths not beneath any current path."""
    current = tuple(current)
    return sorted(path for path in candidate if not beneath_any(path, current))


def denies_dropped(candidate: Iterable[str], current: Iterable[str]) -> list[str]:
    """Current denied paths that no candidate deny covers any more."""
    candidate = tuple(candidate)
    return sorted(path for path in current if not beneath_any(path, candidate))


def bounds_widening(candidate: Bounds | None, current: Bounds | None) -> dict[str, Any] | None:
    """How ``candidate`` loosens ``current``, or None. Adding a bound where none existed narrows."""
    if current is None:
        return None
    if candidate is None:
        return {"removed": current.as_dict()}
    loosened: dict[str, Any] = {}
    if current.minimum is not None and (candidate.minimum is None or candidate.minimum < current.minimum):
        loosened["minimum"] = candidate.minimum
    if current.maximum is not None and (candidate.maximum is None or candidate.maximum > current.maximum):
        loosened["maximum"] = candidate.maximum
    return loosened or None


def limits_widening(candidate: Limits, current: Limits) -> dict[str, int | float]:
    """Every ceiling the candidate raises."""
    ceilings, base = candidate.ceilings(), current.ceilings()
    return {name: ceilings[name] for name in ceilings if ceilings[name] > base[name]}


def egress_widening(candidate: Egress, current: Egress) -> dict[str, Any]:
    """New destinations and raised byte ceilings."""
    widened: dict[str, Any] = {}
    if new := added(candidate.allow, current.allow):
        widened["allow"] = new
    for name in ("max_bytes_out", "max_bytes_in"):
        if getattr(candidate, name) > getattr(current, name):
            widened[name] = getattr(candidate, name)
    return widened


def _path_arguments_widening(candidate: Mapping[str, PathAccess], current: Mapping[str, PathAccess]) -> dict[str, Any]:
    # Dropping a checked path argument widens; so does relaxing a write check to a read check.
    loosened: dict[str, Any] = {}
    for argument, access in current.items():
        if argument not in candidate:
            loosened[argument] = None
        elif access is PathAccess.WRITE and candidate[argument] is PathAccess.READ:
            loosened[argument] = PathAccess.READ.value
    return loosened


def task_widenings(candidate: TaskPolicy, current: TaskPolicy) -> tuple[Widening, ...]:
    """Every widening from ``current`` to ``candidate``, in a fixed order. Empty means narrower or equal."""
    found: list[Widening] = []

    def note(kind: str, detail: Any) -> None:
        if detail:
            found.append(Widening(kind, detail))

    note("tools", added(candidate.tools, current.tools))
    note("deny_tools_removed", removed(candidate.deny_tools, current.deny_tools))
    note("ask_tools_removed", removed(candidate.ask_tools, current.ask_tools))

    candidate_rules = candidate.rules_by_tool()
    for tool, rule in sorted(current.rules_by_tool().items()):
        other = candidate_rules.get(tool)
        bounds = {
            argument: widening
            for argument, current_bounds in sorted(rule.bounds.items())
            if (widening := bounds_widening(other.bounds.get(argument) if other else None, current_bounds)) is not None
        }
        note("bounds", {tool: bounds} if bounds else None)
        paths = _path_arguments_widening(other.paths if other else {}, rule.paths)
        note("path_arguments", {tool: paths} if paths else None)
        destinations = removed(other.destinations if other else (), rule.destinations)
        note("destination_arguments", {tool: destinations} if destinations else None)

    note("fs_read", paths_outside(candidate.fs_read, (*current.fs_read, *current.fs_write)))
    note("fs_write", paths_outside(candidate.fs_write, current.fs_write))
    note("fs_deny_removed", denies_dropped(candidate.fs_deny, current.fs_deny))
    note("egress", egress_widening(candidate.egress, current.egress))
    note("egress_deny_removed", removed(candidate.egress_deny, current.egress_deny))
    note("limits", limits_widening(candidate.limits, current.limits))
    note("taint_sensitive_removed", removed([s.value for s in candidate.taint_sensitive], [s.value for s in current.taint_sensitive]))
    if candidate.required_tier.rank < current.required_tier.rank:
        note("required_tier", candidate.required_tier.value)
    return tuple(found)


def is_narrower(candidate: TaskPolicy, current: TaskPolicy) -> bool:
    """True when ``candidate`` grants nothing ``current`` does not (equal policies count)."""
    return not task_widenings(candidate, current)


__all__ = [
    "Widening",
    "added",
    "bounds_widening",
    "denies_dropped",
    "egress_widening",
    "is_narrower",
    "limits_widening",
    "paths_outside",
    "removed",
    "task_widenings",
]
