"""Seeded, pinned tool-side faults, injected between the loop guard and the handler.

    validate -> loop guard -> injector -> handler

The schedule is a pure function of ``(seed, pins)``. Explicit pins are the
paper trail ("issue_refund call 0 is a lost acknowledgment"). An optional
seeded overlay adds random explicit faults for exploratory runs; its pins are
generated once and serialized with the schedule, so replay never re-rolls them.

``call_index`` is per tool and per episode, 0-based, and counts every call
that reaches the injector: calls that passed validation and the guard,
whether or not a fault fires. A validation failure never consumes an index,
so a model that repairs its arguments still meets the fault on the same
logical call. Policy retries (part 6) are new injector calls: they consume an
index and are tagged ``via="policy"`` so the guard does not count them. To
make every attempt fail, pin indexes 0, 1, and 2.

Faults are tool-side only. Model-output failures stay in ``model_turn``; the
two taxonomies never share a field. The model never learns a fault was
injected: explicit faults read like real failures, and implicit faults are
schema-valid lies.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from jsonschema import Draft202012Validator

from toolsmith.evaluation.models import canonical_hash

from .errors import ErrorType
from .policy import ToolPolicy
from .store import Store
from .types import ToolCall, ToolResult
from .validate import timeout_observation, transient_failure_observation, unavailable_observation

TRANSIENT_ERROR_TYPES = ("timeout", "rate_limit", "server_error")
PERMANENT_ERROR_TYPE = "permanent_unavailable"


class FaultConfigError(ValueError):
    """A pin or overlay that cannot produce the fault it describes."""


class FaultMode(str, Enum):
    EXPLICIT_TRANSIENT = "explicit_transient"
    EXPLICIT_PERMANENT = "explicit_permanent"
    IMPLICIT_TRANSIENT = "implicit_transient"
    IMPLICIT_PERMANENT = "implicit_permanent"
    LOST_ACKNOWLEDGMENT = "lost_acknowledgment"

    @property
    def permanent(self) -> bool:
        return self in (FaultMode.EXPLICIT_PERMANENT, FaultMode.IMPLICIT_PERMANENT)

    @property
    def implicit(self) -> bool:
        return self in (FaultMode.IMPLICIT_TRANSIENT, FaultMode.IMPLICIT_PERMANENT)


@dataclass(frozen=True)
class FaultSpec:
    tool: str
    call_index: int  # per tool, 0-based; a permanent mode holds from here on
    mode: FaultMode
    error_type: str | None = None  # explicit modes only
    implicit_patch: dict[str, Any] | None = None  # implicit modes: JSON merge patch on the real value
    source: str = "pin"  # "pin" or "overlay"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", FaultMode(self.mode))
        if self.call_index < 0:
            raise FaultConfigError("call_index must not be negative")
        if self.source not in ("pin", "overlay"):
            raise FaultConfigError("source must be 'pin' or 'overlay'")
        if self.mode is FaultMode.EXPLICIT_TRANSIENT:
            error_type = self.error_type or "timeout"
            if error_type not in TRANSIENT_ERROR_TYPES:
                raise FaultConfigError(f"explicit_transient error_type must be one of {list(TRANSIENT_ERROR_TYPES)}")
            object.__setattr__(self, "error_type", error_type)
        elif self.mode is FaultMode.EXPLICIT_PERMANENT:
            if self.error_type not in (None, PERMANENT_ERROR_TYPE):
                raise FaultConfigError(f"explicit_permanent error_type must be {PERMANENT_ERROR_TYPE!r}")
            object.__setattr__(self, "error_type", PERMANENT_ERROR_TYPE)
        elif self.error_type is not None:
            raise FaultConfigError(f"{self.mode.value} takes no error_type")
        if self.implicit_patch is not None and not self.mode.implicit:
            raise FaultConfigError(f"{self.mode.value} takes no implicit_patch")
        if self.implicit_patch is not None and not isinstance(self.implicit_patch, dict):
            raise FaultConfigError("implicit_patch must be a JSON object")

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "call_index": self.call_index,
            "mode": self.mode.value,
            "error_type": self.error_type,
            "implicit_patch": deepcopy(self.implicit_patch),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FaultSpec:
        return cls(
            tool=value["tool"],
            call_index=value["call_index"],
            mode=FaultMode(value["mode"]),
            error_type=value.get("error_type"),
            implicit_patch=deepcopy(value.get("implicit_patch")),
            source=value.get("source", "pin"),
        )


@dataclass(frozen=True)
class RandomOverlay:
    """Seeded extra faults for exploratory runs. Explicit and lost-ack modes only: lies need a patch."""

    tools: tuple[str, ...]
    max_call_index: int
    rate: float
    modes: tuple[FaultMode, ...] = (FaultMode.EXPLICIT_TRANSIENT,)

    def __post_init__(self) -> None:
        if not 0 <= self.rate <= 1:
            raise FaultConfigError("rate must be between 0 and 1")
        if self.max_call_index < 0:
            raise FaultConfigError("max_call_index must not be negative")
        if any(FaultMode(mode).implicit for mode in self.modes):
            raise FaultConfigError("An overlay cannot fabricate values; pin implicit faults explicitly")


@dataclass(frozen=True)
class FaultSchedule:
    seed: int
    pins: tuple[FaultSpec, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[str, int]] = set()
        for pin in self.pins:
            key = (pin.tool, pin.call_index)
            if key in seen:
                raise FaultConfigError(f"Two pins for {pin.tool} call {pin.call_index}")
            seen.add(key)

    @property
    def sha256(self) -> str:
        """Hashes the seed and the ordered pins (overlay pins included)."""
        return canonical_hash(self.as_dict())

    def lookup(self, tool: str, call_index: int) -> FaultSpec | None:
        """The fault for this call: an exact pin first, else the latest permanent pin already in force."""
        permanent = None
        for pin in self.pins:
            if pin.tool != tool:
                continue
            if pin.call_index == call_index:
                return pin
            if pin.mode.permanent and pin.call_index < call_index:
                if permanent is None or pin.call_index > permanent.call_index:
                    permanent = pin
        return permanent

    def as_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "pins": [pin.as_dict() for pin in self.pins]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FaultSchedule:
        return cls(seed=value["seed"], pins=tuple(FaultSpec.from_dict(pin) for pin in value["pins"]))


def build_schedule(seed: int, pins: Sequence[FaultSpec], *, overlay: RandomOverlay | None = None) -> FaultSchedule:
    """A pure function of its arguments. Uses its own ``random.Random(seed)``, never the global one."""
    all_pins = list(pins)
    if overlay is not None:
        taken = {(pin.tool, pin.call_index) for pin in all_pins}
        generator = random.Random(seed)
        for tool in sorted(overlay.tools):
            for call_index in range(overlay.max_call_index + 1):
                draw = generator.random()
                mode = FaultMode(generator.choice(overlay.modes))
                error_type = generator.choice(TRANSIENT_ERROR_TYPES)
                if (tool, call_index) in taken or draw >= overlay.rate:
                    continue
                all_pins.append(
                    FaultSpec(
                        tool=tool,
                        call_index=call_index,
                        mode=mode,
                        error_type=error_type if mode is FaultMode.EXPLICIT_TRANSIENT else None,
                        source="overlay",
                    )
                )
    return FaultSchedule(seed=seed, pins=tuple(all_pins))


@dataclass(frozen=True)
class InjectedFault:
    """What the journal records. ``mode`` is what happened; ``pinned_mode`` is what was asked for."""

    tool: str
    tool_call_id: str
    call_index: int
    mode: str
    pinned_mode: str
    committed: bool
    error_type: str | None
    note: str | None
    via: str
    schedule_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "tool_call_id": self.tool_call_id,
            "call_index": self.call_index,
            "mode": self.mode,
            "pinned_mode": self.pinned_mode,
            "committed": self.committed,
            "error_type": self.error_type,
            "note": self.note,
            "via": self.via,
            "schedule_sha256": self.schedule_sha256,
        }


def merge_patch(target: Any, patch: Any) -> Any:
    """RFC 7386 JSON Merge Patch: objects merge, ``null`` removes a key, anything else replaces."""
    if not isinstance(patch, dict):
        return deepcopy(patch)
    result = deepcopy(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


Invoke = Callable[[Store], ToolResult]


class FaultInjector:
    """Applies a schedule to one episode. Holds that episode's per-tool counters; never shared."""

    def __init__(self, schedule: FaultSchedule) -> None:
        self.schedule = schedule
        self._schedule_sha256 = schedule.sha256
        self._counts: dict[str, int] = {}

    def next_index(self, tool: str) -> int:
        return self._counts.get(tool, 0)

    def apply(
        self,
        call: ToolCall,
        invoke: Invoke,
        *,
        store: Store,
        policy: ToolPolicy,
        result_schema: Mapping[str, Any] | None = None,
        arguments_sha256: str | None = None,
        via: str = "model",
    ) -> ToolResult:
        """Run ``call`` through the schedule. ``invoke(target)`` runs the real handler on ``target``."""
        call_index = self._counts.get(call.name, 0)
        self._counts[call.name] = call_index + 1
        spec = self.schedule.lookup(call.name, call_index)
        if spec is None:
            return invoke(store)

        def fault(mode: FaultMode, *, committed: bool = False, note: str | None = None) -> InjectedFault:
            return InjectedFault(
                tool=call.name,
                tool_call_id=call.id,
                call_index=call_index,
                mode=mode.value,
                pinned_mode=spec.mode.value,
                committed=committed,
                error_type=spec.error_type,
                note=note,
                via=via,
                schedule_sha256=self._schedule_sha256,
            )

        def refused(observation: Any) -> ToolResult:
            # The handler never runs, so the arguments hash is all the result carries over.
            return ToolResult(
                call_id=call.id,
                tool=call.name,
                arguments=call.arguments,
                arguments_sha256=arguments_sha256,
                ok=False,
                error=observation,
                handler_called=False,
                fault=fault(spec.mode),
            )

        if spec.mode is FaultMode.EXPLICIT_TRANSIENT:
            return refused(transient_failure_observation(call.name, spec.error_type or "timeout", policy))
        if spec.mode is FaultMode.EXPLICIT_PERMANENT:
            return refused(unavailable_observation(call.name))

        if spec.mode.implicit:
            # The real handler runs on a throwaway copy: a realistic value, nothing committed.
            real = invoke(store.clone())
            if not real.ok:
                return replace(real, handler_called=False, fault=fault(spec.mode, note="base_call_failed"))
            value = merge_patch(real.value, spec.implicit_patch or {})
            if result_schema is not None:
                errors = sorted(error.message for error in Draft202012Validator(result_schema).iter_errors(value))
                if errors:
                    raise FaultConfigError(
                        f"The {spec.mode.value} patch for {call.name} makes an invalid result: {errors[0]}"
                    )
            return replace(real, value=value, handler_called=False, fault=fault(spec.mode))

        # Lost acknowledgment: the handler commits, then the caller is told it timed out.
        real = invoke(store)
        if real.ok:
            return replace(
                real,
                ok=False,
                value=None,
                error=timeout_observation(call.name, policy),
                fault=fault(spec.mode, committed=True),
            )
        if real.error is not None and real.error.error_type is ErrorType.TIMEOUT:
            return replace(real, fault=fault(FaultMode.EXPLICIT_TRANSIENT, note="handler_timeout"))
        return replace(
            real,
            error=timeout_observation(call.name, policy),
            fault=fault(spec.mode, note="handler_error"),
        )


__all__ = [
    "FaultConfigError",
    "FaultInjector",
    "FaultMode",
    "FaultSchedule",
    "FaultSpec",
    "InjectedFault",
    "PERMANENT_ERROR_TYPE",
    "RandomOverlay",
    "TRANSIENT_ERROR_TYPES",
    "build_schedule",
    "merge_patch",
]
