"""Coarse episode taint: trust labels, sink classes, and the per-episode taint state.

This is deliberately coarse information-flow control, not per-value
capabilities (it does not implement CaMeL-style tracking). An episode starts
``TRUSTED``. The first observation that carries outside content (tool output
from a tool whose results are not declared trusted) makes it ``UNTRUSTED``, and
nothing in this track ever cleans it again. While untrusted, the policy engine
turns an otherwise allowed call into ``ASK`` when that call reaches a
taint-sensitive sink; an explicit ``DENY`` stays a denial, and calls without a
sensitive sink (pure reads) are unaffected.

``EpisodeTaintState`` is an immutable value owned by one episode's security
gate. It records bounded source descriptors (tool call id, tool, journal
sequence number), never the outside content itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

MAX_TAINT_SOURCES = 32
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_~.-]{0,127}")


class Trust(str, Enum):
    """How far a tool's output may be trusted."""

    TRUSTED = "trusted"  # deterministic internal results, declared trusted after their fields were reviewed
    UNTRUSTED = "untrusted"  # anything that can carry outside content; the default for every new tool


class Sink(str, Enum):
    """Where a call's effects go. A tool with no sink (``sinks=()``, the "none" class) only reads.

    ``NETWORK`` covers any outbound communication, including a nominally
    read-only fetch: URLs, query strings, headers, and DNS names can all carry
    data out. ``FILESYSTEM`` is a write inside the tool's disposable workspace;
    ``FILESYSTEM_SENSITIVE`` is a write outside it or to a path the task
    declares sensitive. ``EXTERNAL_EFFECT`` is a payment, ticket, message, or
    other outside-visible change; tasks decide which of those are sensitive.
    """

    NETWORK = "network"
    FILESYSTEM = "filesystem"
    FILESYSTEM_SENSITIVE = "filesystem_sensitive"
    EXTERNAL_EFFECT = "external_effect"


# What a task treats as taint-sensitive unless it says otherwise.
DEFAULT_TAINT_SENSITIVE = (Sink.FILESYSTEM_SENSITIVE, Sink.NETWORK)


@dataclass(frozen=True)
class TaintSource:
    """Where outside content entered: the call, the tool, and the journal record that carried it."""

    tool_call_id: str
    tool: str
    seq: int

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_id, str) or not _ID.fullmatch(self.tool_call_id):
            raise ValueError(f"bad tool_call_id {self.tool_call_id!r}")
        if not isinstance(self.tool, str) or not _ID.fullmatch(self.tool):
            raise ValueError(f"bad tool {self.tool!r}")
        if type(self.seq) is not int or self.seq < 0:
            raise ValueError("seq must be a non-negative integer")

    def as_dict(self) -> dict[str, Any]:
        return {"tool_call_id": self.tool_call_id, "tool": self.tool, "seq": self.seq}


@dataclass(frozen=True)
class TaintTransition:
    """A label change and the source that caused it (journaled as ``taint_transition``)."""

    before: Trust
    after: Trust
    source: TaintSource


@dataclass(frozen=True)
class EpisodeTaintState:
    """One episode's taint: the label and an ordered, bounded list of distinct sources.

    ``observe`` returns the next state and, only when the label changes, the
    transition. Repeated untrusted results add their source (up to
    ``MAX_TAINT_SOURCES``; later ones are counted in ``overflow``) without a
    redundant transition. There is no operation that makes an episode trusted again.
    """

    label: Trust = Trust.TRUSTED
    sources: tuple[TaintSource, ...] = ()
    overflow: int = 0

    @property
    def tainted(self) -> bool:
        return self.label is Trust.UNTRUSTED

    def observe(self, source: TaintSource, trust: Trust) -> tuple[EpisodeTaintState, TaintTransition | None]:
        """Account for one observation carrying content from a tool whose output is ``trust``."""
        if Trust(trust) is Trust.TRUSTED:
            return self, None
        if source in self.sources:
            state = self
        elif len(self.sources) < MAX_TAINT_SOURCES:
            state = EpisodeTaintState(Trust.UNTRUSTED, (*self.sources, source), self.overflow)
        else:
            state = EpisodeTaintState(Trust.UNTRUSTED, self.sources, self.overflow + 1)
        if self.label is Trust.UNTRUSTED:
            return state, None
        return state, TaintTransition(Trust.TRUSTED, Trust.UNTRUSTED, source)

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label.value, "sources": [source.as_dict() for source in self.sources], "overflow": self.overflow}


__all__ = [
    "DEFAULT_TAINT_SENSITIVE",
    "MAX_TAINT_SOURCES",
    "EpisodeTaintState",
    "Sink",
    "TaintSource",
    "TaintTransition",
    "Trust",
]
