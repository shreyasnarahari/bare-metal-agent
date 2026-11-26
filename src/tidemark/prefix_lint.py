"""PrefixLint: where a prompt stopped matching the previous one, and whether that was allowed.

For each render the session keeps the previous prompt's full token ids
(locally, never in the journal) and the length of its stable prefix: every
token before the old recitation tail. Comparing the two gives:

- ``first_divergent_index``: the first differing position, or the shorter
  length when one prompt is a prefix of the other (``None`` when identical);
- ``observed_common_prefix_tokens``;
- ``expected_common_prefix_floor``: the previous stable prefix on an ordinary
  step, or the previous frozen and pinned segments on an epoch transition;
- ``first_changed_segment`` and ``divergence_cause``, from the segment hashes
  of the two logical states (``token_boundary_merge`` when no segment changed
  but the tokens did anyway);
- ``lint_status``: ``cold_start``, ``expected_append``, ``expected_tail_only``,
  ``expected_epoch``, or ``violation``.

An ordinary step must match every previous token before the old tail at the
same index. PrefixLint shows prompt identity; it does not show that
llama.cpp reused its KV cache. ``backend_timings`` holds that evidence, so a
stable prompt on a cold or evicted slot is not mistaken for a broken prefix.
"""

from __future__ import annotations

import json
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .segments import FROZEN, PINNED, RECENT, SUMMARY, TAIL

COLD_START = "cold_start"
EXPECTED_APPEND = "expected_append"
EXPECTED_TAIL_ONLY = "expected_tail_only"
EXPECTED_EPOCH = "expected_epoch"
VIOLATION = "violation"


@dataclass(frozen=True)
class Fingerprint:
    """What one render leaves for the next lint: ids, spans, and the logical segment hashes."""

    token_ids: tuple[int, ...]
    prompt_sha256: str
    stable_prefix_tokens: int | None  # tokens before the tail; None when spans are unavailable
    pinned_end_tokens: int | None  # end of the pinned segment (the epoch floor)
    segment_hashes: Mapping[str, str | None]
    recent_hashes: tuple[str, ...]
    epoch: int
    spans: tuple[tuple[str, int, int], ...] = ()

    def segment_at(self, index: int) -> str | None:
        return next((name for name, start, end in self.spans if start <= index < end), None)


@dataclass(frozen=True)
class LintResult:
    status: str
    first_divergent_index: int | None
    observed_common_prefix_tokens: int
    expected_common_prefix_floor: int | None
    first_changed_segment: str | None
    divergence_cause: str
    previous_prompt_sha256: str | None
    stable_prefix_tokens: int | None
    detail: dict[str, Any] = field(default_factory=dict)

    def fields(self) -> dict[str, Any]:
        """The ``context_render`` PrefixLint fields."""
        return {
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "expected_common_prefix_floor": self.expected_common_prefix_floor,
            "observed_common_prefix_tokens": self.observed_common_prefix_tokens,
            "first_divergent_index": self.first_divergent_index,
            "first_changed_segment": self.first_changed_segment,
            "divergence_cause": self.divergence_cause,
            "lint_status": self.status,
            "previous_prompt_sha256": self.previous_prompt_sha256,
        }


def common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def logical_change(previous: Fingerprint, current: Fingerprint) -> tuple[str | None, str]:
    """The first segment whose logical content changed, and why (appends and epochs are named as such)."""
    epoch_changed = current.epoch != previous.epoch
    for name in (FROZEN, PINNED, SUMMARY):
        if previous.segment_hashes.get(name) != current.segment_hashes.get(name):
            if name == FROZEN:
                return FROZEN, "frozen_changed"
            if name == SUMMARY and epoch_changed:
                return SUMMARY, "epoch_transition"
            if epoch_changed:
                return name, "epoch_transition"
            return name, "pinned_changed_mid_epoch" if name == PINNED else "summary_changed_without_epoch"
    if epoch_changed:
        return RECENT, "epoch_transition"
    kept = current.recent_hashes[: len(previous.recent_hashes)]
    if kept != previous.recent_hashes:
        return RECENT, "recent_rewritten"
    if len(current.recent_hashes) > len(previous.recent_hashes):
        return RECENT, "recent_append"
    if previous.segment_hashes.get(TAIL) != current.segment_hashes.get(TAIL):
        return TAIL, "tail_rewrite"
    return None, "unchanged"


def lint(previous: Fingerprint | None, current: Fingerprint) -> LintResult:
    """Compare ``current`` with the render before it (see the module docstring)."""
    if previous is None:
        return LintResult(COLD_START, None, 0, 0, None, COLD_START, None, current.stable_prefix_tokens)
    common = common_prefix(previous.token_ids, current.token_ids)
    identical = common == len(previous.token_ids) == len(current.token_ids)
    first = None if identical else common
    segment, cause = logical_change(previous, current)
    epoch_changed = current.epoch != previous.epoch
    floor = previous.pinned_end_tokens if epoch_changed else previous.stable_prefix_tokens
    detail: dict[str, Any] = {}
    if floor is None:
        # Without spans there is no per-segment floor; only an exact logical comparison can be made.
        allowed = cause in ("recent_append", "tail_rewrite", "unchanged", "epoch_transition")
        status = VIOLATION if not allowed else (
            EXPECTED_EPOCH if epoch_changed else EXPECTED_APPEND if cause == "recent_append" else EXPECTED_TAIL_ONLY)
        detail["floor"] = "span_mapping_unavailable"
    elif common >= floor and cause not in ("frozen_changed", "pinned_changed_mid_epoch",
                                           "summary_changed_without_epoch", "recent_rewritten"):
        status = EXPECTED_EPOCH if epoch_changed else EXPECTED_APPEND if cause == "recent_append" else EXPECTED_TAIL_ONLY
    else:
        status = VIOLATION
        if cause in ("recent_append", "tail_rewrite", "unchanged"):
            # No logical segment changed before the floor, yet the tokens did: a tokenization effect.
            segment, cause = previous.segment_at(common) or segment, "token_boundary_merge"
    if status == VIOLATION and first is None:
        first = common
    return LintResult(status, first, common, floor, segment, cause, previous.prompt_sha256,
                      current.stable_prefix_tokens, detail)


class FingerprintStore:
    """The previous prompt's token ids as a compact local binary file (never JSON arrays in the journal).

    Resume re-derives every fingerprint by replaying the journal through the
    session. This store is the local checkpoint used to diff prompts across
    processes and by the live prefix smoke.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def save(self, key: str, fingerprint: Fingerprint) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        ids = array("I", fingerprint.token_ids)
        binary = self.directory / f"{key}.ids"
        temporary = binary.with_name(f".{binary.name}.tmp")
        temporary.write_bytes(ids.tobytes())
        temporary.replace(binary)
        meta = {
            "prompt_sha256": fingerprint.prompt_sha256, "stable_prefix_tokens": fingerprint.stable_prefix_tokens,
            "pinned_end_tokens": fingerprint.pinned_end_tokens, "segment_hashes": dict(fingerprint.segment_hashes),
            "recent_hashes": list(fingerprint.recent_hashes), "epoch": fingerprint.epoch,
            "spans": [list(span) for span in fingerprint.spans], "itemsize": ids.itemsize,
        }
        meta_path = self.directory / f"{key}.json"
        temporary = meta_path.with_name(f".{meta_path.name}.tmp")
        temporary.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
        temporary.replace(meta_path)
        return binary

    def load(self, key: str) -> Fingerprint | None:
        binary, meta_path = self.directory / f"{key}.ids", self.directory / f"{key}.json"
        if not binary.exists() or not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ids = array("I")
        if ids.itemsize != meta["itemsize"]:
            raise ValueError("The fingerprint was written with a different integer width")
        ids.frombytes(binary.read_bytes())
        return Fingerprint(
            tuple(ids), meta["prompt_sha256"], meta["stable_prefix_tokens"], meta["pinned_end_tokens"],
            meta["segment_hashes"], tuple(meta["recent_hashes"]), meta["epoch"],
            tuple(tuple(span) for span in meta["spans"]),
        )


__all__ = [
    "COLD_START", "EXPECTED_APPEND", "EXPECTED_EPOCH", "EXPECTED_TAIL_ONLY", "Fingerprint", "FingerprintStore",
    "LintResult", "VIOLATION", "common_prefix", "lint", "logical_change",
]
