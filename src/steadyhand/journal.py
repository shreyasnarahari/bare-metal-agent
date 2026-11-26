"""One append-only JSONL journal per episode, written through ``JsonlRecorder``.

Redaction stays inside ``JsonlRecorder``: keys such as ``authorization``,
``hf_token``, and ``local_key`` are replaced, and so is every secret value
passed in. ``HF_TOKEN`` and ``LOCAL_KEY`` from the environment are always
added to the secret list. Parallel episodes never share a file.

The event schema is frozen at ``JOURNAL_VERSION``. ``read_journal`` refuses
unknown events, unexpected or missing fields, and any gap or duplicate in
``seq``. Changing a field means bumping the version. Adding a new event type
does not: files written before it still read correctly (``injected_fault``
arrived in part 5 this way, the recovery events in part 6). An optional field
is written only when it differs from its default, which a reader assumes when
the field is absent (``recovery_policy`` defaults to ``typed_hint``).

Bulkhead's security events are added the same way: ``policy_decision``,
``approval_requested``, ``taint_transition``, ``sandbox_execution``, and
``egress_decision``. Their exact fields and value rules live in
``bulkhead.audit``; every value is checked both when a record is written and
when a journal is read (enum labels, hashes, bounds, and consistency, such as a
verdict matching its rule). They carry hashes, rule ids, labels, sizes, and
bounded details, never raw arguments, command lines, URLs, or secrets.
``episode_started`` carries a ``bulkhead`` summary only when Bulkhead is
enabled, so journals written without it are byte-for-byte unchanged.

Tidemark's context events are added the same way, and only on episodes run
with a context session: ``context_render`` (immediately before its
``model_turn``) and ``backend_timings`` (immediately after it).
``episode_started`` then carries a ``context`` block. Their values are checked
on write and on read (hashes, non-negative counts, ordered dropped ranges). In
``backend_timings`` the counts, the measurement path, and the slot id compare
exactly; the milliseconds (``attempt_timings``, ``ttft_ms``, ``prompt_ms``,
``predicted_ms``, ``response_elapsed_ms``) are volatile. Missing values stay
null and never become zero. Journals written without a session are unchanged.

Only genuinely variable fields are volatile: model and tool timings, and a
sandbox run's or egress decision's elapsed time, resource usage, and proxy
byte counts. Verdicts, rule ids, taint labels, request hashes, and every
observation compare exactly, so editing any of them is a ``ReplayMismatch``
naming the record and field.

Security records are evidence of what the control plane decided and what a
backend reported, not proof of enforcement. Replay re-derives the decisions and
serves recorded ``sandbox_execution`` results back, which shows the control
plane reproduces; it does not show that a kernel enforced anything. Only a live
run under the real backend (``steadyhand.replay.live_replay_report``) tests that.

A journal built with ``expected=`` records is in verify mode: each appended
record is compared with the recorded line at the same ``seq`` instead of
being written. Replay uses it with ``write=False``; resume uses it with
``write=True`` so events past the recorded prefix are appended.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bulkhead import audit
from toolsmith.artifacts import copy_file_atomic, copy_snapshot
from toolsmith.json_utils import StrictJSONError, strict_json_loads
from toolsmith.recording import JsonlRecorder

from .context import RENDER_OK, RENDER_STATUSES
from .types import EpisodeStatus, ModelTurn, ToolResult

JOURNAL_VERSION = 1
ENVIRONMENT_SECRETS = ("HF_TOKEN", "LOCAL_KEY")
# Local runs write under the gitignored results/ tree. Colab passes /content/steadyhand.
LOCAL_JOURNAL_DIR = Path(__file__).resolve().parents[2] / "results" / "steadyhand"
_EPISODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")

COMMON_FIELDS = frozenset({"event", "episode_id", "seq", "recorded_at"})
EVENT_FIELDS: dict[str, frozenset[str]] = {
    "episode_started": frozenset({
        "journal_version", "settings", "fixture_sha256", "store_sha256", "budgets",
        "guard_limit", "paraphrase_final", "tool_names", "schema_sha256", "policy_sha256",
        "messages",
    }),
    "budget_checked": frozenset({
        "step", "steps", "prompt_tokens", "completion_tokens", "elapsed_s", "budgets",
        "exhausted", "slot_overflow",
    }),
    "model_turn": frozenset({
        "step", "kind", "assistant_message", "usage", "error_code", "error_message",
        "raw_output", "repair_attempts", "elapsed_s", "messages_sha256", "tool_choice",
    }),
    "tool_result": frozenset({
        "step", "tool_call_id", "tool", "arguments", "arguments_sha256", "ok",
        "handler_called", "content", "elapsed_s",
    }),
    "observation": frozenset({"step", "role", "content", "observation"}),
    "guard_tripped": frozenset({
        "step", "action", "tool", "tool_call_id", "arguments", "call_hash", "count",
        "limit", "last_outcome",
    }),
    "escalated": frozenset({"step", "reason"}),
    # Policy-layer calls made by a recovery strategy (retry, verify, switch).
    "policy_attempt": frozenset({
        "step", "tool_call_id", "purpose", "tool", "arguments", "ok", "error_type",
        "handler_called", "idempotency_key", "waited_s",
    }),
    "sanity_checked": frozenset({"step", "tool_call_id", "tool", "passed", "problem"}),
    # Bulkhead security events (policy_decision, approval_requested, taint_transition,
    # sandbox_execution, egress_decision); fields and value rules come from bulkhead.audit.
    **{event: frozenset(fields) for event, fields in audit.SECURITY_EVENT_FIELDS.items()},
    "recovery": frozenset({"step", "tool_call_id", "tool", "strategy", "recovered_by", "escalated"}),
    # Tool-side faults only. Model-output failures stay on model_turn.
    "injected_fault": frozenset({
        "step", "tool", "tool_call_id", "call_index", "mode", "pinned_mode", "committed",
        "error_type", "note", "via", "schedule_sha256",
    }),
    "episode_finished": frozenset({
        "status", "steps", "final_content", "prompt_tokens", "completion_tokens",
        "error_code", "error_message", "store_sha256",
    }),
    # Tidemark context events (only with a context session; see steadyhand.context).
    "context_render": frozenset({
        "step", "epoch", "strategy", "status", "detail", "context_config_sha256", "messages_sha256",
        "prompt_sha256", "prompt_token_ids_sha256", "rendered_prompt_tokens", "segment_budgets",
        "segment_tokens", "max_tokens", "slot_headroom", "epoch_start_seq", "history_end_seq",
        "recitation_sha256", "tool_schema_sha256", "compaction_pending", "dropped_ranges",
    }),
    "backend_timings": frozenset({
        "step", "attempt_metrics", "attempt_timings", "rendered_prompt_tokens", "prompt_evaluated_tokens",
        "reused_tokens", "cache_reuse_fraction", "cache_status", "ttft_ms", "prompt_ms", "predicted_ms",
        "response_elapsed_ms", "slot_id", "measurement_path",
    }),
}
CONTEXT_EVENTS = frozenset({"context_render", "backend_timings"})
# Fields an event carries in exactly one of its forms.
ONE_OF_FIELDS: dict[str, frozenset[str]] = {"tool_result": frozenset({"value", "error"})}
# Fields written only when they differ from the default a reader assumes.
OPTIONAL_FIELDS: dict[str, dict[str, Any]] = {
    "episode_started": {"recovery_policy": "typed_hint", "bulkhead": None, "context": None},
    "context_render": {"lint": None},  # PrefixLint evidence (Tidemark part 2)
}
# Timing differs on every run, so it never takes part in replay equality.
VOLATILE_FIELDS: dict[str, frozenset[str]] = {
    "model_turn": frozenset({"elapsed_s"}),
    "tool_result": frozenset({"elapsed_s"}),
    "backend_timings": frozenset({"attempt_timings", "ttft_ms", "prompt_ms", "predicted_ms", "response_elapsed_ms"}),
    **audit.VOLATILE_SECURITY_FIELDS,
}


class ReplayMismatch(RuntimeError):
    """The code no longer reproduces the recorded episode, or the journal is not replayable."""

    def __init__(self, message: str, *, seq: int | None = None, field: str | None = None) -> None:
        location = []
        if seq is not None:
            location.append(f"seq {seq}")
        if field is not None:
            location.append(f"field {field}")
        super().__init__(f"{message} ({', '.join(location)})" if location else message)
        self.seq = seq
        self.field = field


class JournalFormatError(ReplayMismatch):
    """The file is not a well-formed journal of the current version."""


def new_episode_id() -> str:
    """A random id for an ad-hoc episode (claimed episodes derive theirs from the manifest)."""
    return f"ep_{uuid.uuid4().hex[:16]}"


def manifest_path_for(journal_path: str | Path) -> Path:
    """``<episode_id>.manifest.json`` beside ``<episode_id>.jsonl``."""
    path = Path(journal_path)
    return path.with_name(f"{path.stem}.manifest.json")


def comparable(record: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a record replay compares: everything except timestamps and timings."""
    ignored = {"recorded_at"} | VOLATILE_FIELDS.get(str(record.get("event")), frozenset())
    return {key: value for key, value in record.items() if key not in ignored}


def first_difference(expected: Any, actual: Any, path: str = "") -> str | None:
    """A JSON-Pointer-like path to the first place two JSON values differ, or None."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                return f"{path}/{key}"
            found = first_difference(expected[key], actual[key], f"{path}/{key}")
            if found is not None:
                return found
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return path or "/"
        for index, (left, right) in enumerate(zip(expected, actual)):
            found = first_difference(left, right, f"{path}/{index}")
            if found is not None:
                return found
        return None
    if type(expected) is not type(actual) or expected != actual:
        return path or "/"
    return None


def read_journal(path: str | Path, *, allow_torn_tail: bool = False) -> list[dict[str, Any]]:
    """Parse and check a journal. ``allow_torn_tail`` drops an unterminated last line (resume only)."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    tail = lines.pop()  # "" when the file ends with a newline
    if tail:
        if not allow_torn_tail:
            raise JournalFormatError(f"{path} ends with an unterminated line", seq=len(lines))
    records = []
    for index, line in enumerate(lines):
        try:
            record = strict_json_loads(line)
        except StrictJSONError as exc:
            raise JournalFormatError(f"{path} line {index + 1} is not strict JSON: {exc}", seq=index) from exc
        if not isinstance(record, dict):
            raise JournalFormatError(f"{path} line {index + 1} is not a JSON object", seq=index)
        check_record(record, index)
        records.append(record)
    if not records:
        raise JournalFormatError(f"{path} has no events")
    if records[0]["event"] != "episode_started":
        raise JournalFormatError("A journal must start with episode_started", seq=0, field="event")
    version = records[0].get("journal_version")
    if version != JOURNAL_VERSION:
        raise JournalFormatError(
            f"Journal version {version!r} is not {JOURNAL_VERSION}", seq=0, field="journal_version"
        )
    episode_ids = {record["episode_id"] for record in records}
    if len(episode_ids) != 1:
        raise JournalFormatError(f"A journal holds one episode, found {sorted(episode_ids)}", field="episode_id")
    return records


def check_record(record: Mapping[str, Any], expected_seq: int) -> None:
    """Check one record against the frozen schema: a known event, the next seq, exactly its fields."""
    event = record.get("event")
    if event not in EVENT_FIELDS:
        raise JournalFormatError(f"Unknown journal event {event!r}", seq=expected_seq, field="event")
    if record.get("seq") != expected_seq:
        raise JournalFormatError(
            f"Expected seq {expected_seq}, found {record.get('seq')!r}; seq must have no gaps or duplicates",
            seq=expected_seq,
            field="seq",
        )
    required = COMMON_FIELDS | EVENT_FIELDS[event]
    one_of = ONE_OF_FIELDS.get(event, frozenset())
    optional = set(OPTIONAL_FIELDS.get(event, {}))
    keys = set(record)
    missing = required - keys
    unexpected = keys - required - one_of - optional
    if missing or unexpected:
        problem = f"missing {sorted(missing)}" if missing else f"unexpected {sorted(unexpected)}"
        raise JournalFormatError(f"{event} has {problem}", seq=expected_seq, field="fields")
    if one_of and len(keys & one_of) != 1:
        raise JournalFormatError(f"{event} needs exactly one of {sorted(one_of)}", seq=expected_seq, field="fields")
    if event in audit.SECURITY_EVENT_FIELDS:
        try:
            audit.validate(event, {name: record[name] for name in EVENT_FIELDS[event]})
        except audit.AuditError as exc:
            raise JournalFormatError(f"{event}: {exc}", seq=expected_seq, field=exc.field) from None
    if event in CONTEXT_EVENTS:
        problem = check_context_values(event, record)
        if problem is not None:
            raise JournalFormatError(f"{event}: {problem[1]}", seq=expected_seq, field=problem[0])


_SHA256 = re.compile(r"[0-9a-f]{64}")
MEASUREMENT_PATHS = ("buffered_stream", "unavailable")


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _optional_count(value: Any) -> bool:
    return value is None or _count(value)


def _optional_ms(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and 0 <= value < float("inf")
    )


def _optional_hash(value: Any) -> bool:
    return value is None or (isinstance(value, str) and bool(_SHA256.fullmatch(value)))


def check_context_values(event: str, record: Mapping[str, Any]) -> tuple[str, str] | None:
    """The first bad value in a Tidemark context record, as ``(field, problem)``, or None."""
    if not _count(record.get("step")):
        return "step", "step must be a non-negative integer"
    if event == "context_render":
        if record["status"] not in RENDER_STATUSES:
            return "status", f"unknown status {record['status']!r}"
        if not isinstance(record["strategy"], str) or not record["strategy"]:
            return "strategy", "strategy must be a non-empty string"
        if record["detail"] is not None and not isinstance(record["detail"], str):
            return "detail", "detail must be a string or null"
        for name in ("epoch", "epoch_start_seq", "history_end_seq"):
            if not _count(record[name]):
                return name, f"{name} must be a non-negative integer"
        for name in ("context_config_sha256", "tool_schema_sha256"):
            if record[name] is None or not _optional_hash(record[name]):
                return name, f"{name} must be a sha256 hex digest"
        for name in ("messages_sha256", "prompt_sha256", "prompt_token_ids_sha256", "recitation_sha256"):
            if not _optional_hash(record[name]):
                return name, f"{name} must be a sha256 hex digest or null"
        if (record["status"] == RENDER_OK) != (record["messages_sha256"] is not None):
            return "messages_sha256", "a sent render has a messages hash; an exhausted one has none"
        for name in ("rendered_prompt_tokens", "max_tokens"):
            if not _optional_count(record[name]):
                return name, f"{name} must be a non-negative integer or null"
        headroom = record["slot_headroom"]
        if headroom is not None and (not isinstance(headroom, int) or isinstance(headroom, bool)):
            return "slot_headroom", "slot_headroom must be an integer or null"
        for name in ("segment_budgets", "segment_tokens"):
            value = record[name]
            if not isinstance(value, dict) or not all(isinstance(k, str) and _optional_count(v) for k, v in value.items()):
                return name, f"{name} must map names to non-negative integers or null"
        if not isinstance(record["compaction_pending"], bool):
            return "compaction_pending", "compaction_pending must be a boolean"
        previous_end = -1
        for item in record["dropped_ranges"] if isinstance(record["dropped_ranges"], list) else [None]:
            if not (isinstance(item, list) and len(item) == 2 and all(_count(x) for x in item) and item[0] < item[1]):
                return "dropped_ranges", "dropped_ranges holds [start, end) pairs with start < end"
            if item[0] < previous_end:
                return "dropped_ranges", "dropped_ranges must be sorted and must not overlap"
            previous_end = item[1]
        if "lint" in record and record["lint"] is not None and not isinstance(record["lint"], dict):
            return "lint", "lint must be an object"
        return None
    # backend_timings
    metrics, timings = record["attempt_metrics"], record["attempt_timings"]
    if not isinstance(metrics, list) or not all(isinstance(item, dict) for item in metrics):
        return "attempt_metrics", "attempt_metrics must be a list of objects"
    if not isinstance(timings, list) or len(timings) != len(metrics) or not all(isinstance(item, dict) for item in timings):
        return "attempt_timings", "attempt_timings must hold one object per attempt"
    for name in ("rendered_prompt_tokens", "prompt_evaluated_tokens", "reused_tokens", "slot_id"):
        if not _optional_count(record[name]):
            return name, f"{name} must be a non-negative integer or null"
    fraction = record["cache_reuse_fraction"]
    if fraction is not None and not (
        isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and 0 <= fraction <= 1
    ):
        return "cache_reuse_fraction", "cache_reuse_fraction must be in [0, 1] or null"
    if record["reused_tokens"] is not None and (
        record["rendered_prompt_tokens"] is None or record["prompt_evaluated_tokens"] is None
    ):
        return "reused_tokens", "reused_tokens needs both the rendered and the evaluated count"
    for name in ("ttft_ms", "prompt_ms", "predicted_ms", "response_elapsed_ms"):
        if not _optional_ms(record[name]):
            return name, f"{name} must be a non-negative number or null"
    if record["measurement_path"] not in MEASUREMENT_PATHS:
        return "measurement_path", f"unknown measurement path {record['measurement_path']!r}"
    if record["measurement_path"] == "unavailable" and record["ttft_ms"] is not None:
        return "ttft_ms", "a TTFT needs the buffered_stream measurement path"
    if not isinstance(record["cache_status"], str) or not record["cache_status"]:
        return "cache_status", "cache_status must be a non-empty string"
    return None


class EpisodeJournal:
    """Append-only record of one episode, or (with ``expected``) a checker that replays one.

    Every record gets the next ``seq``. While records from ``expected`` remain,
    each new event is compared with the recorded one instead of being written;
    after that it is written, or, with ``write=False``, refused.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        episode_id: str,
        secrets: Iterable[str] = (),
        expected: Sequence[Mapping[str, Any]] | None = None,
        write: bool = True,
    ) -> None:
        if not _EPISODE_ID.fullmatch(episode_id):
            raise ValueError(f"Episode ids must be short and filename-safe: {episode_id!r}")
        # Secrets in the environment (HF_TOKEN, LOCAL_KEY) are always redacted, even if not passed in.
        environment = tuple(os.environ.get(name, "") for name in ENVIRONMENT_SECRETS)
        self.path = Path(path)
        self.episode_id = episode_id
        self._recorder = JsonlRecorder(self.path, secrets=(*secrets, *environment))
        # The same values, for the output gate's final redaction of model observations.
        self.secrets = tuple(value for value in (*secrets, *environment) if value)
        self._seq = 0
        self._expected = [dict(record) for record in expected or ()]
        self._write = write
        # Snapshots of every line, so the controller renders its final message
        # from exactly what the file holds.
        self.records: list[dict[str, Any]] = []

    @classmethod
    def create(
        cls,
        directory: str | Path | None = None,
        episode_id: str | None = None,
        *,
        secrets: Iterable[str] = (),
    ) -> EpisodeJournal:
        """Start a new journal at ``directory/<episode_id>.jsonl``. Refuses an existing file."""
        episode_id = episode_id or new_episode_id()
        root = Path(directory) if directory is not None else LOCAL_JOURNAL_DIR
        journal = cls(root / f"{episode_id}.jsonl", episode_id=episode_id, secrets=secrets)
        if journal.path.exists():
            raise FileExistsError(f"Journal already exists: {journal.path}")
        return journal

    @property
    def verifying(self) -> bool:
        """True while appended records are still being checked against the recorded prefix."""
        return self._seq < len(self._expected)

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        """Write (or, while replaying, verify) the next record, and return it as stored."""
        record = {"event": event, "episode_id": self.episode_id, "seq": self._seq, **fields}
        if self.verifying:
            recorded = self._expected[self._seq]
            produced = self._recorder.snapshot(record)
            difference = first_difference(comparable(recorded), comparable(produced))
            if difference is not None:
                raise ReplayMismatch(
                    f"Replayed {event} differs from the recorded {recorded.get('event')}",
                    seq=self._seq,
                    field=difference,
                )
            snapshot = recorded
        elif not self._write:
            raise ReplayMismatch(
                f"The code produced {event} after the recorded journal ended", seq=self._seq, field="event"
            )
        else:
            snapshot = self._recorder.append(record)
        self._seq += 1
        self.records.append(snapshot)
        return snapshot

    def verify_complete(self) -> None:
        """In replay, every recorded line must have been reproduced."""
        if self.verifying:
            recorded = self._expected[self._seq]
            raise ReplayMismatch(
                f"The episode ended before the recorded {recorded.get('event')}", seq=self._seq, field="event"
            )

    # One method per event type, so every field is named and checked at the call site.

    def episode_started(
        self,
        *,
        settings: Mapping[str, Any],
        fixture_sha256: str | None,
        store_sha256: str,
        budgets: Mapping[str, Any],
        guard_limit: int,
        paraphrase_final: bool,
        tool_names: Sequence[str],
        schema_sha256: str,
        policy_sha256: str,
        messages: Sequence[Mapping[str, Any]],
        recovery_policy: str = "typed_hint",
        bulkhead: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        optional: dict[str, Any] = {} if recovery_policy == "typed_hint" else {"recovery_policy": recovery_policy}
        if bulkhead is not None:
            optional["bulkhead"] = dict(bulkhead)
        if context is not None:
            optional["context"] = dict(context)
        return self.append(
            "episode_started",
            **optional,
            journal_version=JOURNAL_VERSION,
            settings=dict(settings),
            fixture_sha256=fixture_sha256,
            store_sha256=store_sha256,
            budgets=dict(budgets),
            guard_limit=guard_limit,
            paraphrase_final=paraphrase_final,
            tool_names=list(tool_names),
            schema_sha256=schema_sha256,
            policy_sha256=policy_sha256,
            messages=[dict(message) for message in messages],
        )

    def model_turn(
        self,
        *,
        step: int,
        turn: ModelTurn,
        messages_sha256: str,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> dict[str, Any]:
        """``messages_sha256`` hashes the message list sent to ``complete()``, not the rendered prompt."""
        return self.append(
            "model_turn",
            step=step,
            kind=turn.kind.value,
            assistant_message=turn.assistant_message,
            usage={
                "prompt_tokens": turn.prompt_tokens,
                "completion_tokens": turn.completion_tokens,
            },
            error_code=turn.error_code,
            error_message=turn.error_message,
            raw_output=turn.raw_output,
            repair_attempts=turn.repair_attempts,
            elapsed_s=round(turn.elapsed_s, 6),
            messages_sha256=messages_sha256,
            tool_choice=tool_choice,
        )

    def context_render(
        self,
        *,
        step: int,
        epoch: int,
        strategy: str,
        status: str,
        detail: str | None,
        context_config_sha256: str | None,
        messages_sha256: str | None,
        prompt_sha256: str | None,
        prompt_token_ids_sha256: str | None,
        rendered_prompt_tokens: int | None,
        segment_budgets: Mapping[str, int | None],
        segment_tokens: Mapping[str, int | None],
        max_tokens: int | None,
        slot_headroom: int | None,
        epoch_start_seq: int,
        history_end_seq: int,
        recitation_sha256: str | None,
        tool_schema_sha256: str,
        compaction_pending: bool,
        dropped_ranges: Sequence[Sequence[int]],
        lint: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """What a context session rendered for the next model turn: hashes and counts, never the prompt."""
        fields: dict[str, Any] = {
            "step": step, "epoch": epoch, "strategy": strategy, "status": status, "detail": detail,
            "context_config_sha256": context_config_sha256, "messages_sha256": messages_sha256,
            "prompt_sha256": prompt_sha256, "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "rendered_prompt_tokens": rendered_prompt_tokens, "segment_budgets": dict(segment_budgets),
            "segment_tokens": dict(segment_tokens), "max_tokens": max_tokens, "slot_headroom": slot_headroom,
            "epoch_start_seq": epoch_start_seq, "history_end_seq": history_end_seq,
            "recitation_sha256": recitation_sha256, "tool_schema_sha256": tool_schema_sha256,
            "compaction_pending": compaction_pending, "dropped_ranges": [list(item) for item in dropped_ranges],
        }
        if lint:
            fields["lint"] = dict(lint)
        return self._context("context_render", fields)

    def backend_timings(
        self,
        *,
        step: int,
        attempt_metrics: Sequence[Mapping[str, Any]],
        attempt_timings: Sequence[Mapping[str, Any]],
        rendered_prompt_tokens: int | None,
        prompt_evaluated_tokens: int | None,
        reused_tokens: int | None,
        cache_reuse_fraction: float | None,
        cache_status: str,
        ttft_ms: float | None,
        prompt_ms: float | None,
        predicted_ms: float | None,
        response_elapsed_ms: float | None,
        slot_id: int | None,
        measurement_path: str,
    ) -> dict[str, Any]:
        """The measured cost of the preceding model turn. Counts replay exactly; milliseconds are volatile."""
        return self._context("backend_timings", {
            "step": step,
            "attempt_metrics": [dict(item) for item in attempt_metrics],
            "attempt_timings": [dict(item) for item in attempt_timings],
            "rendered_prompt_tokens": rendered_prompt_tokens,
            "prompt_evaluated_tokens": prompt_evaluated_tokens,
            "reused_tokens": reused_tokens,
            "cache_reuse_fraction": cache_reuse_fraction,
            "cache_status": cache_status,
            "ttft_ms": _round_ms(ttft_ms),
            "prompt_ms": _round_ms(prompt_ms),
            "predicted_ms": _round_ms(predicted_ms),
            "response_elapsed_ms": _round_ms(response_elapsed_ms),
            "slot_id": slot_id,
            "measurement_path": measurement_path,
        })

    def _context(self, event: str, fields: dict[str, Any]) -> dict[str, Any]:
        # Checked before writing, so an invalid context record never reaches the file.
        problem = check_context_values(event, fields)
        if problem is not None:
            raise ValueError(f"{event}: {problem[1]} ({problem[0]})")
        return self.append(event, **fields)

    def tool_result(self, *, step: int, result: ToolResult) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "step": step,
            "tool_call_id": result.call_id,
            "tool": result.tool,
            "arguments": result.arguments,
            "arguments_sha256": result.arguments_sha256,
            "ok": result.ok,
            "handler_called": result.handler_called,
        }
        if result.ok:
            fields["value"] = result.value
        else:
            assert result.error is not None
            fields["error"] = result.error.as_dict()
        fields["content"] = result.content()
        fields["elapsed_s"] = round(result.elapsed_s, 6)
        return self.append("tool_result", **fields)

    def observation(
        self,
        *,
        step: int,
        message: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """A loop-authored message that is not a tool result, such as parse-error feedback."""
        return self.append(
            "observation",
            step=step,
            role=message["role"],
            content=message["content"],
            observation=dict(observation),
        )

    def budget_checked(self, *, step: int, check: Mapping[str, Any]) -> dict[str, Any]:
        """The numbers behind one continue-or-stop decision before a model call."""
        return self.append("budget_checked", step=step, **dict(check))

    def guard_tripped(
        self,
        *,
        step: int,
        action: str,
        tool: str,
        tool_call_id: str,
        arguments: str,
        call_hash: str,
        count: int,
        limit: int,
        last_outcome: str | None,
    ) -> dict[str, Any]:
        """``action`` is ``block`` (handler skipped) or ``escalate`` (the episode stops)."""
        return self.append(
            "guard_tripped",
            step=step,
            action=action,
            tool=tool,
            tool_call_id=tool_call_id,
            arguments=arguments,
            call_hash=call_hash,
            count=count,
            limit=limit,
            last_outcome=last_outcome,
        )

    def escalated(self, *, step: int, reason: str) -> dict[str, Any]:
        return self.append("escalated", step=step, reason=reason)

    def policy_attempt(
        self,
        *,
        step: int,
        tool_call_id: str,
        purpose: str,
        tool: str,
        arguments: str,
        ok: bool,
        error_type: str | None,
        handler_called: bool,
        idempotency_key: str | None,
        waited_s: float,
    ) -> dict[str, Any]:
        """A call the recovery strategy made itself; ``tool_call_id`` is the model call it serves."""
        return self.append(
            "policy_attempt",
            step=step,
            tool_call_id=tool_call_id,
            purpose=purpose,
            tool=tool,
            arguments=arguments,
            ok=ok,
            error_type=error_type,
            handler_called=handler_called,
            idempotency_key=idempotency_key,
            waited_s=waited_s,
        )

    def policy_decision(
        self,
        *,
        step: int,
        tool_call_id: str,
        via: str,
        tool: str,
        arguments_sha256: str,
        verdict: str,
        rule_id: str,
        reason_code: str,
        taint_before: str,
        policy_sha256: str,
        grant_id: str | None,
        normalized_request_sha256: str,
    ) -> dict[str, Any]:
        """One Bulkhead decision. ``tool_result.handler_called`` stays the evidence of execution."""
        return self._security(
            "policy_decision",
            step=step,
            tool_call_id=tool_call_id,
            via=via,
            tool=tool,
            arguments_sha256=arguments_sha256,
            verdict=verdict,
            rule_id=rule_id,
            reason_code=reason_code,
            taint_before=taint_before,
            policy_sha256=policy_sha256,
            grant_id=grant_id,
            normalized_request_sha256=normalized_request_sha256,
        )

    def approval_requested(
        self,
        *,
        step: int,
        tool_call_id: str,
        rule_id: str,
        request_sha256: str,
        normalized_grant: Mapping[str, Any],
        parent_episode_id: str | None,
        approval_mode: str,
    ) -> dict[str, Any]:
        """The privileges an ASK needs (no credentials). ``parent_episode_id`` is this episode's own parent."""
        return self._security(
            "approval_requested",
            step=step,
            tool_call_id=tool_call_id,
            rule_id=rule_id,
            request_sha256=request_sha256,
            normalized_grant=dict(normalized_grant),
            parent_episode_id=parent_episode_id,
            approval_mode=approval_mode,
        )

    def taint_transition(
        self, *, step: int, tool_call_id: str, source_tool: str, before: str, after: str, source_seq: int,
    ) -> dict[str, Any]:
        """The episode became untrusted; ``source_seq`` is the record that carried the outside content."""
        return self._security(
            "taint_transition",
            step=step,
            tool_call_id=tool_call_id,
            source_tool=source_tool,
            before=before,
            after=after,
            source_seq=source_seq,
        )

    def sandbox_execution(self, *, step: int, tool_call_id: str, via: str, tool: str, result: Mapping[str, Any], **audit_fields: Any) -> dict[str, Any]:
        """One backend run: its audit fields (``bulkhead.replay.AUDIT_FIELDS``) and its typed result."""
        return self._security(
            "sandbox_execution", step=step, tool_call_id=tool_call_id, via=via, tool=tool, result=dict(result), **audit_fields,
        )

    def egress_decision(self, **fields: Any) -> dict[str, Any]:
        """One proxy decision: the normalized authority and byte counts, never a URL, body, or header."""
        return self._security("egress_decision", **fields)

    def output_gate(self, **fields: Any) -> dict[str, Any]:
        """What the output gate did to a backend result's streams (counts and flags, never content)."""
        return self._security("output_gate", **fields)

    def _security(self, event: str, **fields: Any) -> dict[str, Any]:
        # Checked before writing, so an invalid security record never reaches the file.
        return self.append(event, **audit.build(event, **fields))

    def sanity_checked(
        self, *, step: int, tool_call_id: str, tool: str, passed: bool, problem: str | None,
    ) -> dict[str, Any]:
        return self.append(
            "sanity_checked", step=step, tool_call_id=tool_call_id, tool=tool, passed=passed, problem=problem,
        )

    def recovery(
        self,
        *,
        step: int,
        tool_call_id: str,
        tool: str,
        strategy: str,
        recovered_by: str | None,
        escalated: str | None,
    ) -> dict[str, Any]:
        return self.append(
            "recovery",
            step=step,
            tool_call_id=tool_call_id,
            tool=tool,
            strategy=strategy,
            recovered_by=recovered_by,
            escalated=escalated,
        )

    def injected_fault(self, *, step: int, fault: Mapping[str, Any]) -> dict[str, Any]:
        """The scorer's record of a tool-side fault; the model never sees it."""
        return self.append("injected_fault", step=step, **dict(fault))

    def episode_finished(
        self,
        *,
        status: EpisodeStatus,
        steps: int,
        final_content: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        error_code: str | None,
        store_sha256: str,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        return self.append(
            "episode_finished",
            status=status.value,
            steps=steps,
            final_content=final_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_code=error_code,
            error_message=error_message,
            store_sha256=store_sha256,
        )


class JournalCopier:
    """Copies finished journals (and other files or calls) off the workers, one at a time.

    Workers never wait long on a slow or hung Drive mount: ``submit`` waits at
    most ``submit_timeout`` seconds for room in the bounded queue, then parks
    the item on an overflow list the copier drains as soon as it catches up.
    A failed copy is retried with backoff before it lands in ``errors``.
    ``close`` waits up to ``close_timeout`` seconds, then records what was left
    (the journals stay on local disk and the next run copies them).
    """

    def __init__(
        self,
        destination: str | Path,
        *,
        depth: int = 2,
        submit_timeout: float = 5.0,
        retries: int = 3,
        retry_delay: float = 2.0,
        close_timeout: float = 600.0,
        sleep: Any = time.sleep,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be positive")
        self.destination = Path(destination)
        self.copied: list[Path] = []
        self.errors: list[tuple[Path, str]] = []
        self.pending_at_close: list[str] = []
        self.submit_timeout = submit_timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.close_timeout = close_timeout
        self._sleep = sleep
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=depth)
        self._overflow: deque[tuple[str, Any]] = deque()
        self._overflow_lock = threading.Lock()
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._run, name="steadyhand-copier", daemon=True)
        self._thread.start()

    def submit(self, journal_path: str | Path) -> None:
        path = Path(journal_path)

        def copy_episode() -> None:
            for source in (path, manifest_path_for(path)):
                if source.exists():
                    self.copied.append(copy_file_atomic(source, self.destination / source.name))

        self._put(str(path), copy_episode)

    def submit_file(self, source: str | Path, destination: str | Path) -> None:
        """Copy any file (a sweep's scores, its manifest) to an exact destination, as it is when copied."""
        source, destination = Path(source), Path(destination)
        self._put(str(source), lambda: self.copied.append(copy_snapshot(source, destination)))

    def submit_call(self, name: str, fn: Any, *args: Any) -> None:
        """Run ``fn(*args)`` on the copier thread (the ``BackgroundJobs.submit`` signature, for ``MirroredFile``)."""
        self._put(name, lambda: fn(*args))

    def close(self) -> None:
        self._closing.set()
        self._thread.join(self.close_timeout)
        if self._thread.is_alive():
            with self._overflow_lock:
                self.pending_at_close = [label for label, _ in list(self._queue.queue) + list(self._overflow)]
            self.errors.append((
                self.destination,
                f"copier still busy after {self.close_timeout:g}s with {len(self.pending_at_close)} copies pending; "
                "the files are still on local disk",
            ))

    def __enter__(self) -> JournalCopier:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _put(self, label: str, action: Any) -> None:
        try:
            self._queue.put((label, action), timeout=self.submit_timeout)
        except queue.Full:
            with self._overflow_lock:
                self._overflow.append((label, action))

    def _next(self) -> tuple[str, Any] | None:
        try:
            return self._queue.get(timeout=0.2)
        except queue.Empty:
            with self._overflow_lock:
                return self._overflow.popleft() if self._overflow else None

    def _run(self) -> None:
        while True:
            item = self._next()
            if item is None:
                if self._closing.is_set() and self._queue.empty():
                    return
                continue
            label, action = item
            for attempt in range(self.retries + 1):
                try:
                    action()
                    break
                except Exception as exc:  # recorded, so one bad copy cannot stop the queue
                    if attempt == self.retries:
                        self.errors.append((Path(label), f"{type(exc).__name__}: {exc}"))
                    else:
                        self._sleep(self.retry_delay * (2 ** attempt))


def _round_ms(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def dump_record(record: Mapping[str, Any]) -> str:
    """A record serialized exactly as ``JsonlRecorder`` writes it."""
    return json.dumps(dict(record), ensure_ascii=False, sort_keys=True)


__all__ = [
    "CONTEXT_EVENTS",
    "ENVIRONMENT_SECRETS",
    "EVENT_FIELDS",
    "EpisodeJournal",
    "JOURNAL_VERSION",
    "JournalCopier",
    "JournalFormatError",
    "LOCAL_JOURNAL_DIR",
    "ReplayMismatch",
    "check_context_values",
    "check_record",
    "comparable",
    "dump_record",
    "first_difference",
    "manifest_path_for",
    "new_episode_id",
    "read_journal",
]
