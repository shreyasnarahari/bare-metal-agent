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

from toolsmith.artifacts import copy_file_atomic, copy_snapshot
from toolsmith.json_utils import StrictJSONError, strict_json_loads
from toolsmith.recording import JsonlRecorder

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
}
# Fields an event carries in exactly one of its forms.
ONE_OF_FIELDS: dict[str, frozenset[str]] = {"tool_result": frozenset({"value", "error"})}
# Fields written only when they differ from the default a reader assumes.
OPTIONAL_FIELDS: dict[str, dict[str, Any]] = {"episode_started": {"recovery_policy": "typed_hint"}}
# Timing differs on every run, so it never takes part in replay equality.
VOLATILE_FIELDS: dict[str, frozenset[str]] = {
    "model_turn": frozenset({"elapsed_s"}),
    "tool_result": frozenset({"elapsed_s"}),
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
    ) -> dict[str, Any]:
        optional = {} if recovery_policy == "typed_hint" else {"recovery_policy": recovery_policy}
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


def dump_record(record: Mapping[str, Any]) -> str:
    """A record serialized exactly as ``JsonlRecorder`` writes it."""
    return json.dumps(dict(record), ensure_ascii=False, sort_keys=True)


__all__ = [
    "ENVIRONMENT_SECRETS",
    "EVENT_FIELDS",
    "EpisodeJournal",
    "JOURNAL_VERSION",
    "JournalCopier",
    "JournalFormatError",
    "LOCAL_JOURNAL_DIR",
    "ReplayMismatch",
    "check_record",
    "comparable",
    "dump_record",
    "first_difference",
    "manifest_path_for",
    "new_episode_id",
    "read_journal",
]
