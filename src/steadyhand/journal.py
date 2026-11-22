"""One append-only JSONL journal per episode, written through ``JsonlRecorder``.

Redaction stays inside ``JsonlRecorder``: keys such as ``authorization``,
``hf_token``, and ``local_key`` are replaced, and so is every secret value
passed in. ``HF_TOKEN`` and ``LOCAL_KEY`` from the environment are always
added to the secret list. Parallel episodes never share a file.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from toolsmith.recording import JsonlRecorder

from .types import EpisodeStatus, ModelTurn, ToolResult

ENVIRONMENT_SECRETS = ("HF_TOKEN", "LOCAL_KEY")
# Local runs write under the gitignored results/ tree. Colab passes /content/steadyhand.
LOCAL_JOURNAL_DIR = Path(__file__).resolve().parents[2] / "results" / "steadyhand"
_EPISODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def new_episode_id() -> str:
    return f"ep_{uuid.uuid4().hex[:16]}"


class EpisodeJournal:
    def __init__(
        self,
        path: str | Path,
        *,
        episode_id: str,
        secrets: Iterable[str] = (),
    ) -> None:
        if not _EPISODE_ID.fullmatch(episode_id):
            raise ValueError(f"Episode ids must be short and filename-safe: {episode_id!r}")
        environment = tuple(os.environ.get(name, "") for name in ENVIRONMENT_SECRETS)
        self.path = Path(path)
        self.episode_id = episode_id
        self._recorder = JsonlRecorder(self.path, secrets=(*secrets, *environment))
        self._seq = 0
        # Redacted snapshots of every appended line, so the controller can render
        # its final message from exactly what the file holds.
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

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"event": event, "episode_id": self.episode_id, "seq": self._seq, **fields}
        snapshot = self._recorder.append(record)
        self._seq += 1
        self.records.append(snapshot)
        return snapshot

    def episode_started(
        self,
        *,
        settings: Mapping[str, Any],
        fixture_sha256: str | None,
        store_sha256: str,
        budgets: Mapping[str, Any],
        guard_limit: int,
        tool_names: Sequence[str],
        schema_sha256: str,
        policy_sha256: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self.append(
            "episode_started",
            **dict(settings),
            fixture_sha256=fixture_sha256,
            store_sha256=store_sha256,
            budgets=dict(budgets),
            guard_limit=guard_limit,
            tool_names=list(tool_names),
            schema_sha256=schema_sha256,
            policy_sha256=policy_sha256,
            messages=[dict(message) for message in messages],
        )

    def model_turn(self, *, step: int, turn: ModelTurn) -> dict[str, Any]:
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


__all__ = ["ENVIRONMENT_SECRETS", "EpisodeJournal", "LOCAL_JOURNAL_DIR", "new_episode_id"]
