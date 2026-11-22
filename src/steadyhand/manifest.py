"""The episode manifest: what an episode was run with, and the key for claim and resume.

It sits beside the journal as ``<episode_id>.manifest.json``. It is created
once, atomically and only if absent (the claim), and rewritten once to set
``completed``. The episode id is ``ep_`` plus a prefix of the hash of the
identity fields, so the same task under the same configuration always gets
the same id.

Server-shape fields (``server_parallel``, ``context_size``,
``per_slot_context``) are not part of the id. Resume compares them, so a
journal recorded against a different slot size is refused rather than resumed.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import toolsmith
from toolsmith.evaluation.models import canonical_hash
from toolsmith.evaluation.runner import atomic_write_json
from toolsmith.json_utils import StrictJSONError, strict_json_loads

from . import __version__ as STEADYHAND_VERSION
from .faults import FaultSchedule
from .journal import JOURNAL_VERSION, JournalFormatError, ReplayMismatch
from .messages import PROMPT_VARIANTS
from .registry import SteadyhandRegistry
from .store import Store

IDENTITY_FIELDS = (
    "journal_version",
    "profile",
    "model",
    "protocol",
    "use_grammars",
    "enable_thinking",
    "seed",
    "temperature",
    "max_tokens",
    "budgets",
    "guard_limit",
    "paraphrase_final",
    "fixture_sha256",
    "schema_sha256",
    "policy_sha256",
    "system_prompt_sha256",
    "task_sha256",
    "fault_schedule_sha256",
    "run_label",
    "recovery_policy",
    "system_prompt_variant",
)
SERVER_FIELDS = ("server_parallel", "context_size", "per_slot_context")


@dataclass(frozen=True)
class EpisodeManifest:
    episode_id: str
    steadyhand_version: str
    toolsmith_version: str
    journal_version: int
    profile: str | None
    model: str | None
    protocol: str | None
    use_grammars: bool | None
    enable_thinking: bool | None
    seed: int | None
    temperature: float | None
    max_tokens: int | None
    budgets: dict[str, Any]
    guard_limit: int
    paraphrase_final: bool
    fixture_sha256: str
    schema_sha256: str
    policy_sha256: str
    system_prompt_sha256: str
    task_sha256: str
    fault_schedule_sha256: str | None = None
    run_label: str | None = None  # distinguishes deliberate repeats of the same task
    # The full schedule, overlay pins included, so replay rebuilds rather than re-rolls it.
    fault_schedule: dict[str, Any] | None = None
    # Part 6. Absent in older manifests, which ran the default strategy and prompt.
    recovery_policy: str = "typed_hint"
    system_prompt_variant: str = "plain"
    server_parallel: int | None = None
    context_size: int | None = None
    per_slot_context: int | None = None
    completed: bool = False

    def identity(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in IDENTITY_FIELDS}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EpisodeManifest:
        """Unknown keys are refused; optional fields may be absent (older manifests)."""
        known = {item.name for item in fields(cls)}
        required = {item.name for item in fields(cls) if item.default is MISSING}
        missing, unexpected = required - set(value), set(value) - known
        if missing or unexpected:
            problem = f"missing {sorted(missing)}" if missing else f"unexpected {sorted(unexpected)}"
            raise JournalFormatError(f"Manifest has {problem}", field="manifest")
        return cls(**dict(value))


def episode_id_for(identity: Mapping[str, Any]) -> str:
    return "ep_" + canonical_hash(dict(identity))[:16]


def system_prompt_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash([message.get("content") for message in messages if message.get("role") == "system"])


def build_manifest(
    settings: Mapping[str, Any],
    registry: SteadyhandRegistry,
    store: Store,
    messages: Sequence[Mapping[str, Any]],
    *,
    budgets: Mapping[str, Any],
    guard_limit: int,
    paraphrase_final: bool = False,
    run_label: str | None = None,
    server_parallel: int | None = None,
    context_size: int | None = None,
    fault_schedule: FaultSchedule | None = None,
    recovery_policy: str = "typed_hint",
    system_prompt_variant: str = "plain",
) -> EpisodeManifest:
    """The manifest for an episode about to start on ``store`` (hashed before any write)."""
    task = [dict(message) for message in messages]
    _check_prompt_variant(task, system_prompt_variant)
    values: dict[str, Any] = {
        "journal_version": JOURNAL_VERSION,
        "profile": settings.get("profile"),
        "model": settings.get("model"),
        "protocol": settings.get("protocol"),
        "use_grammars": settings.get("use_grammars"),
        "enable_thinking": settings.get("enable_thinking"),
        "seed": settings.get("seed"),
        "temperature": settings.get("temperature"),
        "max_tokens": settings.get("max_tokens"),
        "budgets": dict(budgets),
        "guard_limit": guard_limit,
        "paraphrase_final": paraphrase_final,
        "fixture_sha256": store.snapshot_sha256(),
        "schema_sha256": registry.schemas_sha256(),
        "policy_sha256": registry.policies_sha256(),
        "system_prompt_sha256": system_prompt_sha256(task),
        "task_sha256": canonical_hash(task),
        "fault_schedule_sha256": fault_schedule.sha256 if fault_schedule is not None else None,
        "run_label": run_label,
        "recovery_policy": recovery_policy,
        "system_prompt_variant": system_prompt_variant,
    }
    return EpisodeManifest(
        episode_id=episode_id_for(values),
        steadyhand_version=STEADYHAND_VERSION,
        toolsmith_version=toolsmith.__version__,
        server_parallel=server_parallel,
        context_size=context_size,
        per_slot_context=settings.get("slot_context"),
        fault_schedule=fault_schedule.as_dict() if fault_schedule is not None else None,
        **values,
    )


def _check_prompt_variant(messages: Sequence[Mapping[str, Any]], variant: str) -> None:
    """The variant label must describe the system prompt actually sent."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown system prompt variant {variant!r}")
    addendum = PROMPT_VARIANTS[variant]
    system = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
    if addendum and not system.endswith(addendum):
        raise ValueError(f"The system prompt does not carry the {variant!r} addendum")
    for name, other in PROMPT_VARIANTS.items():
        if other and name != variant and system.endswith(other):
            raise ValueError(f"The system prompt carries the {name!r} addendum but is labeled {variant!r}")


def load_manifest(path: str | Path) -> EpisodeManifest:
    try:
        value = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    except StrictJSONError as exc:
        raise JournalFormatError(f"Manifest {path} is not strict JSON: {exc}", field="manifest") from exc
    if not isinstance(value, dict):
        raise JournalFormatError(f"Manifest {path} is not a JSON object", field="manifest")
    return EpisodeManifest.from_dict(value)


def create_manifest(path: str | Path, manifest: EpisodeManifest) -> bool:
    """Write ``manifest`` only if ``path`` does not exist yet. False means another worker has it.

    The content is written to a temporary file and then hard-linked into place,
    so a reader never sees a half-written manifest and only one creator wins.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, destination)
        return True
    except FileExistsError:
        return False
    finally:
        temporary.unlink(missing_ok=True)


def mark_completed(path: str | Path) -> EpisodeManifest:
    completed = replace(load_manifest(path), completed=True)
    atomic_write_json(path, completed.as_dict())
    return completed


def require_match(
    recorded: EpisodeManifest,
    current: EpisodeManifest,
    fields: Sequence[str] = (*IDENTITY_FIELDS, *SERVER_FIELDS),
) -> None:
    """Raise ``ReplayMismatch`` naming the first field where the two manifests disagree."""
    for name in fields:
        if getattr(recorded, name) != getattr(current, name):
            raise ReplayMismatch(
                f"Manifest {name} was {getattr(recorded, name)!r}, the current run has "
                f"{getattr(current, name)!r}",
                field=name,
            )


def try_claim(path: str | Path) -> bool:
    """Create a claim marker exclusively. False means another worker holds it."""
    try:
        descriptor = os.open(Path(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as marker:
        marker.write(f"{os.getpid()}\n")
    return True


__all__ = [
    "EpisodeManifest",
    "IDENTITY_FIELDS",
    "SERVER_FIELDS",
    "build_manifest",
    "create_manifest",
    "episode_id_for",
    "load_manifest",
    "mark_completed",
    "require_match",
    "system_prompt_sha256",
    "try_claim",
]
