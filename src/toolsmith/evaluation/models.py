"""Data types shared by every evaluation runner, and the canonical hash used in manifests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Mode(str, Enum):
    """How a case is run: prompt only, grammar-constrained, grammar plus one repair, or thinking."""

    PROMPT = "prompt"
    GRAMMAR = "grammar"
    REPAIR = "repair"
    THINKING = "thinking"
    NATIVE = "native"


@dataclass(frozen=True)
class RunConfig:
    """The settings of one evaluation run (hashed into its manifest)."""

    run_id: str
    dataset: str
    model: str
    mode: Mode
    seed: int = 0
    max_tokens: int = 1024
    temperature: float = 0.0
    context_size: int = 16384

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        return value


@dataclass(frozen=True)
class AttemptRecord:
    """One request to the model for a case; ``reused`` marks an attempt copied from another mode."""

    attempt: int
    raw_response: Any
    elapsed_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: dict[str, Any] | None = None
    reused: bool = False


@dataclass(frozen=True)
class CaseResult:
    """The outcome of one case: status, attempts, parsed calls, and validity at each level.

    Validity is reported separately at each level: the output parsed, matched
    the schema, and was semantically right (``None`` when not judged).
    """

    case_id: str
    category: str
    model: str
    mode: str
    status: str
    attempts: tuple[AttemptRecord, ...] = ()
    parsed_calls: list[dict[str, Any]] = field(default_factory=list)
    content: str | None = None
    reasoning_content: str | None = None
    parse_valid: bool = False
    schema_valid: bool = False
    semantic_valid: bool | None = None
    repair_attempts: int = 0
    grammar_sha256: str | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "model": self.model,
            "mode": self.mode,
            "status": self.status,
            "attempts": [asdict(attempt) for attempt in self.attempts],
            "parsed_calls": self.parsed_calls,
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "parse_valid": self.parse_valid,
            "schema_valid": self.schema_valid,
            "semantic_valid": self.semantic_valid,
            "repair_attempts": self.repair_attempts,
            "grammar_sha256": self.grammar_sha256,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CaseResult:
        return cls(
            case_id=value["case_id"],
            category=value["category"],
            model=value["model"],
            mode=value["mode"],
            status=value["status"],
            attempts=tuple(
                AttemptRecord(**attempt) for attempt in value.get("attempts", [])
            ),
            parsed_calls=value.get("parsed_calls", []),
            content=value.get("content"),
            reasoning_content=value.get("reasoning_content"),
            parse_valid=value.get("parse_valid", False),
            schema_valid=value.get("schema_valid", False),
            semantic_valid=value.get("semantic_valid"),
            repair_attempts=value.get("repair_attempts", 0),
            grammar_sha256=value.get("grammar_sha256"),
            error=value.get("error"),
            metadata=value.get("metadata", {}),
        )


def canonical_hash(value: Any) -> str:
    """sha256 of compact JSON with sorted keys: the same value always gives the same hash."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AttemptRecord",
    "CaseResult",
    "Mode",
    "RunConfig",
    "canonical_hash",
]
