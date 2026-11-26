"""Per-attempt backend evidence from llama-server responses, sanitized for records.

llama.cpp reports token counts and timings in a ``timings`` object plus a few
top-level fields. Their meaning differs between builds: on the pinned builds
``timings.prompt_n`` appears to count only the prompt tokens evaluated for this
request, while ``tokens_evaluated`` counts the whole prompt. Nothing here
interprets them. A repeated-prompt probe on the pinned build has to confirm
the meaning (see ``tidemark.telemetry``). Values are copied only when they are
finite, non-negative numbers under a fixed whitelist of names. Strings,
booleans, NaN, infinities, negative values, and unknown keys are dropped, so an
attempt record never carries headers, credentials, prompts, or server text.

``ttft_ms`` is the time to the first non-empty streamed content event, and it
exists only on the ``buffered_stream`` generation path. A non-streaming
attempt has ``ttft_ms=None`` and ``measurement_path="unavailable"``. Its total
elapsed time is never relabeled TTFT.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

GENERATION_PATHS = ("non_streaming", "buffered_stream")
# Fields llama-server puts in ``timings``. Counts are integers; the rest are milliseconds or rates.
TIMING_FIELDS = (
    "prompt_n",
    "prompt_ms",
    "prompt_per_token_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_token_ms",
    "predicted_per_second",
    "cache_n",
)
# Top-level response fields.
RESPONSE_FIELDS = ("tokens_evaluated", "tokens_cached", "tokens_predicted", "id_slot")
COUNT_FIELDS = frozenset({
    "prompt_n", "predicted_n", "cache_n", "tokens_evaluated", "tokens_cached", "tokens_predicted", "id_slot",
})


def finite_number(value: Any, *, integer: bool = False) -> int | float | None:
    """``value`` as a finite non-negative number, or ``None``. Booleans and strings are not numbers.

    With ``integer``, a float has to be integral (``12.0`` becomes 12, ``12.5`` is refused).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value < 0:
        return None
    if integer:
        if isinstance(value, float):
            return int(value) if value.is_integer() else None
        return int(value)
    return float(value) if isinstance(value, float) else value


def sanitize_backend_fields(response: Mapping[str, Any] | None) -> dict[str, int | float]:
    """The whitelisted numeric fields of one llama-server response, flattened into one dict."""
    if not isinstance(response, Mapping):
        return {}
    clean: dict[str, int | float] = {}
    timings = response.get("timings")
    if isinstance(timings, Mapping):
        for name in TIMING_FIELDS:
            value = finite_number(timings.get(name), integer=name in COUNT_FIELDS)
            if value is not None:
                clean[name] = value
    for name in RESPONSE_FIELDS:
        value = finite_number(response.get(name), integer=True)
        if value is not None:
            clean[name] = value
    return clean


@dataclass(frozen=True)
class AttemptTiming:
    """What one generation attempt measured. ``outcome`` matches the Toolsmith recorder's labels."""

    attempt: int
    outcome: str
    measurement_path: str
    response_elapsed_ms: float | None
    ttft_ms: float | None = None
    backend: dict[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.measurement_path not in ("buffered_stream", "unavailable"):
            raise ValueError(f"Unknown measurement path {self.measurement_path!r}")
        if self.measurement_path == "unavailable" and self.ttft_ms is not None:
            raise ValueError("A non-streaming attempt cannot have a TTFT")

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "outcome": self.outcome,
            "measurement_path": self.measurement_path,
            "response_elapsed_ms": _rounded(self.response_elapsed_ms),
            "ttft_ms": _rounded(self.ttft_ms),
            "backend": dict(self.backend),
        }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


__all__ = [
    "AttemptTiming",
    "COUNT_FIELDS",
    "GENERATION_PATHS",
    "RESPONSE_FIELDS",
    "TIMING_FIELDS",
    "finite_number",
    "sanitize_backend_fields",
]
