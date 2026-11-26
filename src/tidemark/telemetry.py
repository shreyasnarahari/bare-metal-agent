"""Cache and prefill metrics, derived in one place from rendered and server-reported counts.

Three token counts are kept apart and never substituted for each other:

- **rendered**: the exact prompt measured with the real tokenizer (``ContextSession``);
- **newly evaluated**: what llama-server reports it evaluated for this request;
- **reused**: what it did not have to evaluate because the slot cache held it.

The formulas::

    reused_tokens            = max(0, rendered_prompt_tokens - prompt_evaluated_tokens)
    cache_reuse_fraction     = reused_tokens / rendered_prompt_tokens      # zero denominator -> None
    prefill_savings_fraction = 1 - prompt_evaluated_tokens / naive_evaluated_tokens  # paired rows only

They apply only once a repeated-prompt A/B probe on the pinned build has
confirmed which server field means "newly evaluated" (``probe_cache_semantics``).
Until then, or when a count is missing, or when the server's own total
disagrees with the rendered count, the cache metrics are ``None`` and
``cache_status`` says why. A missing count is never a zero. The first attempt
of a turn defines its cache metrics; repair attempts render a longer prompt,
so they keep their own raw counts in ``attempt_metrics`` and are not folded in.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from steadyhand.types import TurnMetrics
from toolsmith.backends import CompletionTruncatedError
from toolsmith.timings import sanitize_backend_fields

# cache_status labels: why the cache metrics of one turn are (or are not) available.
AVAILABLE = "available"
UNCONFIRMED = "unconfirmed_semantics"
MISSING_RENDERED = "missing_rendered_count"
MISSING_SERVER = "missing_server_count"
UNRECONCILED = "unreconciled_counts"


def reused_tokens(rendered: int | None, evaluated: int | None) -> int | None:
    """Prompt tokens the server did not evaluate again; ``None`` if either count is unknown."""
    if rendered is None or evaluated is None:
        return None
    return max(0, rendered - evaluated)


def cache_reuse_fraction(rendered: int | None, evaluated: int | None) -> float | None:
    """Reused over rendered; ``None`` for an unknown count or a zero-token prompt."""
    reused = reused_tokens(rendered, evaluated)
    if reused is None or not rendered:
        return None
    return reused / rendered


def prefill_savings_fraction(evaluated: int | None, naive_evaluated: int | None) -> float | None:
    """For a paired row: the share of the naive baseline's evaluated tokens this mode avoided."""
    if evaluated is None or not naive_evaluated:
        return None
    return 1 - evaluated / naive_evaluated


@dataclass(frozen=True)
class CacheDerivation:
    """The cache metrics of one turn and why they are what they are."""

    rendered_prompt_tokens: int | None
    prompt_evaluated_tokens: int | None
    reused_tokens: int | None
    cache_reuse_fraction: float | None
    cache_status: str


def derive_cache(
    rendered: int | None,
    metrics: TurnMetrics | None,
    semantics: Mapping[str, Any],
) -> CacheDerivation:
    """Apply the formulas to a turn's first attempt, or say why they cannot apply."""
    first = metrics.first if metrics is not None else None
    raw = first.raw if first is not None else {}
    new_field = semantics.get("new_tokens_field")
    total_field = semantics.get("total_tokens_field")
    evaluated = raw.get(new_field) if new_field else (first.prompt_evaluated_tokens if first else None)
    evaluated = evaluated if isinstance(evaluated, int) and not isinstance(evaluated, bool) else None

    def unavailable(status: str) -> CacheDerivation:
        return CacheDerivation(rendered, evaluated, None, None, status)

    if not semantics.get("confirmed"):
        return unavailable(UNCONFIRMED)
    if rendered is None:
        return unavailable(MISSING_RENDERED)
    if evaluated is None:
        return unavailable(MISSING_SERVER)
    if evaluated > rendered:
        return unavailable(UNRECONCILED)
    total = raw.get(total_field) if total_field else None
    if total is not None and total != rendered:
        # The server counted a different prompt than the tokenizer did: keep the raw evidence only.
        return unavailable(UNRECONCILED)
    return CacheDerivation(
        rendered, evaluated, reused_tokens(rendered, evaluated), cache_reuse_fraction(rendered, evaluated), AVAILABLE,
    )


def backend_timing_fields(
    metrics: TurnMetrics | None,
    rendered_prompt_tokens: int | None,
    semantics: Mapping[str, Any],
) -> dict[str, Any]:
    """The ``backend_timings`` journal fields (except ``step``) for one turn."""
    cache = derive_cache(rendered_prompt_tokens, metrics, semantics)
    attempts = metrics.attempts if metrics is not None else ()
    return {
        "attempt_metrics": [attempt.counts() for attempt in attempts],
        "attempt_timings": [attempt.timings() for attempt in attempts],
        "rendered_prompt_tokens": rendered_prompt_tokens,
        "prompt_evaluated_tokens": cache.prompt_evaluated_tokens,
        "reused_tokens": cache.reused_tokens,
        "cache_reuse_fraction": cache.cache_reuse_fraction,
        "cache_status": cache.cache_status,
        "ttft_ms": metrics.ttft_ms if metrics is not None else None,
        "prompt_ms": metrics.prompt_ms if metrics is not None else None,
        "predicted_ms": metrics.predicted_ms if metrics is not None else None,
        "response_elapsed_ms": metrics.response_elapsed_ms if metrics is not None else None,
        "slot_id": metrics.slot_id if metrics is not None else None,
        "measurement_path": metrics.measurement_path if metrics is not None else "unavailable",
    }


@dataclass(frozen=True)
class CacheSemantics:
    """What a repeated-prompt probe showed the pinned build's counters to mean."""

    confirmed: bool
    new_tokens_field: str | None
    total_tokens_field: str | None
    evidence: dict[str, Any] = field(default_factory=dict)

    def config(self) -> dict[str, Any]:
        """The ``ContextConfig.cache_semantics`` value (the evidence stays in the probe record)."""
        return {
            "confirmed": self.confirmed,
            "new_tokens_field": self.new_tokens_field,
            "total_tokens_field": self.total_tokens_field,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.config(), "evidence": dict(self.evidence)}


def interpret_cache_probe(cold: Mapping[str, Any], warm: Mapping[str, Any], rendered: int | None) -> CacheSemantics:
    """Decide what ``prompt_n`` and ``tokens_evaluated`` mean from a cold and a warm run of one prompt.

    ``prompt_n`` counts newly evaluated tokens when the cold run evaluated the
    whole rendered prompt and the warm run on the same slot evaluated fewer.
    ``tokens_evaluated`` counts the whole prompt when both runs report the
    rendered length. Anything else leaves the semantics unconfirmed.
    """
    first, second = sanitize_backend_fields(cold), sanitize_backend_fields(warm)
    evidence = {
        "rendered_prompt_tokens": rendered,
        "cold": {name: first.get(name) for name in ("prompt_n", "tokens_evaluated", "tokens_cached", "cache_n", "id_slot")},
        "warm": {name: second.get(name) for name in ("prompt_n", "tokens_evaluated", "tokens_cached", "cache_n", "id_slot")},
    }
    cold_new, warm_new = first.get("prompt_n"), second.get("prompt_n")
    new_field = None
    if rendered is not None and cold_new == rendered and warm_new is not None and warm_new < cold_new:
        new_field = "prompt_n"
    total_field = None
    if rendered is not None and first.get("tokens_evaluated") == rendered == second.get("tokens_evaluated"):
        total_field = "tokens_evaluated"
    if new_field is None:
        evidence["reason"] = "prompt_n did not behave as newly evaluated tokens (cold == rendered, warm smaller)"
    return CacheSemantics(new_field is not None, new_field, total_field, evidence)


def probe_cache_semantics(
    complete: Callable[[str], Mapping[str, Any]],
    prompt: str,
    count_tokens: Callable[[str], int],
) -> CacheSemantics:
    """Run ``prompt`` twice on one slot (``complete`` pins the slot) and interpret the counters.

    A random marker at the start of the prompt makes the first run cold. ``complete``
    is a raw llama-server call (``LlamaCppClient.complete`` with ``id_slot``), not an agent turn.
    """
    marked = f"[cache-probe {uuid.uuid4().hex}]\n{prompt}"
    cold = complete(marked)
    warm = complete(marked)
    return interpret_cache_probe(cold, warm, count_tokens(marked))


def slot_completion(client: Any, *, id_slot: int, max_tokens: int = 1) -> Callable[[str], Mapping[str, Any]]:
    """A raw one-token completion pinned to ``id_slot``, for the cache probe (a truncation is the expected stop)."""

    def complete(prompt: str) -> Mapping[str, Any]:
        try:
            return client.complete(prompt, max_tokens=max_tokens, temperature=0.0, seed=0, id_slot=id_slot)
        except CompletionTruncatedError as exc:
            return exc.response

    return complete


__all__ = [
    "AVAILABLE",
    "CacheDerivation",
    "CacheSemantics",
    "MISSING_RENDERED",
    "MISSING_SERVER",
    "UNCONFIRMED",
    "UNRECONCILED",
    "backend_timing_fields",
    "cache_reuse_fraction",
    "derive_cache",
    "interpret_cache_probe",
    "prefill_savings_fraction",
    "probe_cache_semantics",
    "reused_tokens",
    "slot_completion",
]
