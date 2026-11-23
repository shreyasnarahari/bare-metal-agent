"""Idempotency keys for non-idempotent writes, derived by Steadyhand, never by the model.

The key is ``canonical_hash([episode_id, tool, normalized arguments])``. The
arguments are the validated, decoded ones, normalized like the loop guard
(``120`` equals ``120.0``, outer spaces dropped), so rephrasing the same
call cannot mint a second key. The same call in another episode gets another
key, so two episodes may each make the change once. The key table lives on
the ``Store`` and copies with ``clone()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from toolsmith.evaluation.models import canonical_hash

from .controller import normalize_arguments


def idempotency_key(episode_id: str, tool: str, arguments: Mapping[str, Any]) -> str:
    """``idk_`` plus 32 hex characters of the hash of (episode, tool, normalized arguments)."""
    return "idk_" + canonical_hash([episode_id, tool, normalize_arguments(dict(arguments))])[:32]


__all__ = ["idempotency_key"]
