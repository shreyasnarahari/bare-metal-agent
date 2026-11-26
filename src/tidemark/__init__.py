"""Tidemark: context and memory management for the Steadyhand agent.

Part 1 provides the measurement baseline and the compatibility contract:

- ``tidemark.baseline.NaiveAppendTruncate``: append everything, drop the oldest
  complete turns only when the exact rendered prompt cannot fit the slot.
- ``tidemark.telemetry``: sanitized backend counts and the cache formulas.
- ``tidemark.tokenizer``: the tokenizer protocol and the llama-server parity probe.
- ``tidemark.runtime`` and ``tidemark.scheduler``: the hardware envelope, the
  resource sampler, and the shared slot-leased scheduler.
- ``tidemark.eval``: the long-horizon seed set, the resumable baseline runner,
  T4/L4 capacity plans and autotuning, and the raw-record report.

Nothing is enabled by installing the package. Steadyhand only uses Tidemark when
a caller passes a session as ``run_episode(context=...)``.
"""

__version__ = "0.1.0"

from .types import AttemptMetrics, ContextConfig, SegmentBudget, TurnMetrics

__all__ = ["AttemptMetrics", "ContextConfig", "SegmentBudget", "TurnMetrics", "__version__"]
