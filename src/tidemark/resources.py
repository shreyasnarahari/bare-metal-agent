"""Trusted execution resources a context session hands to the handlers that ask for them.

``steadyhand.executor`` injects ``execution_context`` only into handlers whose
signature names it (today, ``read_blob``), exactly like the server-side
idempotency key: the parameter never appears in a model-facing schema, so the
model cannot supply or replace it. The loop builds one per allowed call, in
call order, from facts the server owns (episode id, step, a read-only journal
view, taint), and the session adds the blob view bound to the episode's
namespace. The handler never sees the store's root, another namespace, or a
filesystem path. ``SecuredExecutor`` forwards the context only after Bulkhead
has allowed the call. Part 6 extends this with memory resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from steadyhand.context import CallEnvironment, JournalView

if TYPE_CHECKING:
    from .blobs import NamespacedBlobs


@dataclass(frozen=True)
class ToolExecutionContext:
    """What a trusted handler may use for one call. Built by the server; nothing here comes from the model."""

    episode_id: str
    step: int
    tool: str
    tool_call_id: str
    journal: JournalView
    namespace: str | None = None
    blobs: NamespacedBlobs | None = None
    taint: str | None = None

    @classmethod
    def for_call(cls, environment: CallEnvironment, *, namespace: str | None = None,
                 blobs: NamespacedBlobs | None = None) -> ToolExecutionContext:
        return cls(
            episode_id=environment.episode_id, step=environment.step, tool=environment.tool,
            tool_call_id=environment.tool_call_id, journal=environment.journal, namespace=namespace, blobs=blobs,
            taint=environment.taint,
        )

    def summary(self) -> dict[str, Any]:
        """A loggable description (no journal content, no paths)."""
        return {"episode_id": self.episode_id, "step": self.step, "tool": self.tool, "namespace": self.namespace,
                "blobs": self.blobs is not None, "taint": self.taint, "journal_records": len(self.journal)}


__all__ = ["CallEnvironment", "JournalView", "ToolExecutionContext"]
