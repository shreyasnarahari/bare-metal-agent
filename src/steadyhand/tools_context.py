"""``read_blob``: the governed range read for observations Tidemark shaped into a preview plus a handle.

When a large tool result arrives, Tidemark keeps a bounded preview in the
conversation and writes the complete payload to an immutable, content-addressed
blob (``tidemark.blobs``). The preview names a handle such as
``blob://sha256:<64 lowercase hex>``; this tool reads a byte range of it.

The model supplies only ``handle``, an optional byte ``offset``, and an
optional ``max_bytes``. It never supplies a path, root, namespace, URL, glob,
or SQL. The server injects the episode's blob namespace through the hidden
``execution_context`` parameter (``steadyhand.executor``), and the store
resolves only handles cataloged in that namespace: a digest known from another
session or tenant is not authorization to read it. Handle syntax and numeric
ceilings are semantic checks (the closed schema stays inside Toolsmith's
grammar subset); an unknown handle is ``not_found``; an offset past the end
or inside a UTF-8 character is ``invalid_arguments``; corrupt bytes are a
handler error, never silently returned.

Policy: a read, idempotent, retried only on transient failures (never on a
malformed or unknown handle), no alternatives, so recovery cannot switch it to
another file-reading tool. Security: no sandbox (a trusted library reads its
own catalog), no sinks, and ``UNTRUSTED`` output: the returned text is the
original tool output, so reading it taints the episode like the first read did.
With Bulkhead enabled the task policy must list ``read_blob`` explicitly.

The tool is opt-in (``build_registry(context_tools=True)``), registered from
session creation so the frozen tool list never changes mid-session. The
default registry's schemas, hashes, and every golden episode are unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

from pydantic import Field

from bulkhead.manifest import SecuritySpec, Trust

from .errors import FieldProblem
from .policy import Effect, ToolPolicy
from .registry import SteadyhandRegistry
from .store import Store
from .tools import SAFE_RETRY, TOOL_TIMEOUT_S, result_schema

BLOB_HANDLE_PATTERN = re.compile(r"blob://sha256:[0-9a-f]{64}")
READ_BLOB_DEFAULT_BYTES = 4096
READ_BLOB_MAX_BYTES = 16384  # the per-call ceiling; a byte limit, never a token budget
READ_BLOB_POLICY = ToolPolicy(Effect.READ, TOOL_TIMEOUT_S, idempotent=True, retry=SAFE_RETRY)
READ_BLOB_SECURITY = SecuritySpec(sandbox=None, output_trust=Trust.UNTRUSTED, sinks=())
CONTEXT_TOOL_NAMES = ("read_blob",)


class BlobRange(TypedDict):
    """One range read: the bytes ``[offset, end_offset)`` as text, and where to continue."""

    handle: str
    sha256: str
    media_type: str
    encoding: str
    offset: int
    end_offset: int
    next_offset: int | None
    total_bytes: int
    truncated: bool
    boundary_adjusted: bool
    content: str


class BlobStoreUnavailable(RuntimeError):
    """The episode runs without a blob store, so no handle can be read."""


def read_blob(
    store: Store,
    handle: Annotated[str, Field(description="A blob handle from an earlier observation: blob://sha256: and 64 hex digits.")],
    offset: Annotated[
        int | None,
        Field(description="Byte offset to start reading at (default 0). To continue, pass next_offset from the last read."),
    ] = None,
    max_bytes: Annotated[
        int | None,
        Field(description=f"Most bytes to return (default {READ_BLOB_DEFAULT_BYTES}, at most {READ_BLOB_MAX_BYTES})."),
    ] = None,
    *,
    execution_context: Any = None,
) -> dict[str, Any]:
    """Read part of an observation that was shortened into a preview, by its blob handle and a byte range."""
    blobs = getattr(execution_context, "blobs", None)
    if blobs is None:
        raise BlobStoreUnavailable("this episode has no blob store")
    return dict(blobs.read(handle, offset=0 if offset is None else offset,
                           max_bytes=READ_BLOB_DEFAULT_BYTES if max_bytes is None else max_bytes))


def check_read_blob(store: Store, arguments: Mapping[str, Any]) -> list[FieldProblem]:
    """Handle syntax and numeric ceilings (semantic, because the grammar subset has no regex or ranges)."""
    problems = []
    handle = arguments.get("handle")
    if not isinstance(handle, str) or not BLOB_HANDLE_PATTERN.fullmatch(handle):
        problems.append(FieldProblem("/handle", "must be blob://sha256: followed by 64 lowercase hex digits", handle))
    offset = arguments.get("offset")
    if offset is not None and (type(offset) is not int or offset < 0):
        problems.append(FieldProblem("/offset", "must be a non-negative integer", offset))
    max_bytes = arguments.get("max_bytes")
    if max_bytes is not None and (type(max_bytes) is not int or not 1 <= max_bytes <= READ_BLOB_MAX_BYTES):
        problems.append(FieldProblem("/max_bytes", f"must be an integer from 1 to {READ_BLOB_MAX_BYTES}", max_bytes))
    return problems


def register_context_tools(registry: SteadyhandRegistry) -> SteadyhandRegistry:
    """Add ``read_blob`` (policy, semantic check, result schema, security spec) to an unfrozen registry."""
    registry.register(read_blob, READ_BLOB_POLICY, check_read_blob, result_schema(BlobRange), security=READ_BLOB_SECURITY)
    return registry


__all__ = [
    "BLOB_HANDLE_PATTERN",
    "BlobRange",
    "BlobStoreUnavailable",
    "CONTEXT_TOOL_NAMES",
    "READ_BLOB_DEFAULT_BYTES",
    "READ_BLOB_MAX_BYTES",
    "READ_BLOB_POLICY",
    "READ_BLOB_SECURITY",
    "check_read_blob",
    "read_blob",
    "register_context_tools",
]
