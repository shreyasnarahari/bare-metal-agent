"""A bounded, thread-safe, content-keyed cache for exact reuse (rendered fragments, token ids, grammars).

A key must hold every input that decides the value: content hashes, the
model/tokenizer/template revision, protocol, thinking flag, frozen schema
hash, and any tool subset. A hit returns the identical object, never a
"similar" one. Hit and miss counters measure CPU overhead saved; they are not
llama.cpp KV-cache hits and never feed ``cache_reuse_fraction``.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class ContentCache:
    """Least-recently-used mapping with ``get``/``put`` (the interface ``ToolsmithService.grammar_cache`` uses)."""

    def __init__(self, name: str, *, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.name = name
        self.max_entries = max_entries
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"name": self.name, "entries": len(self._entries), "hits": self.hits, "misses": self.misses}


__all__ = ["ContentCache"]
