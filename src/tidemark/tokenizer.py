"""The tokenizer protocol, a deterministic CPU fake, and the llama-server parity probe.

Token budgets are only as good as the tokenizer behind them. Live runs count
with the model's real tokenizer, and that tokenizer must reproduce, id for id,
what the pinned llama-server does with the same prompt string: ``/completion``
tokenizes with special-token parsing and ``add_special``, so the probe asks
``/tokenize`` for exactly that. The corpus covers empty and non-empty system
prompts, native tool schemas, generation markers, non-ASCII text, combining
characters, emoji, newlines, strings glued to role and tool markup, and prompts
before and after a planted recitation tail. Text with a lone surrogate is
rejected before it reaches either side.

A parity mismatch blocks token-based acceptance. It never falls back to
character counts. Repeated tokenization is cached by content hash only after
parity has passed (``CachedTokenizer``).

CLI (live)::

    LOCAL_KEY=... python -m tidemark.tokenizer --base-url http://127.0.0.1:8090 \\
        --model Qwen/Qwen3-8B --gguf-sha256 <sha> --llama-cpp-tag b5401 --out parity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from toolsmith.evaluation.models import canonical_hash


class InvalidText(ValueError):
    """Text that is not valid Unicode for UTF-8 (a lone surrogate); never sent to a tokenizer."""


class ParityError(RuntimeError):
    """Token-based acceptance needs a passed parity report, and none was given."""


def check_text(text: str) -> str:
    """``text`` unchanged, or ``InvalidText`` when it cannot be encoded as UTF-8."""
    if not isinstance(text, str):
        raise InvalidText(f"Expected text, got {type(text).__name__}")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidText(f"Text contains an unpaired surrogate at index {exc.start}") from None
    return text


def token_ids_sha256(ids: Sequence[int]) -> str:
    return canonical_hash(list(ids))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Tokenizer(Protocol):
    """What context budgeting needs: exact token ids for a prompt string, and an identity to record."""

    def encode(self, text: str) -> tuple[int, ...]: ...

    def identity(self) -> dict[str, Any]: ...


class RegexTokenizer:
    """A deterministic fake for CPU tests (never for live budgets).

    Pieces are a whole ``<|marker|>``, a run of word characters, a run of
    whitespace, or one other character; each id is a stable hash of its piece.
    """

    PATTERN = re.compile(r"<\|[^|<>]*\|>|\w+|\s+|[^\w\s]", re.UNICODE)

    def __init__(self, name: str = "regex-fake-v1") -> None:
        self.name = name

    def encode(self, text: str) -> tuple[int, ...]:
        check_text(text)
        return tuple(
            int.from_bytes(hashlib.sha256(piece.encode("utf-8")).digest()[:4], "big") % 150_000
            for piece in self.PATTERN.findall(text)
        )

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "kind": "fake", "parity": "not_applicable"}


class HFTokenizer:
    """A Hugging Face tokenizer, encoding without adding special tokens (the chat template already did).

    Fast tokenizers parse special-token text such as ``<|im_start|>`` into its id,
    as llama-server does with special-token parsing on.
    """

    def __init__(self, tokenizer: Any, *, name: str | None = None, revision: str | None = None) -> None:
        self.tokenizer = tokenizer
        self.name = name or getattr(tokenizer, "name_or_path", "unknown")
        self.revision = revision
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, model: str, *, revision: str | None = None, token: str | None = None) -> HFTokenizer:
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(model, revision=revision, token=token), name=model, revision=revision)

    def encode(self, text: str) -> tuple[int, ...]:
        check_text(text)
        with self._lock:
            return tuple(int(i) for i in self.tokenizer.encode(text, add_special_tokens=False))

    def identity(self) -> dict[str, Any]:
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        serialized = backend.to_str() if backend is not None and hasattr(backend, "to_str") else None
        template = getattr(self.tokenizer, "chat_template", None)
        return {
            "name": self.name,
            "kind": "huggingface",
            "revision": self.revision,
            "tokenizer_sha256": text_sha256(serialized) if isinstance(serialized, str) else None,
            "template_sha256": text_sha256(template) if isinstance(template, str) else None,
            "vocab_size": len(self.tokenizer) if hasattr(self.tokenizer, "__len__") else None,
        }


class ServerTokenizer:
    """llama-server's ``/tokenize``, called the way ``/completion`` tokenizes a string prompt."""

    def __init__(
        self, base_url: str, api_key: str, *, add_special: bool = True, timeout: float = 60.0, transport: Any = None,
    ) -> None:
        import httpx

        self.add_special = add_special
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout,
            transport=transport,
        )

    def encode(self, text: str) -> tuple[int, ...]:
        check_text(text)
        response = self._client.post("/tokenize", json={"content": text, "add_special": self.add_special})
        response.raise_for_status()
        tokens = response.json().get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in tokens):
            raise ValueError("llama-server /tokenize returned no integer token list")
        return tuple(tokens)

    def identity(self) -> dict[str, Any]:
        return {"kind": "llama_server", "add_special": self.add_special}

    def close(self) -> None:
        self._client.close()


@dataclass(frozen=True)
class ParityCase:
    case_id: str
    text: str


RECITATION_TAIL = "\n\n[recitation] Open items: confirm the refund amount; keep error code PAY-4471-K verbatim."
# Plain strings: Unicode, combining characters, emoji, newlines, and text glued to markup.
PLAIN_CASES = (
    ParityCase("empty", ""),
    ParityCase("ascii", "Refund 120 INR on order ord_1001 for the dented lid."),
    ParityCase("non_ascii", "Kunde: Jürgen Ölmann — Bestellung №1001, Straße 5, 東京都, ₹120.50"),
    ParityCase("combining", "Café vs Café; ñ; ǟ; Zalgo z̶̧a̷"),
    ParityCase("emoji", "Kettle arrived dented 😞🔧 — family 👨‍👩‍👧 flag 🇮🇳 skin 👍🏽"),
    ParityCase("newlines", "line one\nline two\r\n\n\ttabbed\n\n\n  spaced  \n"),
    ParityCase("markup_adjacent", "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"),
    ParityCase("tool_markup_adjacent", 'x<tool_call>\n{"name": "get_order", "arguments": {"order_id": "ord_1001"}}\n</tool_call>y'),
    ParityCase("think_markup_adjacent", "<think>\n\n</think>\n\nOrder ord_1001 was delivered.<|im_end|>"),
    ParityCase("json_dense", '{"ok":true,"value":{"order_id":"ord_1001","total_inr":"1499.00","items":[]}}'),
)
INVALID_CASES = (ParityCase("lone_surrogate", "abc\ud800def"), ParityCase("reversed_pair", "x\udc00\ud800y"))

_PARITY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order",
        "description": "Look up one order by id.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "An order id such as ord_1001."}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}


def parity_corpus(
    render: Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], str] | None = None,
) -> list[ParityCase]:
    """The plain cases, plus (with ``render``, the model's chat template) whole rendered prompts.

    Rendered prompts cover an empty and a non-empty system prompt, native tool
    schemas, the generation marker every prompt ends with, a tool round trip,
    and the same prompt before and after a planted recitation tail.
    """
    cases = list(PLAIN_CASES)
    if render is None:
        return cases
    user = {"role": "user", "content": "What is the current status of order ord_1001?"}
    call = {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_0_0", "type": "function", "function": {"name": "get_order", "arguments": '{"order_id":"ord_1001"}'},
    }]}
    result = {"role": "tool", "tool_call_id": "call_0_0", "content": '{"ok": true, "value": {"status": "delivered"}}'}
    rendered = {
        "rendered_empty_system": ([{"role": "system", "content": ""}, user], []),
        "rendered_system": ([{"role": "system", "content": "You are a careful support agent. Café ✓"}, user], []),
        "rendered_tools": ([{"role": "system", "content": "Use tools when needed."}, user], [_PARITY_TOOL]),
        "rendered_tool_round_trip": ([{"role": "system", "content": "Use tools."}, user, call, result], [_PARITY_TOOL]),
        "rendered_before_recitation": ([{"role": "system", "content": "Use tools."}, user, call, result], [_PARITY_TOOL]),
        "rendered_after_recitation": (
            [{"role": "system", "content": "Use tools."}, user, call, {**result, "content": result["content"] + RECITATION_TAIL}],
            [_PARITY_TOOL],
        ),
    }
    for case_id, (messages, tools) in rendered.items():
        cases.append(ParityCase(case_id, render(messages, tools)))
    return cases


@dataclass(frozen=True)
class ParityResult:
    case_id: str
    text_sha256: str
    local_count: int
    server_count: int
    local_ids_sha256: str
    server_ids_sha256: str
    match: bool
    first_mismatch: int | None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ParityReport:
    """Every case's counts and id hashes (never the ids or prompts), and whether all matched."""

    results: tuple[ParityResult, ...]
    invalid_rejected: dict[str, bool]
    identity: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.match for r in self.results) and all(self.invalid_rejected.values())

    @property
    def mismatches(self) -> list[str]:
        return [r.case_id for r in self.results if not r.match] + [
            f"{case_id} (not rejected)" for case_id, rejected in self.invalid_rejected.items() if not rejected
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "mismatches": self.mismatches,
            "cases": [result.as_dict() for result in self.results],
            "invalid_rejected": dict(self.invalid_rejected),
            "identity": dict(self.identity),
        }

    def sha256(self) -> str:
        return canonical_hash(self.as_dict())


def compare(case: ParityCase, local: Sequence[int], server: Sequence[int]) -> ParityResult:
    first = next((index for index, (a, b) in enumerate(zip(local, server)) if a != b), None)
    if first is None and len(local) != len(server):
        first = min(len(local), len(server))
    return ParityResult(
        case_id=case.case_id,
        text_sha256=text_sha256(case.text),
        local_count=len(local),
        server_count=len(server),
        local_ids_sha256=token_ids_sha256(local),
        server_ids_sha256=token_ids_sha256(server),
        match=first is None,
        first_mismatch=first,
    )


def run_parity(
    local: Tokenizer,
    server: Tokenizer,
    cases: Sequence[ParityCase],
    *,
    identity: Mapping[str, Any] | None = None,
    invalid: Sequence[ParityCase] = INVALID_CASES,
) -> ParityReport:
    """Compare ``local`` with ``server`` on every case; invalid text must be refused by ``check_text``."""
    results = tuple(compare(case, local.encode(case.text), server.encode(case.text)) for case in cases)
    rejected = {}
    for case in invalid:
        try:
            check_text(case.text)
            rejected[case.case_id] = False
        except InvalidText:
            rejected[case.case_id] = True
    return ParityReport(results, rejected, dict(identity or {}))


class CachedTokenizer:
    """Caches token ids by content hash, and only for a tokenizer whose parity report passed."""

    def __init__(self, inner: Tokenizer, report: ParityReport | None, *, max_entries: int = 4096) -> None:
        if report is None or not report.passed:
            mismatches = report.mismatches if report is not None else ["no parity report"]
            raise ParityError(f"Tokenizer parity has not passed ({mismatches}); token counts cannot be cached or trusted")
        self.inner = inner
        self.report = report
        self.max_entries = max_entries
        self._cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def encode(self, text: str) -> tuple[int, ...]:
        key = text_sha256(check_text(text))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
        ids = self.inner.encode(text)
        with self._lock:
            self.misses += 1
            self._cache[key] = ids
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return ids

    def identity(self) -> dict[str, Any]:
        return {**self.inner.identity(), "parity": "passed", "parity_sha256": self.report.sha256()}


def main(argv: Sequence[str] | None = None) -> int:
    """Live probe: compare the model's HF tokenizer with a running llama-server and write the report."""
    parser = argparse.ArgumentParser(description="Tokenizer parity probe against a running llama-server")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="Hugging Face tokenizer repository, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--gguf-sha256", default=None)
    parser.add_argument("--llama-cpp-tag", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    api_key = os.environ.get("LOCAL_KEY", "").strip()
    if not api_key:
        parser.error("Set LOCAL_KEY in the environment (it is never accepted on the command line)")
    from toolsmith.prompting import ChatPromptRenderer

    local = HFTokenizer.from_pretrained(args.model, revision=args.revision, token=os.environ.get("HF_TOKEN") or None)
    renderer = ChatPromptRenderer(local.tokenizer)
    server = ServerTokenizer(args.base_url, api_key)
    try:
        report = run_parity(
            local, server,
            parity_corpus(lambda messages, tools: renderer.render(messages, tools, enable_thinking=False)),
            identity={
                **local.identity(), "model_repository": args.model, "gguf_sha256": args.gguf_sha256,
                "llama_cpp_tag": args.llama_cpp_tag,
            },
        )
    finally:
        server.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report.passed, "mismatches": report.mismatches, "report": str(args.out)}))
    return 0 if report.passed else 1


__all__ = [
    "CachedTokenizer",
    "HFTokenizer",
    "INVALID_CASES",
    "InvalidText",
    "PLAIN_CASES",
    "ParityCase",
    "ParityError",
    "ParityReport",
    "ParityResult",
    "RECITATION_TAIL",
    "RegexTokenizer",
    "ServerTokenizer",
    "Tokenizer",
    "check_text",
    "compare",
    "main",
    "parity_corpus",
    "run_parity",
    "text_sha256",
    "token_ids_sha256",
]


if __name__ == "__main__":
    sys.exit(main())
