"""Typed, deterministic reducers that turn one large tool observation into a bounded preview.

A reducer never guesses from a tool's name: ``tidemark.observations`` picks it
from a frozen policy table. Each reducer is pure. ``plan_*`` takes the raw
value and its rule and returns a ``ReductionPlan``: the complete payload(s) to
keep in the blob store (the restorable source of every omitted byte) and an
ordered list of preview candidates, least to most aggressive. ``choose``
measures candidates with the real tokenizer (binary search, then a linear
check) and keeps the first that fits the rule's token cap; if none fits it
keeps the smallest and says so. Byte limits never stand in for token counts.

- ``log_head_tail_v1``: lines split with their endings kept; every line that
  matches an error, exception, assertion, failing-test, or compiler-error
  pattern (tier 1) or a compiler warning or note, a workspace path, or a test
  id (tier 2) is marked, plus the rule's own selectors. Only *consecutive
  exact* duplicates are coalesced (with a count); separated failures that
  share text both stay. A head, a
  tail, and every marked line are kept in original order, with explicit
  omitted line and byte ranges. Under pressure the head and tail shrink, then
  tier 2 marks go, then only the earliest and latest critical lines remain.
  Retained lines are never rewritten; a line too long to show is clipped
  around its first critical match with the omitted byte spans named.
- ``text_head_tail_v1``: the same selection without pattern marks.
- ``exec_streams_v1``: an ``ExecResult`` (exit code, signal, gated stdout and
  stderr). Each stream is reduced separately and stays a separate stream; the
  output gate's ``[BEGIN/END UNTRUSTED OUTPUT]`` wrapper lines are always kept,
  so Tidemark compression never weakens or relabels Bulkhead's boundary. Each
  stream's complete gated text is its own blob.
- ``json_projection_v1``: a per-tool allowlist of JSON Pointers (``*`` for
  array items). Types and array order are preserved; omitted pointers are
  recorded with counts, never values; arrays are then cut to a prefix and long
  strings clipped, each recorded. The blob is canonical JSON (sorted keys,
  finite numbers).
- ``json_bounded_v1``: the default for unknown tools: the same truncation with
  nothing projected away.
- ``html_text_v1``: a standard-library ``HTMLParser`` extracts visible text in
  order (headings, list items, table cells separated by ``|``, ``pre``/``code``
  blocks verbatim, links as ``[text](href)``) and drops script, style,
  template, and navigation-menu content from the preview only. Headings and
  lines with absolute link targets are tier 2. The blob keeps the original page.
  Extraction is not sanitization: the page is still untrusted.

Inline omission markers read ``[tidemark: ...]`` and give the byte offset to
pass to ``read_blob``. For a page, offsets are source byte offsets of the
extracted lines (exact at line starts, approximate inside a clipped line). The envelope's ``context`` member (built by the shaper)
is provenance for the model, not a security label.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

REDUCER_VERSION = "tidemark-reducers-v1"
LOG_HEAD_TAIL = "log_head_tail_v1"
TEXT_HEAD_TAIL = "text_head_tail_v1"
EXEC_STREAMS = "exec_streams_v1"
JSON_PROJECTION = "json_projection_v1"
JSON_BOUNDED = "json_bounded_v1"
HTML_TEXT = "html_text_v1"
NO_REDUCER = "none"
REDUCERS = (LOG_HEAD_TAIL, TEXT_HEAD_TAIL, EXEC_STREAMS, JSON_PROJECTION, JSON_BOUNDED, HTML_TEXT, NO_REDUCER)

TEXT_MEDIA = "text/plain; charset=utf-8"
JSON_MEDIA = "application/json"
HTML_MEDIA = "text/html; charset=utf-8"

# Tier 1: lines that carry the evidence a later step usually needs.
CRITICAL_PATTERNS = (
    r"\b(?:error|errors|ERROR|Error)\b",
    r"\b(?:fatal|FATAL|Fatal)\b",
    r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Failure)\b",
    r"Traceback \(most recent call last\)",
    r"\b(?:assert|assertion|AssertionError)\b",
    r"\b(?:FAILED|FAIL|FAILURE|[Ff]ailed|[Ff]ailures?)\b",
    r"^E\s{2,}\S",
    r"\b(?:panic(?:ked)?|Segmentation fault|core dumped|[Kk]illed|KILL|Aborted)\b",
    r"\bundefined reference\b|\bld returned\b|\bcollect2\b|\bmultiple definition\b",
    r"^[^\s:]+:\d+(?::\d+)?: (?:fatal error|error):",
    r"\bnpm ERR!|\bexit (?:code|status) [1-9]\d*\b",
)
# Tier 2: compiler warnings and notes, workspace-relative paths with a location, and test ids. They go before
# tier 1 under pressure, so a flood of warnings never pushes the one error out of the preview.
PATH_PATTERNS = (
    r"^[^\s:]+:\d+(?::\d+)?: (?:warning|note):",
    r"(?:^|[\s(\"'`=])(?:\./)?(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+:\d+",
    r"\b[\w./-]+\.py::[\w\[\]\-.:]+",
)
# Page headings and lines that carry an absolute link target (tier 2 for pages).
PAGE_PATTERNS = (r"^#{1,6} ", r"\]\(https?://")
DEFAULT_MAX_LINE_CHARS = 800


def _compile(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.MULTILINE) for pattern in patterns)


@dataclass(frozen=True)
class BlobPart:
    """One complete payload kept in the blob store: the source of every byte a preview omits."""

    part: str
    payload: bytes
    media_type: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", hashlib.sha256(self.payload).hexdigest())

    @property
    def handle(self) -> str:
        return f"blob://sha256:{self.digest}"

    @property
    def size(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class Candidate:
    """One preview: the shaped value, extra ``context`` members, and the evidence for the journal."""

    value: Any
    context: dict[str, Any]
    omitted_ranges: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReductionPlan:
    reducer: str
    parts: tuple[BlobPart, ...]
    candidates: tuple[Callable[[], Candidate], ...]


@dataclass(frozen=True)
class Choice:
    candidate: Candidate
    tokens: int
    level: int
    levels: int
    over_cap: bool
    measured: int


def choose(plan: ReductionPlan, measure: Callable[[Candidate], int], cap: int) -> Choice:
    """The least aggressive candidate whose measured tokens fit ``cap`` (binary search, then verified)."""
    built: dict[int, Candidate] = {}
    tokens: dict[int, int] = {}

    def size(index: int) -> int:
        if index not in tokens:
            built[index] = plan.candidates[index]()
            tokens[index] = measure(built[index])
        return tokens[index]

    count = len(plan.candidates)
    low, high = 0, count - 1
    if size(high) > cap:
        return Choice(built[high], tokens[high], high, count, True, len(tokens))
    while low < high:  # the first index that fits, assuming size shrinks with aggressiveness
        middle = (low + high) // 2
        if size(middle) <= cap:
            high = middle
        else:
            low = middle + 1
    index = low
    while size(index) > cap:  # a non-monotone anomaly: walk on to a candidate that really fits
        index += 1
    return Choice(built[index], tokens[index], index, count, False, len(tokens))


# Lines, groups, and selections (logs, text, streams, page text)


def split_lines(text: str) -> list[str]:
    """Lines split on ``\\n`` only, each keeping its ending (the last may have none)."""
    pieces = text.split("\n")
    lines = [piece + "\n" for piece in pieces[:-1]]
    if pieces[-1]:
        lines.append(pieces[-1])
    return lines


@dataclass(frozen=True)
class LineDoc:
    """A text split into lines with byte offsets, coalesced into groups of consecutive exact duplicates."""

    lines: tuple[str, ...]
    offsets: tuple[int, ...]  # byte offset of each line in the blob payload, plus the end
    groups: tuple[tuple[int, int], ...]  # (first line index, repeat count)
    tiers: tuple[int, ...]  # per group: 0 unmarked, 1 critical, 2 path/heading
    forced: frozenset[int]  # group indices always kept (wrapper lines)
    critical: tuple[re.Pattern[str], ...]

    @property
    def total_bytes(self) -> int:
        return self.offsets[-1]


def line_doc(
    text: str,
    *,
    critical: Sequence[re.Pattern[str]] = (),
    secondary: Sequence[re.Pattern[str]] = (),
    forced_lines: Sequence[int] = (),
    offsets: Sequence[int] | None = None,
) -> LineDoc:
    lines = split_lines(text)
    if offsets is None:
        positions = [0]
        for line in lines:
            positions.append(positions[-1] + len(line.encode("utf-8")))
    else:
        positions = list(offsets)
    groups: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        if groups and lines[groups[-1][0]] == line and groups[-1][0] + groups[-1][1] == index:
            groups[-1] = (groups[-1][0], groups[-1][1] + 1)
        else:
            groups.append((index, 1))
    tiers = []
    for start, _count in groups:
        line = lines[start]
        if any(pattern.search(line) for pattern in critical):
            tiers.append(1)
        elif any(pattern.search(line) for pattern in secondary):
            tiers.append(2)
        else:
            tiers.append(0)
    wanted = {len(lines) + index if index < 0 else index for index in forced_lines}
    forced = frozenset(g for g, (start, count) in enumerate(groups) if any(start <= w < start + count for w in wanted))
    return LineDoc(tuple(lines), tuple(positions), tuple(groups), tuple(tiers), forced, tuple(critical))


def _ladder(head: int, tail: int) -> list[tuple[int, int]]:
    steps, current = [], (head, tail)
    while True:
        if current not in steps:
            steps.append(current)
        if current == (0, 0):
            return steps
        current = (current[0] // 2, current[1] // 2) if current[1] > 1 else (0, 0)


def selections(doc: LineDoc, head: int, tail: int) -> list[frozenset[int]]:
    """Group sets to keep, least to most aggressive."""
    count = len(doc.groups)
    marked_all = {g for g, tier in enumerate(doc.tiers) if tier}
    marked_critical = [g for g, tier in enumerate(doc.tiers) if tier == 1]
    result: list[frozenset[int]] = []
    for marks in (marked_all, set(marked_critical)):
        for h, t in _ladder(head, tail):
            keep = set(range(min(h, count))) | set(range(max(0, count - t), count)) | marks | doc.forced
            result.append(frozenset(keep))
    k = (len(marked_critical) + 1) // 2
    while k > 0:
        result.append(frozenset(set(marked_critical[:k]) | set(marked_critical[-k:]) | doc.forced))
        k //= 2
    result.append(frozenset(doc.forced))
    unique: list[frozenset[int]] = []
    for item in result:
        if not unique or unique[-1] != item:
            unique.append(item)
    return unique


def _clip(doc: LineDoc, index: int, max_chars: int, part: str) -> tuple[str, list[dict[str, Any]]]:
    """A retained line, clipped around its first critical match when longer than ``max_chars``."""
    line = doc.lines[index]
    body = line[:-1] if line.endswith("\n") else line
    ending = line[len(body):]
    if len(body) <= max_chars:
        return line, []
    focus = min((m.start() for m in (p.search(body) for p in doc.critical) if m is not None), default=0)
    start = max(0, min(focus - max_chars // 4, len(body) - max_chars))
    end = start + max_chars
    base = doc.offsets[index]
    start_byte = base + len(body[:start].encode("utf-8"))
    end_byte = base + len(body[:end].encode("utf-8"))
    ranges, text = [], ""
    if start > 0:
        ranges.append({"part": part, "lines": [index, index + 1], "bytes": [base, start_byte]})
        text += f"[tidemark: {start_byte - base} bytes of this line omitted; read_blob offset {base}] "
    text += body[start:end]
    line_end = base + len(body.encode("utf-8"))
    if end < len(body):
        ranges.append({"part": part, "lines": [index, index + 1], "bytes": [end_byte, line_end]})
        text += f" [tidemark: {line_end - end_byte} bytes of this line omitted; read_blob offset {end_byte}]"
    return text + ending, ranges


def render_lines(
    doc: LineDoc,
    keep: frozenset[int],
    *,
    part: str,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
    noun: str = "lines",
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """The preview text for one selection, its omitted ranges, and kept/omitted line counts."""
    out: list[str] = []
    ranges: list[dict[str, Any]] = []
    kept_lines = omitted_lines = 0
    gap_start: int | None = None

    def close_gap(end_group: int) -> None:
        nonlocal gap_start, omitted_lines
        if gap_start is None:
            return
        first = doc.groups[gap_start][0]
        last_group = doc.groups[end_group - 1]
        last = last_group[0] + last_group[1]
        start_byte, end_byte = doc.offsets[first], doc.offsets[last]
        omitted_lines += last - first
        ranges.append({"part": part, "lines": [first, last], "bytes": [start_byte, end_byte]})
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        out.append(
            f"[tidemark: omitted {last - first} {noun} ({first + 1}-{last}), bytes [{start_byte}, {end_byte}); "
            f"read_blob offset {start_byte}]\n"
        )
        gap_start = None

    for g, (start, count) in enumerate(doc.groups):
        if g not in keep:
            if gap_start is None:
                gap_start = g
            continue
        close_gap(g)
        text, clipped = _clip(doc, start, max_line_chars, part)
        ranges.extend(clipped)
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        out.append(text)
        kept_lines += count
        if count > 1:
            if not text.endswith("\n"):
                out.append("\n")
            out.append(f"[tidemark: previous line repeated {count - 1} more times ({start + 2}-{start + count})]\n")
    close_gap(len(doc.groups))
    return "".join(out), ranges, {"kept_lines": kept_lines, "omitted_lines": omitted_lines}


def _text_object(preview: str, part: BlobPart, *, original_kind: str = "string", **extra: Any) -> dict[str, Any]:
    """The typed value that replaces an offloaded string."""
    return {"blob": part.handle, "media_type": part.media_type, "original_kind": original_kind, "preview": preview,
            "truncated": True, **extra}


# Text, logs, and exec streams


def _text_rule(rule: Any) -> tuple[int, int, int, tuple[re.Pattern[str], ...], tuple[re.Pattern[str], ...]]:
    critical = _compile(tuple(CRITICAL_PATTERNS) + tuple(rule.must_keep_patterns)) if rule.marks else _compile(rule.must_keep_patterns)
    secondary = _compile(PATH_PATTERNS) if rule.marks else ()
    return rule.head_lines, rule.tail_lines, rule.max_line_chars, critical, secondary


def plan_text(value: Any, rule: Any, *, reducer: str, pointer: str = "") -> ReductionPlan | None:
    """Head/tail (and, for logs, marked lines) of one string at ``pointer`` inside ``value``."""
    target = resolve_pointer(value, pointer)
    if not isinstance(target, str):
        return None
    head, tail, max_chars, critical, secondary = _text_rule(rule)
    part = BlobPart("value" if not pointer else pointer, target.encode("utf-8"), rule.media_type or TEXT_MEDIA)
    doc = line_doc(target, critical=critical, secondary=secondary)
    picks = selections(doc, head, tail)

    def build(keep: frozenset[int], level: int) -> Callable[[], Candidate]:
        def candidate() -> Candidate:
            preview, ranges, counts = render_lines(doc, keep, part=part.part, max_line_chars=max_chars)
            shaped = _text_object(preview, part)
            return Candidate(
                replace_pointer(value, pointer, shaped), {}, tuple(ranges),
                {"level": level, "lines": len(doc.lines), **counts, "marked": sum(1 for t in doc.tiers if t)},
            )
        return candidate

    return ReductionPlan(reducer, (part,), tuple(build(keep, level) for level, keep in enumerate(picks)))


def is_exec_result(value: Any) -> bool:
    return (
        isinstance(value, Mapping) and set(value) == {"exit_code", "signal", "stdout", "stderr"}
        and all(isinstance(value[name], Mapping) and isinstance(value[name].get("content"), str)
                for name in ("stdout", "stderr"))
    )


def plan_exec(value: Any, rule: Any) -> ReductionPlan | None:
    """Each gated stream reduced on its own (wrapper lines kept); exit code and signal pass through."""
    if not is_exec_result(value):
        return None
    head, tail, max_chars, critical, secondary = _text_rule(rule)
    streams = [name for name in ("stderr", "stdout") if value[name]["content"]]
    docs, parts, picks = {}, {}, {}
    for name in streams:
        text = value[name]["content"]
        parts[name] = BlobPart(name, text.encode("utf-8"), TEXT_MEDIA)
        docs[name] = line_doc(text, critical=critical, secondary=secondary, forced_lines=(0, -1))
        picks[name] = selections(docs[name], head, tail)
    levels = max((len(item) for item in picks.values()), default=1)

    def build(level: int) -> Callable[[], Candidate]:
        def candidate() -> Candidate:
            shaped = {key: value[key] for key in ("exit_code", "signal")}
            ranges: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {"level": level}
            for name in ("stdout", "stderr"):
                stream = dict(value[name])
                if name in docs:
                    keep = picks[name][min(level, len(picks[name]) - 1)]
                    preview, stream_ranges, counts = render_lines(docs[name], keep, part=name, max_line_chars=max_chars)
                    ranges.extend(stream_ranges)
                    stream.update(content=preview, blob=parts[name].handle, total_bytes=parts[name].size)
                    metadata[name] = {"lines": len(docs[name].lines), **counts}
                shaped[name] = stream
            return Candidate(shaped, {}, tuple(ranges), metadata)
        return candidate

    return ReductionPlan(EXEC_STREAMS, tuple(parts[name] for name in streams), tuple(build(i) for i in range(levels)))


# JSON


def resolve_pointer(value: Any, pointer: str) -> Any:
    """RFC 6901 lookup; a missing path gives ``None``."""
    if pointer in ("", "/"):
        return value
    current = value
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return None
    return current


def replace_pointer(value: Any, pointer: str, new: Any) -> Any:
    """A copy of ``value`` with the item at ``pointer`` replaced (``""`` replaces the whole value)."""
    if pointer in ("", "/"):
        return new
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer.lstrip("/").split("/")]

    def walk(current: Any, rest: list[str]) -> Any:
        if not rest:
            return new
        head = rest[0]
        if isinstance(current, Mapping):
            return {key: (walk(item, rest[1:]) if key == head else item) for key, item in current.items()}
        if isinstance(current, list):
            return [walk(item, rest[1:]) if str(index) == head else item for index, item in enumerate(current)]
        return current

    return walk(value, tokens)


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def pointer_tree(pointers: Sequence[str]) -> dict[str, Any]:
    """An allowlist tree; ``True`` marks a fully allowed subtree, ``*`` matches every array item."""
    tree: dict[str, Any] = {}
    for pointer in pointers:
        node = tree
        tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer.lstrip("/").split("/")]
        for index, token in enumerate(tokens):
            if index == len(tokens) - 1:
                node[token] = True
            else:
                child = node.get(token)
                if child is True:
                    break
                node = node.setdefault(token, {})
    return tree


def project(value: Any, tree: Any, path: str = "") -> tuple[Any, dict[str, int]]:
    """The allowed part of ``value`` (types and order kept) and the omitted pointers with counts."""
    omitted: dict[str, int] = {}
    if tree is True:
        return value, omitted
    if isinstance(value, Mapping):
        kept = {}
        for key, item in value.items():
            if key in tree:
                kept[key], deeper = project(item, tree[key], f"{path}/{_escape(str(key))}")
                for pointer, count in deeper.items():
                    omitted[pointer] = omitted.get(pointer, 0) + count
            else:
                pointer = f"{path}/{_escape(str(key))}"
                omitted[pointer] = omitted.get(pointer, 0) + 1
        return kept, omitted
    if isinstance(value, list):
        if "*" not in tree:
            omitted[path or "/"] = omitted.get(path or "/", 0) + 1
            return [], omitted
        items = []
        for item in value:
            projected, deeper = project(item, tree["*"], f"{path}/*")
            items.append(projected)
            for pointer, count in deeper.items():
                omitted[pointer] = omitted.get(pointer, 0) + count
        return items, omitted
    return value, omitted


_INDEX = re.compile(r"/\d+(?=/|$)")


def _generic(path: str) -> str:
    return _INDEX.sub("/*", path)


def bound(value: Any, *, items: int, keys: int | None, chars: int, path: str = "",
          notes: list[dict[str, Any]] | None = None, protect: frozenset[str] = frozenset(),
          depth: int | None = None) -> Any:
    """Arrays cut to a prefix, objects to their first keys, long strings clipped; each cut recorded.

    Object members are taken scalars first (ids, statuses, counts), then containers, each group in sorted
    order. Below ``depth`` levels a container keeps its type but not its contents. ``protect`` holds
    pointers (``*`` for array items) that are never clipped or dropped while their parent stays.
    """
    notes = notes if notes is not None else []
    deeper = None if depth is None else depth - 1
    if isinstance(value, str):
        if len(value) <= chars or _generic(path) in protect:
            return value
        notes.append({"pointer": path or "/", "chars": [chars, len(value)]})
        return value[:chars] + f"...[tidemark: {len(value) - chars} chars omitted]"
    if isinstance(value, list):
        if depth is not None and depth <= 0 and value:
            notes.append({"pointer": path or "/", "items": [0, len(value)]})
            return []
        if len(value) > items:
            notes.append({"pointer": path or "/", "items": [items, len(value)]})
        return [bound(item, items=items, keys=keys, chars=chars, path=f"{path}/{index}", notes=notes, protect=protect,
                      depth=deeper) for index, item in enumerate(value[:items])]
    if isinstance(value, Mapping):
        names = sorted(value, key=lambda name: (isinstance(value[name], (Mapping, list)), name))
        if depth is not None and depth <= 0 and names and not any(
            _generic(f"{path}/{_escape(str(name))}") in protect for name in names
        ):
            notes.append({"pointer": path or "/", "count": len(names)})
            return {}
        if keys is not None and len(names) > keys:
            def protected(name: str) -> bool:
                child = _generic(f"{path}/{_escape(str(name))}")
                return child in protect or any(pointer.startswith(child + "/") for pointer in protect)

            chosen = set(names[:keys]) | {name for name in names if protected(name)}
            if len(chosen) < len(names):
                notes.append({"pointer": path or "/", "count": len(names) - len(chosen)})
            names = [name for name in names if name in chosen]
        return {name: bound(value[name], items=items, keys=keys, chars=chars, path=f"{path}/{_escape(str(name))}",
                            notes=notes, protect=protect, depth=deeper) for name in names}
    return value


# (array items, object members, string chars, container depth) per level, least to most aggressive.
JSON_LEVELS = ((None, None, 4000, None), (64, None, 2000, None), (32, None, 1000, 12), (16, 64, 500, 8),
               (8, 32, 240, 6), (4, 16, 120, 4), (2, 8, 60, 3), (1, 4, 40, 2), (0, 4, 24, 1))
MAX_CONTEXT_NOTES = 8


def canonical_payload(value: Any) -> bytes:
    """The blob payload for JSON: sorted keys, finite numbers, two-space indent (deterministic and range-readable)."""
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _bounded_ranges(notes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranges = []
    for note in notes:
        item: dict[str, Any] = {"part": "value", "pointer": note["pointer"]}
        for name in ("items", "chars", "count"):
            if name in note:
                item[name] = note[name]
        ranges.append(item)
    return ranges


def plan_json(value: Any, rule: Any, *, projection: bool) -> ReductionPlan:
    """Projection (when ``projection``) then deterministic truncation; the blob is canonical JSON."""
    part = BlobPart("value", canonical_payload(value), JSON_MEDIA)
    if projection and rule.pointers:
        projected, omitted = project(value, pointer_tree(tuple(rule.pointers) + tuple(rule.required)))
    else:
        projected, omitted = value, {}
    omitted_fields = [{"pointer": pointer, "count": count} for pointer, count in sorted(omitted.items())]
    if isinstance(value, str):  # a bare string has no structure to project: head and tail with a typed object
        text_plan = plan_text(value, rule, reducer=TEXT_HEAD_TAIL)
        assert text_plan is not None
        return text_plan
    protect = frozenset(rule.required)  # required passthrough fields survive every level

    def build(level: int, limits: tuple[int | None, int | None, int, int | None]) -> Callable[[], Candidate]:
        def candidate() -> Candidate:
            items, keys, chars, depth = limits
            notes: list[dict[str, Any]] = []
            shaped = bound(projected, items=items if items is not None else 1 << 30, keys=keys, chars=chars,
                           notes=notes, protect=protect, depth=depth)
            context: dict[str, Any] = {}
            if omitted_fields:
                context["omitted_fields"] = omitted_fields[:MAX_CONTEXT_NOTES]
                if len(omitted_fields) > MAX_CONTEXT_NOTES:
                    context["omitted_field_pointers"] = len(omitted_fields)
            if notes:
                context["cut"] = notes[:MAX_CONTEXT_NOTES]
                if len(notes) > MAX_CONTEXT_NOTES:
                    context["cut_count"] = len(notes)
            ranges = [{"part": "value", "pointer": item["pointer"], "count": item["count"]} for item in omitted_fields]
            ranges += _bounded_ranges(notes)
            return Candidate(shaped, context, tuple(ranges), {"level": level, "omitted_fields": len(omitted_fields),
                                                              "cuts": len(notes)})
        return candidate

    reducer = JSON_PROJECTION if projection and rule.pointers else JSON_BOUNDED
    return ReductionPlan(reducer, (part,), tuple(build(i, limits) for i, limits in enumerate(JSON_LEVELS)))


# HTML


BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "body", "dd", "details", "div", "dl", "dt", "fieldset", "figcaption",
    "figure", "footer", "form", "header", "hr", "li", "main", "nav", "ol", "p", "section", "summary", "table", "tbody",
    "thead", "tfoot", "tr", "ul", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "br", "title", "caption",
})
# Left out of the preview (the page blob keeps them): code, styling, embedded documents, and navigation menus.
SKIPPED_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "head", "iframe", "object", "nav"})


class _PageText(HTMLParser):
    """Visible text in source order, one output line per block, each with its source byte offset."""

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source = source
        # HTMLParser counts lines by "\n" alone, so offsets use the same split.
        self._source_lines = split_lines(source) or [""]
        starts, position = [0], 0
        for line in self._source_lines:
            position += len(line.encode("utf-8"))
            starts.append(position)
        self._byte_starts = starts
        self.lines: list[tuple[str, int]] = []
        self.title = ""
        self._current: list[str] = []
        self._current_offset: int | None = None
        self._skip = 0
        self._head = False
        self._title = False
        self._titled = False
        self._pre = 0
        self._links: list[str | None] = []
        self._cells = 0

    def _offset(self) -> int:
        line, column = self.getpos()
        row = min(max(line - 1, 0), len(self._source_lines) - 1)
        return self._byte_starts[row] + len(self._source_lines[row][:column].encode("utf-8"))

    def _emit(self, text: str) -> None:
        if not text:
            return
        if self._current_offset is None:
            self._current_offset = self._offset()
        self._current.append(text)

    def _flush(self) -> None:
        if self._pre:
            self._flush_pre()
            return
        text = " ".join("".join(self._current).split())
        if text:
            self.lines.append((text, self._current_offset or 0))
        self._current, self._current_offset, self._cells = [], None, 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIPPED_TAGS and not (tag == "head"):
            self._skip += 1
            return
        if tag == "head":
            self._head = True
            return
        if tag == "body":
            self._head = False  # an omitted </head> ends at <body>
        if tag == "title":
            self._title = not self._titled  # only the first title counts
            self._titled = True
            return
        if self._skip or self._head:
            return
        if tag in BLOCK_TAGS:
            self._flush()
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self._emit("#" * int(tag[1]) + " ")
            elif tag == "li":
                self._emit("- ")
            elif tag == "pre":
                self._pre += 1
                self.lines.append(("```", self._offset()))
        elif tag in ("td", "th"):
            if self._cells:
                self._emit(" | ")
            self._cells += 1
        elif tag == "a":
            href = dict(attrs).get("href")
            meaningful = href and not href.startswith(("javascript:", "#")) and href.strip()
            self._links.append(href.strip() if meaningful else None)
            if meaningful:
                self._emit("[")
        elif tag == "img":
            alt = dict(attrs).get("alt")
            if alt:
                self._emit(f"[image: {alt}]")
        elif tag == "code" and not self._pre:
            self._emit("`")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED_TAGS and tag != "head":
            self._skip = max(0, self._skip - 1)
            return
        if tag == "head":
            self._head = False
            return
        if tag == "title":
            self._title = False
            return
        if self._skip or self._head:
            return
        if tag == "a" and self._links:
            href = self._links.pop()
            if href:
                self._emit(f"]({href})")
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag in BLOCK_TAGS:
            self._flush()
            if tag == "pre":
                self._pre = max(0, self._pre - 1)
                self.lines.append(("```", self._offset()))

    def _flush_pre(self) -> None:
        text = "".join(self._current)
        if text.strip("\n"):
            offset = self._current_offset if self._current_offset is not None else self._offset()
            for line in text.strip("\n").split("\n"):
                self.lines.append((line, offset))
        self._current, self._current_offset = [], None

    def handle_data(self, data: str) -> None:
        if self._title:
            self.title += data
            return
        if self._skip or self._head:
            return
        self._emit(data)

    def close(self) -> None:
        super().close()
        self._flush()


def page_text(source: str) -> tuple[str, list[tuple[str, int]]]:
    """The page title and its visible text lines with source byte offsets."""
    parser = _PageText(source)
    parser.feed(source)
    parser.close()
    return " ".join(parser.title.split()), parser.lines


def plan_html(value: Any, rule: Any, *, pointer: str) -> ReductionPlan | None:
    """Visible page text for the preview; the original page (script and style included) is the blob."""
    target = resolve_pointer(value, pointer)
    if not isinstance(target, str):
        return None
    title, lines = page_text(target)
    part = BlobPart("body" if pointer else "value", target.encode("utf-8"), HTML_MEDIA)
    text = "".join(line + "\n" for line, _ in lines)
    offsets = [offset for _, offset in lines] + [part.size]
    critical = _compile(tuple(rule.must_keep_patterns) + (tuple(CRITICAL_PATTERNS) if rule.marks else ()))
    doc = line_doc(text, critical=critical, secondary=_compile(PAGE_PATTERNS), offsets=offsets)
    picks = selections(doc, rule.head_lines, rule.tail_lines)

    def build(keep: frozenset[int], level: int) -> Callable[[], Candidate]:
        def candidate() -> Candidate:
            preview, ranges, counts = render_lines(doc, keep, part=part.part, max_line_chars=rule.max_line_chars,
                                                   noun="page lines")
            shaped = _text_object(preview, part, title=title)
            return Candidate(replace_pointer(value, pointer, shaped), {}, tuple(ranges),
                             {"level": level, "page_lines": len(lines), **counts})
        return candidate

    return ReductionPlan(HTML_TEXT, (part,), tuple(build(keep, level) for level, keep in enumerate(picks)))


def plan_for(reducer: str, value: Any, rule: Any) -> ReductionPlan:
    """The plan for ``reducer``; a value of the wrong shape falls back to bounded JSON, never to a guess."""
    plan: ReductionPlan | None = None
    if reducer in (LOG_HEAD_TAIL, TEXT_HEAD_TAIL):
        plan = plan_text(value, rule, reducer=reducer, pointer=rule.text_pointer)
    elif reducer == EXEC_STREAMS:
        plan = plan_exec(value, rule)
    elif reducer == HTML_TEXT:
        plan = plan_html(value, rule, pointer=rule.text_pointer)
    elif reducer == JSON_PROJECTION:
        plan = plan_json(value, rule, projection=True)
    elif reducer == JSON_BOUNDED:
        plan = plan_json(value, rule, projection=False)
    if plan is None:
        plan = plan_json(value, rule, projection=False)
    return plan


__all__ = [
    "BlobPart", "Candidate", "Choice", "CRITICAL_PATTERNS", "EXEC_STREAMS", "HTML_TEXT", "JSON_BOUNDED",
    "JSON_PROJECTION", "LOG_HEAD_TAIL", "LineDoc", "NO_REDUCER", "PATH_PATTERNS", "REDUCERS", "REDUCER_VERSION",
    "ReductionPlan", "TEXT_HEAD_TAIL", "canonical_payload", "choose", "is_exec_result", "line_doc", "page_text",
    "plan_for", "pointer_tree", "project", "render_lines", "resolve_pointer", "selections", "split_lines",
]
