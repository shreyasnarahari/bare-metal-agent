"""Immutable, content-addressed blobs and the catalog that authorizes reading them.

When an observation is shaped into a preview, the complete payload is kept
here so anything the preview omits stays restorable through ``read_blob``.

**Objects.** Bytes live at ``<root>/<namespace>/<first two hex>/<full sha256>``.
They are written to a temporary file in the same directory, flushed and
fsynced, then hard-linked into place (a second writer loses the link and
verifies the existing bytes instead), and the directory is fsynced. An
existing object is always verified before its digest is reused; different
bytes under a digest are a ``BlobIntegrityError``, never overwritten.

**Catalog.** ``<root>/_catalog/catalog.jsonl`` is a strict JSONL index: one
header line naming the catalog id and format version, then one entry per
``(namespace, sha256)`` with a monotonically increasing ``catalog_revision``.
An entry is appended (under an advisory file lock, then fsynced) only after
its object is complete, so a crash can leave an unreferenced object (which
maintenance may collect) but never an entry pointing at partial bytes. A torn
last line is ignored on read and cut before the next append. ``snapshot``
writes an atomic copy for Drive.

**Reads.** ``read`` resolves only a handle cataloged in the caller's namespace,
so knowing a digest from another tenant or session grants nothing. It rejects
malformed handles, non-integer, negative, or out-of-range offsets, excessive
ranges, and symlinked object paths; it never takes a path, URL, glob, SQL, or
``..`` from the caller. Ranges are byte offsets; text is returned only at
UTF-8 boundaries (an offset inside a character is refused, a range end inside
one is moved back and reported). The size is checked on every read and the
digest on every full read and on ``verify``.

**Replay** opens the store with ``read_only=True``: ``put`` refuses, and the
shaper calls ``verify_put`` instead, which requires the recorded catalog entry
and identical bytes. Nothing is regenerated or rewritten.

**Maintenance** (``python -m tidemark.blobs verify|gc``) is explicit: ``gc``
scans the catalog and the journals' references first and removes only
uncataloged objects and stale temporary files, and only with ``--delete``.
A finished episode or a lagging Drive copy never deletes anything.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from steadyhand.store import ArgumentRejected, NotFoundError
from toolsmith.artifacts import copy_file_atomic, copy_snapshot
from toolsmith.json_utils import StrictJSONError, strict_json_loads

BLOB_FORMAT_VERSION = 1
CATALOG_KIND = "tidemark-blob-catalog"
CATALOG_DIR = "_catalog"
CATALOG_FILE = "catalog.jsonl"
LOCK_FILE = "catalog.lock"
HANDLE_PREFIX = "blob://sha256:"
HANDLE_PATTERN = re.compile(r"blob://sha256:([0-9a-f]{64})")
_DIGEST = re.compile(r"[0-9a-f]{64}")
# Namespaces are server-chosen labels (a run id plus a task id, for example), never model input.
NAMESPACE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
DEFAULT_MAX_READ_BYTES = 16384
DEFAULT_MAX_OBJECT_BYTES = 64 * 1024 * 1024
TEXT_ENCODING = "utf-8"
ENCODINGS = (TEXT_ENCODING,)


class BlobError(Exception):
    """Base class for blob store failures."""


class BlobNotFound(NotFoundError, BlobError):
    """No catalog entry for this handle in the caller's namespace (whether or not the bytes exist elsewhere)."""

    def __init__(self, handle: str) -> None:
        super().__init__(
            "blob", handle, field="/handle",
            hint="No stored observation has this handle here. Copy the handle exactly from the observation that named it.",
        )


class BlobRequestError(ArgumentRejected, BlobError):
    """A malformed handle, offset, or range. The caller can fix it."""


class BlobIntegrityError(BlobError):
    """Stored bytes do not match their digest or recorded size, or an object path is not a plain file."""


class BlobReadOnlyError(BlobError):
    """A write was attempted on a store opened read-only (replay)."""


class CatalogFormatError(BlobError):
    """The catalog file is not a well-formed catalog of the current version."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def handle_for(digest: str) -> str:
    if not _DIGEST.fullmatch(digest):
        raise ValueError("a handle needs a full lowercase sha256 hex digest")
    return HANDLE_PREFIX + digest


def parse_handle(handle: Any) -> str:
    """The digest inside a full handle. Short prefixes, paths, and anything else are refused."""
    if not isinstance(handle, str):
        raise BlobRequestError("/handle", "must be a string", handle)
    match = HANDLE_PATTERN.fullmatch(handle)
    if match is None:
        raise BlobRequestError("/handle", "must be blob://sha256: followed by 64 lowercase hex digits", handle)
    return match.group(1)


def check_namespace(namespace: Any) -> str:
    if not isinstance(namespace, str) or not NAMESPACE_PATTERN.fullmatch(namespace):
        raise ValueError(f"invalid blob namespace {namespace!r}")
    return namespace


@dataclass(frozen=True)
class BlobRecord:
    """One catalog entry: an object made readable in one namespace."""

    handle: str
    sha256: str
    media_type: str
    byte_length: int
    source_episode_id: str | None
    source_event_seq: int | None
    source_tool: str | None
    created_seq: int | None
    namespace: str
    encoding: str
    reducer_version: str | None
    retained_until: str | None
    catalog_revision: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BlobRecord:
        names = {item.name for item in fields(cls)}
        if set(value) != names:
            raise CatalogFormatError(f"catalog entry keys differ from {sorted(names)}")
        record = cls(**dict(value))
        if record.handle != handle_for(record.sha256) or record.encoding not in ENCODINGS:
            raise CatalogFormatError("catalog entry handle, digest, or encoding is invalid")
        if isinstance(record.byte_length, bool) or not isinstance(record.byte_length, int) or record.byte_length < 0:
            raise CatalogFormatError("catalog entry byte_length must be a non-negative integer")
        check_namespace(record.namespace)
        return record


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class BlobCatalog:
    """The append-only JSONL index of cataloged objects (see the module docstring)."""

    def __init__(self, directory: Path, *, read_only: bool = False) -> None:
        self.directory = directory
        self.path = directory / CATALOG_FILE
        self.read_only = read_only
        self._lock = threading.RLock()
        self._entries: dict[tuple[str, str], BlobRecord] = {}
        self._revision = 0
        self._offset = 0
        self.catalog_id: str | None = None
        if not self.path.exists():
            if read_only:
                raise CatalogFormatError(f"no blob catalog at {self.path}")
            self._create()
        self._refresh()

    def _create(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        header = {"kind": CATALOG_KIND, "version": BLOB_FORMAT_VERSION, "catalog_id": f"cat_{uuid.uuid4().hex}"}
        temporary = self.directory / f".{CATALOG_FILE}.{uuid.uuid4().hex}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(header, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, self.path)  # exactly one creator wins; the others read its header
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_directory(self.directory)

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        with (self.directory / LOCK_FILE).open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _refresh(self) -> None:
        """Read lines appended since the last read (by this or another process); a torn tail is skipped."""
        with self._lock:
            with self.path.open("rb") as handle:
                handle.seek(self._offset)
                data = handle.read()
            end = data.rfind(b"\n") + 1
            for raw in data[:end].split(b"\n")[:-1]:
                try:
                    value = strict_json_loads(raw.decode("utf-8"))
                except (StrictJSONError, UnicodeDecodeError) as exc:
                    raise CatalogFormatError(f"catalog line is not strict JSON: {exc}") from None
                if not isinstance(value, dict):
                    raise CatalogFormatError("catalog lines must be JSON objects")
                if self.catalog_id is None:
                    if value.get("kind") != CATALOG_KIND or value.get("version") != BLOB_FORMAT_VERSION:
                        raise CatalogFormatError(f"{self.path} is not a version {BLOB_FORMAT_VERSION} blob catalog")
                    self.catalog_id = str(value["catalog_id"])
                    continue
                record = BlobRecord.from_dict(value)
                if record.catalog_revision != self._revision + 1:
                    raise CatalogFormatError(
                        f"catalog revision {record.catalog_revision} follows {self._revision}; revisions must increase by one"
                    )
                key = (record.namespace, record.sha256)
                if key in self._entries:
                    raise CatalogFormatError(f"duplicate catalog entry for {record.handle} in {record.namespace}")
                self._entries[key] = record
                self._revision = record.catalog_revision
            self._offset += end
            if self.catalog_id is None:
                raise CatalogFormatError(f"{self.path} has no header")

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def lookup(self, namespace: str, digest: str) -> BlobRecord | None:
        with self._lock:
            found = self._entries.get((namespace, digest))
            if found is None:
                self._refresh()
                found = self._entries.get((namespace, digest))
            return found

    def entries(self) -> list[BlobRecord]:
        with self._lock:
            self._refresh()
            return sorted(self._entries.values(), key=lambda record: record.catalog_revision)

    def append(self, record: BlobRecord) -> tuple[BlobRecord, bool]:
        """Add ``record`` with the next revision, or return the existing entry for its namespace and digest.

        Returns the stored entry and whether this call added it.
        """
        if self.read_only:
            raise BlobReadOnlyError("the catalog is read-only")
        with self._lock, self._file_lock():
            self._refresh()
            existing = self._entries.get((record.namespace, record.sha256))
            if existing is not None:
                return existing, False
            stored = BlobRecord(**{**record.as_dict(), "catalog_revision": self._revision + 1})
            line = (json.dumps(stored.as_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            with self.path.open("r+b") as handle:
                size = handle.seek(0, os.SEEK_END)
                handle.seek(0)
                data = handle.read()
                valid = data.rfind(b"\n") + 1
                if valid != size:  # a torn tail from a crash: cut it before appending
                    handle.truncate(valid)
                handle.seek(valid)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self._refresh()
            return self._entries[(record.namespace, record.sha256)], True

    def snapshot(self, destination: str | Path) -> Path:
        """An atomic copy ending on a whole line (for Drive)."""
        with self._lock:
            return copy_snapshot(self.path, destination)


@dataclass(frozen=True)
class BlobInfo:
    """What ``put`` returns: the catalog entry and whether this call wrote the object or the entry."""

    record: BlobRecord
    wrote_object: bool
    added_entry: bool


class BlobStore:
    """Objects plus catalog under one local root (see the module docstring). Thread-safe."""

    def __init__(
        self,
        root: str | Path,
        *,
        read_only: bool = False,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
        on_write: Callable[[BlobRecord, Path], None] | None = None,
    ) -> None:
        if max_read_bytes < 1 or max_object_bytes < 1:
            raise ValueError("byte ceilings must be positive")
        self.root = Path(root).resolve()
        self.read_only = read_only
        self.max_read_bytes = max_read_bytes
        self.max_object_bytes = max_object_bytes
        self.on_write = on_write
        if not read_only:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir():
            raise CatalogFormatError(f"no blob store at {self.root}")
        self.catalog = BlobCatalog(self.root / CATALOG_DIR, read_only=read_only)
        self.stats = {"objects_written": 0, "objects_reused": 0, "entries_added": 0, "reads": 0, "bytes_read": 0,
                      "verifications": 0}
        self._stats_lock = threading.Lock()

    @property
    def catalog_id(self) -> str:
        assert self.catalog.catalog_id is not None
        return self.catalog.catalog_id

    def identity(self) -> dict[str, Any]:
        """The store's identity for a context configuration: catalog id and format, never a host path."""
        return {"catalog_id": self.catalog_id, "blob_format_version": BLOB_FORMAT_VERSION}

    def object_path(self, namespace: str, digest: str) -> Path:
        check_namespace(namespace)
        if not _DIGEST.fullmatch(digest):
            raise ValueError("object paths need a full sha256 digest")
        return self.root / namespace / digest[:2] / digest

    def _count(self, name: str, amount: int = 1) -> None:
        with self._stats_lock:
            self.stats[name] += amount

    # Writing

    def put(
        self,
        namespace: str,
        payload: bytes,
        *,
        media_type: str,
        encoding: str = TEXT_ENCODING,
        source_episode_id: str | None = None,
        source_event_seq: int | None = None,
        source_tool: str | None = None,
        created_seq: int | None = None,
        reducer_version: str | None = None,
        retained_until: str | None = None,
    ) -> BlobInfo:
        """Store ``payload`` (if new) and catalog it in ``namespace`` (if not yet). Idempotent per namespace and digest."""
        if self.read_only:
            raise BlobReadOnlyError("this blob store is read-only (replay never writes blobs)")
        check_namespace(namespace)
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes")
        payload = bytes(payload)
        if len(payload) > self.max_object_bytes:
            raise BlobError(f"payload of {len(payload)} bytes exceeds the {self.max_object_bytes}-byte object ceiling")
        if encoding == TEXT_ENCODING:
            payload.decode("utf-8")  # strict: invalid UTF-8 is refused, never replaced and hashed as original
        digest = sha256_hex(payload)
        path = self.object_path(namespace, digest)
        wrote = self._write_object(path, payload, digest)
        self._count("objects_written" if wrote else "objects_reused")
        record = BlobRecord(
            handle=handle_for(digest), sha256=digest, media_type=media_type, byte_length=len(payload),
            source_episode_id=source_episode_id, source_event_seq=source_event_seq, source_tool=source_tool,
            created_seq=created_seq, namespace=namespace, encoding=encoding, reducer_version=reducer_version,
            retained_until=retained_until, catalog_revision=0,
        )
        stored, added = self.catalog.append(record)
        if added:
            self._count("entries_added")
        if self.on_write is not None and (wrote or added):
            self.on_write(stored, path)
        return BlobInfo(stored, wrote, added)

    def _write_object(self, path: Path, payload: bytes, digest: str) -> bool:
        """Write the object atomically; False when identical bytes were already there."""
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)
        self._check_plain(path, allow_missing=True)
        if path.exists():
            self._verify_bytes(path, digest, len(payload))
            return False
        temporary = directory / f".{digest}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:  # another writer finished first: its bytes must be ours
                self._verify_bytes(path, digest, len(payload))
                return False
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_directory(directory)
        return True

    def verify_put(self, namespace: str, payload: bytes, *, media_type: str, encoding: str = TEXT_ENCODING) -> BlobRecord:
        """Replay's stand-in for ``put``: the entry must exist in ``namespace`` and its bytes must equal ``payload``."""
        check_namespace(namespace)
        digest = sha256_hex(bytes(payload))
        record = self.catalog.lookup(namespace, digest)
        if record is None:
            raise BlobNotFound(handle_for(digest))
        if record.byte_length != len(payload) or record.media_type != media_type or record.encoding != encoding:
            raise BlobIntegrityError(f"catalog entry for {record.handle} does not describe the replayed payload")
        self._verify_bytes(self.object_path(namespace, digest), digest, len(payload))
        return record

    # Reading

    def _check_plain(self, path: Path, *, allow_missing: bool = False) -> None:
        """Every component under the root must be a real directory or file, never a symlink."""
        relative = path.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current = current / part
            try:
                status = os.lstat(current)
            except FileNotFoundError:
                if allow_missing:
                    return
                raise BlobIntegrityError(f"blob object is missing: {relative}") from None
            if os.path.islink(current) or (current != path and not os.path.isdir(current)):
                raise BlobIntegrityError(f"blob path component is not a plain directory or file: {relative}")
            if current == path and not (status.st_mode & 0o170000 == 0o100000):
                raise BlobIntegrityError(f"blob object is not a regular file: {relative}")

    def _open(self, path: Path) -> int:
        self._check_plain(path)
        try:
            return os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise BlobIntegrityError(f"cannot open blob object: {exc}") from None

    def _verify_bytes(self, path: Path, digest: str, size: int) -> None:
        self._count("verifications")
        descriptor = self._open(path)
        try:
            if os.fstat(descriptor).st_size != size:
                raise BlobIntegrityError(f"blob {digest[:12]} has {os.fstat(descriptor).st_size} bytes, expected {size}")
            hasher = hashlib.sha256()
            while chunk := os.read(descriptor, 1 << 20):
                hasher.update(chunk)
        finally:
            os.close(descriptor)
        if hasher.hexdigest() != digest:
            raise BlobIntegrityError(f"blob {digest[:12]} bytes do not match their digest")

    def resolve(self, namespace: str, handle: Any) -> BlobRecord:
        """The catalog entry for ``handle`` in ``namespace``, or ``BlobNotFound``."""
        check_namespace(namespace)
        digest = parse_handle(handle)
        record = self.catalog.lookup(namespace, digest)
        if record is None:
            raise BlobNotFound(handle)
        return record

    def verify(self, namespace: str, handle: Any) -> BlobRecord:
        """Full-file verification: catalog entry, plain file, size, and digest."""
        record = self.resolve(namespace, handle)
        self._verify_bytes(self.object_path(namespace, record.sha256), record.sha256, record.byte_length)
        return record

    def read_bytes(self, namespace: str, handle: Any) -> bytes:
        """The whole object, digest-verified (evaluation and maintenance only; the model reads ranges)."""
        record = self.resolve(namespace, handle)
        descriptor = self._open(self.object_path(namespace, record.sha256))
        try:
            data = b""
            while chunk := os.read(descriptor, 1 << 20):
                data += chunk
        finally:
            os.close(descriptor)
        if len(data) != record.byte_length or sha256_hex(data) != record.sha256:
            raise BlobIntegrityError(f"blob {record.sha256[:12]} bytes do not match their catalog entry")
        return data

    def read(self, namespace: str, handle: Any, *, offset: Any = 0, max_bytes: Any = None) -> dict[str, Any]:
        """One bounded range (see the module docstring for the contract)."""
        check_namespace(namespace)
        digest = parse_handle(handle)
        if type(offset) is not int or offset < 0:
            raise BlobRequestError("/offset", "must be a non-negative integer", offset)
        limit = self.max_read_bytes if max_bytes is None else max_bytes
        if type(limit) is not int or not 1 <= limit <= self.max_read_bytes:
            raise BlobRequestError("/max_bytes", f"must be an integer from 1 to {self.max_read_bytes}", max_bytes)
        record = self.catalog.lookup(namespace, digest)
        if record is None:
            raise BlobNotFound(handle)
        total = record.byte_length
        if offset > total or (offset == total and total > 0):
            raise BlobRequestError("/offset", f"is past the end of this blob ({total} bytes)", offset)
        path = self.object_path(namespace, digest)
        descriptor = self._open(path)
        try:
            if os.fstat(descriptor).st_size != total:
                raise BlobIntegrityError(f"blob {digest[:12]} size differs from its catalog entry")
            full = offset == 0 and limit >= total
            if full:
                data = b""
                while chunk := os.read(descriptor, 1 << 20):
                    data += chunk
                if sha256_hex(data) != digest:
                    raise BlobIntegrityError(f"blob {digest[:12]} bytes do not match their digest")
                window = data
                after = b""
            else:
                os.lseek(descriptor, offset, os.SEEK_SET)
                window = os.read(descriptor, min(limit, total - offset))
                after = os.read(descriptor, 1) if offset + len(window) < total else b""
                if offset > 0 and window and 0x80 <= window[0] <= 0xBF:
                    raise BlobRequestError("/offset", "is inside a UTF-8 character; use an offset this tool returned", offset)
        finally:
            os.close(descriptor)
        content, end, adjusted = self._text_window(window, after, offset, record)
        self._count("reads")
        self._count("bytes_read", len(content.encode("utf-8")))
        next_offset = end if end < total else None
        return {
            "handle": record.handle, "sha256": record.sha256, "media_type": record.media_type,
            "encoding": record.encoding, "offset": offset, "end_offset": end, "next_offset": next_offset,
            "total_bytes": total, "truncated": next_offset is not None, "boundary_adjusted": adjusted,
            "content": content,
        }

    @staticmethod
    def _text_window(window: bytes, after: bytes, offset: int, record: BlobRecord) -> tuple[str, int, bool]:
        """Decode ``window``, moving its end back to a character boundary when it cuts one."""
        cut = len(window)
        adjusted = False
        if after and 0x80 <= after[0] <= 0xBF:  # the range ends inside a character
            while cut > 0 and 0x80 <= window[cut - 1] <= 0xBF:
                cut -= 1
            if cut > 0:
                cut -= 1  # drop the lead byte of the cut character too
            adjusted = True
            if cut == 0:
                raise BlobRequestError("/max_bytes", "is too small to return one whole character here", len(window))
        try:
            text = window[:cut].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BlobIntegrityError(f"blob {record.sha256[:12]} is not valid UTF-8 at byte {offset + exc.start}") from None
        return text, offset + cut, adjusted

    def namespace(self, namespace: str) -> NamespacedBlobs:
        """A read view bound to one server-chosen namespace (what a handler receives)."""
        return NamespacedBlobs(self, check_namespace(namespace))

    # Durability and maintenance

    def snapshot_catalog(self, destination: str | Path) -> Path:
        return self.catalog.snapshot(destination)

    def object_files(self) -> Iterator[tuple[str, str, Path]]:
        """Every ``(namespace, digest, path)`` object file on disk (temporary files excluded)."""
        for namespace_dir in sorted(self.root.iterdir()):
            if namespace_dir.name == CATALOG_DIR or not namespace_dir.is_dir() or namespace_dir.is_symlink():
                continue
            for prefix in sorted(namespace_dir.iterdir()):
                if not prefix.is_dir() or prefix.is_symlink():
                    continue
                for path in sorted(prefix.iterdir()):
                    if _DIGEST.fullmatch(path.name):
                        yield namespace_dir.name, path.name, path

    def temporary_files(self) -> Iterator[Path]:
        for path in self.root.rglob(".*.tmp"):
            yield path


class NamespacedBlobs:
    """``BlobStore`` reads bound to one namespace. The handler sees only this, never the root or other tenants."""

    def __init__(self, store: BlobStore, namespace: str) -> None:
        self._store = store
        self.namespace = namespace

    def read(self, handle: Any, *, offset: Any = 0, max_bytes: Any = None) -> dict[str, Any]:
        return self._store.read(self.namespace, handle, offset=offset, max_bytes=max_bytes)

    def verify(self, handle: Any) -> BlobRecord:
        return self._store.verify(self.namespace, handle)

    def read_bytes(self, handle: Any) -> bytes:
        return self._store.read_bytes(self.namespace, handle)

    @property
    def max_read_bytes(self) -> int:
        return self._store.max_read_bytes


# Journal references, checkpoints, and resume checks


def journal_references(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every blob an episode journal references, from its ``observation_shaped`` records, in order."""
    found: list[dict[str, Any]] = []
    for record in records:
        if record.get("event") != "observation_shaped":
            continue
        for part in record.get("blob_parts") or ():
            found.append({"handle": part["handle"], "bytes": part["bytes"], "catalog_revision": record["catalog_revision"],
                          "source_seq": record["source_seq"]})
    return found


def checkpoint(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The blob checkpoint an episode's journal implies: its newest catalog revision and every referenced digest."""
    references = journal_references(records)
    return {
        "catalog_revision": max((ref["catalog_revision"] for ref in references), default=None),
        "digests": sorted({parse_handle(ref["handle"]) for ref in references}),
    }


def verify_references(store: BlobStore, namespace: str, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Refuse (``BlobNotFound``/``BlobIntegrityError``) unless every referenced blob is cataloged and intact.

    Checks the namespace's catalog entry, its revision against the recorded one, the size, and the digest.
    """
    references = journal_references(records)
    checked: set[str] = set()
    for reference in references:
        record = store.resolve(namespace, reference["handle"])
        if record.byte_length != reference["bytes"]:
            raise BlobIntegrityError(f"{reference['handle']} size differs from the journal")
        if record.catalog_revision > reference["catalog_revision"]:
            raise BlobIntegrityError(f"{reference['handle']} was cataloged after the journal says it was")
        if record.sha256 not in checked:
            store.verify(namespace, reference["handle"])
            checked.add(record.sha256)
    return {"references": len(references), "verified_objects": len(checked), "catalog_revision": store.catalog.revision}


def collect_orphans(
    store: BlobStore, journal_paths: Sequence[str | Path] = (), *, delete: bool = False, min_age_s: float = 3600.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Explicit maintenance. Scans the catalog and the journals' references first.

    Only uncataloged objects (left by a crash before cataloging) and temporary
    files older than ``min_age_s`` are candidates, and only ``delete=True``
    removes them. Cataloged objects are never deleted here, referenced or not.
    """
    from steadyhand.journal import read_journal

    if store.read_only and delete:
        raise BlobReadOnlyError("maintenance needs a writable store")
    now = time.time() if now is None else now
    cataloged = {(record.namespace, record.sha256) for record in store.catalog.entries()}
    referenced: set[str] = set()
    for path in journal_paths:
        referenced.update(parse_handle(ref["handle"]) for ref in journal_references(read_journal(path, allow_torn_tail=True)))
    orphans = [path for namespace, digest, path in store.object_files()
               if (namespace, digest) not in cataloged and digest not in referenced and now - path.stat().st_mtime >= min_age_s]
    stale = [path for path in store.temporary_files() if now - path.stat().st_mtime >= min_age_s]
    if delete:
        for path in [*orphans, *stale]:
            path.unlink(missing_ok=True)
    return {
        "cataloged": len(cataloged), "referenced": len(referenced),
        "uncataloged_objects": [str(path.relative_to(store.root)) for path in orphans],
        "stale_temporary_files": [str(path.relative_to(store.root)) for path in stale], "deleted": delete,
    }


def verify_all(store: BlobStore) -> dict[str, Any]:
    """Verify every cataloged object (the full-file verification command)."""
    failures = []
    entries = store.catalog.entries()
    for record in entries:
        try:
            store.verify(record.namespace, record.handle)
        except BlobError as exc:
            failures.append({"namespace": record.namespace, "handle": record.handle, "error": str(exc)})
    return {"entries": len(entries), "failures": failures, "passed": not failures, "catalog_revision": store.catalog.revision}


# Drive mirroring: never on the model's critical path


class BlobMirror:
    """Copies verified local objects and catalog snapshots to durable storage on background threads.

    ``submit`` returns at once. Copies are atomic (``copy_file_atomic``), each
    object at most once, and the catalog snapshot is taken after its objects.
    ``drain`` waits for pending copies (call it at a cell's end, never per step).
    """

    def __init__(self, store: BlobStore, destination: str | Path, *, executor: Executor | None = None,
                 copy: Callable[[Path, Path], Any] = copy_file_atomic) -> None:
        self.store = store
        self.destination = Path(destination)
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="tidemark-blob-mirror")
        self._owns_executor = executor is None
        self._copy = copy
        self._lock = threading.Lock()
        self._done: set[str] = set()
        self._pending: list[Future[Any]] = []
        self.copied = 0
        self.failures: list[str] = []
        self.copy_s = 0.0

    def submit(self, record: BlobRecord, path: Path) -> None:
        key = f"{record.namespace}/{record.sha256}"
        with self._lock:
            if key in self._done:
                self._pending.append(self._executor.submit(self._snapshot))
                return
            self._done.add(key)
            self._pending.append(self._executor.submit(self._object, record, path))

    def _object(self, record: BlobRecord, path: Path) -> None:
        started = time.perf_counter()
        try:
            target = self.destination / record.namespace / record.sha256[:2] / record.sha256
            if not target.exists():
                self._copy(path, target)
            with self._lock:
                self.copied += 1
            self._snapshot()
        except Exception as exc:  # recorded, never raised into an episode
            with self._lock:
                self.failures.append(f"{record.handle}: {exc}")
        finally:
            with self._lock:
                self.copy_s += time.perf_counter() - started

    def _snapshot(self) -> None:
        try:
            self.store.snapshot_catalog(self.destination / CATALOG_DIR / CATALOG_FILE)
        except Exception as exc:
            with self._lock:
                self.failures.append(f"catalog: {exc}")

    def drain(self) -> dict[str, Any]:
        with self._lock:
            pending, self._pending = self._pending, []
        for future in pending:
            future.result()
        return {"copied": self.copied, "failures": list(self.failures), "copy_s": round(self.copy_s, 6)}

    def close(self) -> None:
        self.drain()
        if self._owns_executor:
            self._executor.shutdown(wait=True)


def restore(source: str | Path, root: str | Path) -> BlobStore:
    """Copy a mirrored store (catalog and objects) from durable storage into an empty local root."""
    source, root = Path(source), Path(root)
    if (root / CATALOG_DIR / CATALOG_FILE).exists():
        raise FileExistsError(f"a blob catalog already exists at {root}")
    copy_file_atomic(source / CATALOG_DIR / CATALOG_FILE, root / CATALOG_DIR / CATALOG_FILE)
    mirrored = BlobStore(source, read_only=True)
    for namespace, digest, path in mirrored.object_files():
        copy_file_atomic(path, root / namespace / digest[:2] / digest)
    store = BlobStore(root)
    report = verify_all(store)
    if not report["passed"]:
        raise BlobIntegrityError(f"restored blobs failed verification: {report['failures'][:3]}")
    return store


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tidemark.blobs", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    verify_parser = sub.add_parser("verify", help="verify every cataloged object (size and sha256)")
    verify_parser.add_argument("root")
    gc_parser = sub.add_parser("gc", help="list (or with --delete remove) uncataloged objects and stale temp files")
    gc_parser.add_argument("root")
    gc_parser.add_argument("--journals", nargs="*", default=(), help="journal files whose references to scan first")
    gc_parser.add_argument("--delete", action="store_true")
    gc_parser.add_argument("--min-age-s", type=float, default=3600.0)
    args = parser.parse_args(argv)
    if args.command == "verify":
        report = verify_all(BlobStore(args.root, read_only=True))
    else:
        report = collect_orphans(BlobStore(args.root, read_only=not args.delete), args.journals, delete=args.delete,
                                 min_age_s=args.min_age_s)
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed", True) else 1


__all__ = [
    "BLOB_FORMAT_VERSION",
    "BlobCatalog",
    "BlobError",
    "BlobInfo",
    "BlobIntegrityError",
    "BlobMirror",
    "BlobNotFound",
    "BlobReadOnlyError",
    "BlobRecord",
    "BlobRequestError",
    "BlobStore",
    "CatalogFormatError",
    "DEFAULT_MAX_READ_BYTES",
    "HANDLE_PATTERN",
    "NamespacedBlobs",
    "checkpoint",
    "collect_orphans",
    "handle_for",
    "journal_references",
    "parse_handle",
    "restore",
    "sha256_hex",
    "verify_all",
    "verify_references",
]


if __name__ == "__main__":
    sys.exit(main())
