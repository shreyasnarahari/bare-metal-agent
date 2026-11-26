"""The content-addressed blob store: atomic objects, the catalog, namespaces, bounded reads, and maintenance."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from steadyhand.store import ArgumentRejected, NotFoundError
from tidemark.blobs import (
    CATALOG_DIR,
    CATALOG_FILE,
    BlobIntegrityError,
    BlobMirror,
    BlobNotFound,
    BlobReadOnlyError,
    BlobRequestError,
    BlobStore,
    CatalogFormatError,
    checkpoint,
    collect_orphans,
    handle_for,
    main,
    parse_handle,
    restore,
    sha256_hex,
    verify_all,
    verify_references,
)

TEXT = ("ok line é ✓ 🙂 " * 400).encode("utf-8")


def put(store, namespace="ns", payload=TEXT, **fields):
    return store.put(namespace, payload, media_type="text/plain; charset=utf-8", source_episode_id="ep_1",
                     source_event_seq=4, source_tool="read_log", created_seq=5, reducer_version="v1", **fields)


def test_objects_are_content_addressed_atomic_and_cataloged(tmp_path):
    store = BlobStore(tmp_path)
    info = put(store)
    digest = sha256_hex(TEXT)
    assert info.record.handle == f"blob://sha256:{digest}" and info.wrote_object and info.added_entry
    path = tmp_path / "ns" / digest[:2] / digest
    assert path.read_bytes() == TEXT and not list(path.parent.glob(".*.tmp"))
    record = info.record
    assert (record.byte_length, record.source_event_seq, record.created_seq, record.catalog_revision) == (len(TEXT), 4, 5, 1)
    lines = (tmp_path / CATALOG_DIR / CATALOG_FILE).read_text().splitlines()
    assert json.loads(lines[0])["kind"] == "tidemark-blob-catalog" and json.loads(lines[1])["handle"] == record.handle


def test_duplicate_writes_reuse_the_object_and_entry_and_revisions_only_grow(tmp_path):
    store = BlobStore(tmp_path)
    first = put(store)
    again = put(store)
    assert not again.wrote_object and not again.added_entry and again.record == first.record
    other = put(store, payload=b"second")
    elsewhere = put(store, namespace="other")
    assert other.record.catalog_revision == 2 and elsewhere.record.catalog_revision == 3
    assert not elsewhere.wrote_object or elsewhere.record.namespace == "other"
    assert store.catalog.revision == 3 and store.stats["entries_added"] == 3


def test_concurrent_writers_produce_one_object_and_one_entry(tmp_path):
    store = BlobStore(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        infos = list(pool.map(lambda _: put(store), range(32)))
    assert sum(info.wrote_object for info in infos) == 1 and sum(info.added_entry for info in infos) == 1
    assert {info.record.catalog_revision for info in infos} == {1} and store.catalog.revision == 1


def test_a_second_process_view_sees_new_entries(tmp_path):
    writer, reader = BlobStore(tmp_path), BlobStore(tmp_path, read_only=True)
    info = put(writer)
    assert reader.resolve("ns", info.record.handle) == info.record and reader.catalog_id == writer.catalog_id


@pytest.mark.parametrize("offset,limit", [(0, 1), (0, 7), (3, 64), (17, 999), (0, 16384)])
def test_range_reads_return_text_only_at_utf8_boundaries(tmp_path, offset, limit):
    store = BlobStore(tmp_path)
    handle = put(store).record.handle
    start = offset
    while 0x80 <= TEXT[start] <= 0xBF:  # the reader refuses offsets inside a character; use a real boundary
        start += 1
    try:
        window = store.read("ns", handle, offset=start, max_bytes=limit)
    except BlobRequestError as exc:
        assert exc.field == "/max_bytes"  # a window smaller than the character at the offset
        return
    content = window["content"].encode("utf-8")
    assert TEXT[start:window["end_offset"]] == content and len(content) <= limit
    assert window["offset"] == start and window["total_bytes"] == len(TEXT)
    assert window["next_offset"] == (window["end_offset"] if window["end_offset"] < len(TEXT) else None)
    assert window["truncated"] is (window["next_offset"] is not None)


def test_sequential_ranges_reassemble_the_object_including_the_final_partial_range(tmp_path):
    store = BlobStore(tmp_path)
    handle = put(store).record.handle
    pieces, offset, adjusted = [], 0, 0
    while offset is not None:
        window = store.read("ns", handle, offset=offset, max_bytes=333)
        pieces.append(window["content"])
        adjusted += window["boundary_adjusted"]
        offset = window["next_offset"]
    assert "".join(pieces).encode("utf-8") == TEXT and adjusted > 0


def test_zero_length_objects_read_back_empty(tmp_path):
    store = BlobStore(tmp_path)
    handle = put(store, payload=b"").record.handle
    window = store.read("ns", handle)
    assert window["content"] == "" and window["total_bytes"] == 0 and window["next_offset"] is None
    with pytest.raises(BlobRequestError):
        store.read("ns", handle, offset=1)


@pytest.mark.parametrize("handle", [
    "blob://sha256:" + "a" * 63, "blob://sha256:" + "A" * 64, "sha256:" + "a" * 64, "blob://sha256:../../etc/passwd",
    "blob://evil.example/sha256:" + "a" * 64, "/etc/passwd", "blob://sha256:" + "a" * 64 + "/x", 12, None,
    "blob://sha256:" + "a" * 12,
])
def test_malformed_handles_paths_and_urls_are_refused(tmp_path, handle):
    store = BlobStore(tmp_path)
    put(store)
    with pytest.raises(BlobRequestError) as caught:
        store.read("ns", handle)
    assert caught.value.field == "/handle"


@pytest.mark.parametrize("offset,limit,field", [
    (-1, 10, "/offset"), (1.0, 10, "/offset"), (True, 10, "/offset"), ("0", 10, "/offset"),
    (0, 0, "/max_bytes"), (0, 16385, "/max_bytes"), (0, 2.5, "/max_bytes"), (10**9, 10, "/offset"),
])
def test_offsets_and_ranges_are_bounded_integers(tmp_path, offset, limit, field):
    store = BlobStore(tmp_path)
    handle = put(store).record.handle
    with pytest.raises(BlobRequestError) as caught:
        store.read("ns", handle, offset=offset, max_bytes=limit)
    assert caught.value.field == field and isinstance(caught.value, ArgumentRejected)


def test_an_offset_inside_a_character_is_refused_not_silently_moved(tmp_path):
    store = BlobStore(tmp_path)
    handle = put(store, payload="é".encode("utf-8") * 10).record.handle
    with pytest.raises(BlobRequestError) as caught:
        store.read("ns", handle, offset=1, max_bytes=4)
    assert "inside a UTF-8 character" in caught.value.problem


def test_a_digest_known_from_another_namespace_is_not_authorization(tmp_path):
    store = BlobStore(tmp_path)
    handle = put(store, namespace="tenant-a").record.handle
    with pytest.raises(BlobNotFound) as caught:
        store.read("tenant-b", handle)
    assert isinstance(caught.value, NotFoundError) and caught.value.field == "/handle"
    assert store.namespace("tenant-a").read(handle, max_bytes=10)["content"]
    with pytest.raises(ValueError):
        store.read("../tenant-a", handle)


def test_unknown_handles_are_not_found(tmp_path):
    store = BlobStore(tmp_path)
    with pytest.raises(BlobNotFound):
        store.read("ns", handle_for("0" * 64))


def test_corrupted_truncated_and_symlinked_objects_fail_explicitly(tmp_path):
    store = BlobStore(tmp_path)
    handle = put(store).record.handle
    digest = parse_handle(handle)
    path = tmp_path / "ns" / digest[:2] / digest
    original = path.read_bytes()
    path.write_bytes(original[:-1] + b"X")  # same size, wrong digest
    with pytest.raises(BlobIntegrityError):
        store.read("ns", handle, max_bytes=16384)
    with pytest.raises(BlobIntegrityError):
        store.verify("ns", handle)
    path.write_bytes(original[:100])  # a torn object: the size check catches it on every read
    with pytest.raises(BlobIntegrityError):
        store.read("ns", handle, max_bytes=10)
    path.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(original)
    path.symlink_to(outside)
    with pytest.raises(BlobIntegrityError):
        store.read("ns", handle, max_bytes=10)
    with pytest.raises(BlobIntegrityError):
        put(store)  # an existing digest is verified before reuse, never trusted


def test_a_symlinked_namespace_directory_is_refused(tmp_path):
    store = BlobStore(tmp_path / "root")
    handle = put(store).record.handle
    namespace_dir = tmp_path / "root" / "ns"
    moved = tmp_path / "elsewhere"
    namespace_dir.rename(moved)
    namespace_dir.symlink_to(moved, target_is_directory=True)
    with pytest.raises(BlobIntegrityError):
        store.read("ns", handle, max_bytes=10)


def test_the_catalog_ignores_a_torn_tail_and_cuts_it_before_appending(tmp_path):
    store = BlobStore(tmp_path)
    put(store)
    catalog = tmp_path / CATALOG_DIR / CATALOG_FILE
    with catalog.open("ab") as handle:
        handle.write(b'{"handle": "blob://sha256:')  # a crash mid-append
    reopened = BlobStore(tmp_path)
    assert reopened.catalog.revision == 1
    info = put(reopened, payload=b"after the crash")
    assert info.record.catalog_revision == 2
    assert all(line.endswith("}") for line in catalog.read_text().splitlines())
    assert BlobStore(tmp_path).catalog.revision == 2


def test_a_malformed_catalog_is_refused(tmp_path):
    store = BlobStore(tmp_path)
    put(store)
    catalog = tmp_path / CATALOG_DIR / CATALOG_FILE
    lines = catalog.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["catalog_revision"] = 7
    catalog.write_text("\n".join([lines[0], json.dumps(entry)]) + "\n")
    with pytest.raises(CatalogFormatError):
        BlobStore(tmp_path)


def test_read_only_stores_never_write_and_verify_instead(tmp_path):
    put(BlobStore(tmp_path))
    readonly = BlobStore(tmp_path, read_only=True)
    with pytest.raises(BlobReadOnlyError):
        put(readonly)
    record = readonly.verify_put("ns", TEXT, media_type="text/plain; charset=utf-8")
    assert record.catalog_revision == 1
    with pytest.raises(BlobNotFound):
        readonly.verify_put("ns", b"never written", media_type="text/plain; charset=utf-8")
    with pytest.raises(CatalogFormatError):
        BlobStore(tmp_path / "missing", read_only=True)


def test_invalid_utf8_is_refused_rather_than_replaced(tmp_path):
    with pytest.raises(UnicodeDecodeError):
        put(BlobStore(tmp_path), payload=b"\xff\xfe")


def test_journal_references_checkpoints_and_resume_verification(tmp_path):
    store = BlobStore(tmp_path)
    first, second = put(store).record, put(store, payload=b"two").record
    records = [
        {"event": "tool_result", "seq": 3},
        {"event": "observation_shaped", "seq": 4, "source_seq": 3, "catalog_revision": 1,
         "blob_parts": [{"part": "value", "handle": first.handle, "bytes": first.byte_length}]},
        {"event": "observation_shaped", "seq": 6, "source_seq": 5, "catalog_revision": 2,
         "blob_parts": [{"part": "stderr", "handle": second.handle, "bytes": 3}]},
    ]
    assert checkpoint(records) == {"catalog_revision": 2, "digests": sorted([first.sha256, second.sha256])}
    assert verify_references(store, "ns", records)["verified_objects"] == 2
    with pytest.raises(BlobNotFound):
        verify_references(store, "other", records)
    (tmp_path / "ns" / second.sha256[:2] / second.sha256).unlink()
    with pytest.raises(BlobIntegrityError):
        verify_references(store, "ns", records)


def test_maintenance_collects_only_old_uncataloged_objects_and_only_when_asked(tmp_path):
    store = BlobStore(tmp_path)
    kept = put(store).record
    orphan_digest = sha256_hex(b"orphan")
    orphan = tmp_path / "ns" / orphan_digest[:2] / orphan_digest
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"orphan")  # a crash after the object, before the catalog entry
    temporary = tmp_path / "ns" / ".x.tmp"
    temporary.write_bytes(b"partial")
    young = collect_orphans(store, min_age_s=3600)
    assert young["uncataloged_objects"] == [] and young["stale_temporary_files"] == []
    listed = collect_orphans(store, min_age_s=0, now=time.time() + 1)
    assert listed["uncataloged_objects"] == [str(orphan.relative_to(tmp_path))] and orphan.exists()
    collect_orphans(store, min_age_s=0, now=time.time() + 1, delete=True)
    assert not orphan.exists() and not temporary.exists()
    assert store.verify("ns", kept.handle) == kept  # cataloged objects are never collected
    with pytest.raises(BlobReadOnlyError):
        collect_orphans(BlobStore(tmp_path, read_only=True), delete=True)


def test_verify_all_and_the_cli(tmp_path, capsys):
    store = BlobStore(tmp_path)
    put(store)
    assert verify_all(store)["passed"] and main(["verify", str(tmp_path)]) == 0
    record = store.catalog.entries()[0]
    (tmp_path / "ns" / record.sha256[:2] / record.sha256).write_bytes(b"x" * record.byte_length)
    assert not verify_all(store)["passed"] and main(["verify", str(tmp_path)]) == 1
    assert main(["gc", str(tmp_path)]) == 0 and "uncataloged_objects" in capsys.readouterr().out


def test_the_mirror_copies_off_the_critical_path_and_restores(tmp_path):
    gate = threading.Event()
    copied = []

    def slow_copy(source, destination):
        gate.wait(5)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
        copied.append(destination)

    store = BlobStore(tmp_path / "local")
    mirror = BlobMirror(store, tmp_path / "drive", copy=slow_copy)
    store.on_write = mirror.submit
    started = time.monotonic()
    info = put(store)
    put(store)  # a reuse does not copy again
    assert time.monotonic() - started < 1.0 and not copied  # the write returned before the copy ran
    gate.set()
    report = mirror.drain()
    assert report["copied"] == 1 and report["failures"] == [] and len(copied) == 1
    mirror.close()
    restored = restore(tmp_path / "drive", tmp_path / "restored")
    assert restored.catalog_id == store.catalog_id and restored.read_bytes("ns", info.record.handle) == TEXT
    with pytest.raises(FileExistsError):
        restore(tmp_path / "drive", tmp_path / "restored")


def test_mirror_failures_are_recorded_not_raised(tmp_path):
    def broken(source, destination):
        raise OSError("drive unavailable")

    store = BlobStore(tmp_path / "local")
    mirror = BlobMirror(store, tmp_path / "drive", copy=broken)
    store.on_write = mirror.submit
    put(store)
    report = mirror.drain()
    assert report["copied"] == 0 and "drive unavailable" in report["failures"][0]
    mirror.close()


def test_object_paths_need_full_digests_and_safe_namespaces(tmp_path):
    store = BlobStore(tmp_path)
    with pytest.raises(ValueError):
        store.object_path("ns", "abc")
    for namespace in ("", "_catalog", "a/b", "..", "x" * 97):
        with pytest.raises(ValueError):
            store.object_path(namespace, "0" * 64)
    assert os.path.basename(store.object_path("ns", "0" * 64)) == "0" * 64
