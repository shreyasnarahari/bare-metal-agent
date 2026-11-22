import json
import threading
import time
from pathlib import Path

import pytest

from steadyhand.journal import JournalCopier
from toolsmith.artifacts import MirroredFile, copy_snapshot
from toolsmith.drive import drive_responsive, ensure_drive, sync_drive
from toolsmith.evaluation.models import CaseResult
from toolsmith.evaluation.runner import CheckpointRunner
from toolsmith.recording import trim_partial_line


def result(case_id):
    return CaseResult(case_id, "simple", "model", "prompt", "success")


# Partial lines and snapshots


def test_trim_partial_line_repairs_a_cut_file(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n{"b": 2')
    assert trim_partial_line(path) is True and path.read_text() == '{"a": 1}\n'

    path.write_text('{"a": 1}\n{"b": 2}')  # complete, only the newline is missing
    assert trim_partial_line(path) is True and path.read_text() == '{"a": 1}\n{"b": 2}\n'
    assert trim_partial_line(path) is False and trim_partial_line(tmp_path / "missing.jsonl") is False


def test_copy_snapshot_copies_whole_lines_only(tmp_path):
    source = tmp_path / "growing.jsonl"
    source.write_text('{"a": 1}\n{"b": 2}\n{"c": ')
    copy_snapshot(source, tmp_path / "drive" / "growing.jsonl")
    assert (tmp_path / "drive" / "growing.jsonl").read_text() == '{"a": 1}\n{"b": 2}\n'


# MirroredFile


def test_mirrored_file_restores_throttles_and_forces(tmp_path):
    local, durable = tmp_path / "local" / "c.jsonl", tmp_path / "drive" / "c.jsonl"
    durable.parent.mkdir()
    durable.write_text('{"a": 1}\n')
    store = MirroredFile(local, durable, every_seconds=3600)

    assert store.restore() is True and local.read_text() == '{"a": 1}\n'
    assert store.restore() is False  # local already there
    local.write_text('{"a": 1}\n{"b": 2}\n')
    assert store.sync() is True  # the first sync is never throttled
    local.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n')
    assert store.sync() is False and durable.read_text().count("\n") == 2
    assert store.sync(force=True) is True and durable.read_text().count("\n") == 3
    assert store.sibling(".manifest.json").mirror == tmp_path / "drive" / "c.jsonl.manifest.json"


def test_mirrored_file_runs_throttled_syncs_through_submit_and_records_failures(tmp_path):
    submitted = []
    local = tmp_path / "c.jsonl"
    local.write_text("{}\n")
    store = MirroredFile(local, tmp_path / "drive" / "c.jsonl", submit=lambda name, fn: submitted.append((name, fn)))

    store.sync()
    name, copy = submitted[0]
    assert name == "mirror c.jsonl" and not (tmp_path / "drive" / "c.jsonl").exists()
    assert copy() is True and (tmp_path / "drive" / "c.jsonl").exists()

    blocked = MirroredFile(local, tmp_path / "file-not-dir" / "c.jsonl")
    (tmp_path / "file-not-dir").write_text("not a directory")
    assert blocked.sync(force=True) is False and "c.jsonl" in blocked.errors[0]


# Checkpoints


def test_a_checkpoint_resumes_from_drive_after_a_cut_copy(tmp_path):
    durable = tmp_path / "drive" / "own-qwen3-prompt.jsonl"
    first = CheckpointRunner(MirroredFile(tmp_path / "vm1" / "own.jsonl", durable), "sha", manifest={"m": 1})
    first.run([{"case_id": "a"}, {"case_id": "b"}], lambda case: result(case["case_id"]), workers=1)
    assert durable.read_text().count("\n") == 2
    assert json.loads((tmp_path / "drive" / "own-qwen3-prompt.jsonl.manifest.json").read_text())["m"] == 1
    durable.write_text(durable.read_text() + '{"manifest_sha256": "sha", "res')  # the upload was cut off

    calls = []
    fresh = CheckpointRunner(MirroredFile(tmp_path / "vm2" / "own.jsonl", durable), "sha", manifest={"m": 1})
    done = fresh.run([{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}],
                     lambda case: calls.append(case["case_id"]) or result(case["case_id"]), workers=1)

    assert calls == ["c"] and [r.case_id for r in done] == ["a", "b", "c"]
    assert (tmp_path / "vm2" / "own.jsonl.manifest.json").is_file()  # the manifest came back too
    assert [json.loads(line)["result"]["case_id"] for line in durable.read_text().splitlines()] == ["a", "b", "c"]


# JournalCopier on a slow or broken Drive


def test_the_copier_never_blocks_workers_for_long(tmp_path):
    release = threading.Event()
    copier = JournalCopier(tmp_path / "drive", depth=1, submit_timeout=0.05, close_timeout=10)
    copier.submit_call("stuck", release.wait, 5)  # the copier thread is busy with this
    done = []

    started = time.monotonic()
    for index in range(4):
        copier.submit_call(f"later-{index}", done.append, index)
    assert time.monotonic() - started < 1.0  # parked on the overflow list instead of waiting

    release.set()
    copier.close()
    assert sorted(done) == [0, 1, 2, 3] and copier.errors == []


def test_the_copier_retries_then_records_a_failure(tmp_path):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("Transport endpoint is not connected")

    copier = JournalCopier(tmp_path, retries=3, sleep=lambda seconds: None)
    copier.submit_call("flaky", flaky)
    copier.submit_call("broken", lambda: (_ for _ in ()).throw(OSError("gone")))
    copier.close()

    assert len(attempts) == 3
    assert [str(label) for label, _ in copier.errors] == ["broken"]


def test_closing_a_hung_copier_reports_what_is_left(tmp_path):
    release = threading.Event()
    copier = JournalCopier(tmp_path, depth=4, close_timeout=0.2)
    copier.submit_call("hung", release.wait, 5)
    copier.submit_file(tmp_path / "scores.jsonl", tmp_path / "drive" / "scores.jsonl")

    copier.close()

    assert copier.pending_at_close == [str(tmp_path / "scores.jsonl")]
    assert "still busy" in copier.errors[-1][1]
    release.set()


# Drive mount


class FakeDrive:
    def __init__(self, folder=None):
        self.calls, self.folder = [], folder

    def mount(self, mount_point, force_remount=False):
        self.calls.append(("mount", force_remount))
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)

    def flush_and_unmount(self):
        self.calls.append(("flush_and_unmount",))


def test_drive_probe_answers_for_a_writable_folder_only(tmp_path):
    assert drive_responsive(tmp_path) is True
    assert drive_responsive(tmp_path / "missing") is False


def test_ensure_drive_remounts_only_a_broken_mount(tmp_path):
    healthy = FakeDrive()
    assert ensure_drive(probe_folder=tmp_path, colab_drive=healthy, log=lambda m: None) is False
    assert healthy.calls == []

    folder = tmp_path / "MyDrive"
    broken = FakeDrive(folder)  # the remount brings the folder back
    assert ensure_drive(probe_folder=folder, colab_drive=broken, log=lambda m: None) is True
    assert broken.calls == [("mount", True)]

    with pytest.raises(RuntimeError, match="still does not answer"):
        ensure_drive(probe_folder=tmp_path / "gone", colab_drive=FakeDrive(), log=lambda m: None)


class FakeJobs:
    def __init__(self):
        self.matched = []

    def drain(self, timeout, match=None):
        self.matched = [name for name in ("cache model.gguf to Drive", "mirror own.jsonl", "qwen35-server") if match(name)]
        return {}


def test_sync_drive_drains_drive_writers_flushes_and_remounts(monkeypatch):
    monkeypatch.delenv("TOOLSMITH_DRIVE_SYNC", raising=False)
    drive, jobs = FakeDrive(), FakeJobs()

    sync_drive(jobs, stage="day10-own", colab_drive=drive, log=lambda m: None)

    assert jobs.matched == ["cache model.gguf to Drive", "mirror own.jsonl"]  # not the build
    assert drive.calls == [("flush_and_unmount",), ("mount", False)]
    cleanup = FakeDrive()
    sync_drive(None, stage="cleanup", remount=False, colab_drive=cleanup, log=lambda m: None)
    assert cleanup.calls == [("flush_and_unmount",)]


def test_sync_drive_can_be_limited_to_cleanup(monkeypatch):
    monkeypatch.setenv("TOOLSMITH_DRIVE_SYNC", "cleanup")
    drive = FakeDrive()

    sync_drive(None, stage="day10-own", colab_drive=drive, log=lambda m: None)
    assert drive.calls == []
    sync_drive(None, stage="cleanup", colab_drive=drive, log=lambda m: None)
    assert drive.calls[0] == ("flush_and_unmount",)
