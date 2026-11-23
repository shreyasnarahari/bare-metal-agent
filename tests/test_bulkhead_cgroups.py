"""Bulkhead part 5: delegated cgroup v2 leaves, against an in-memory fake (no real cgroup is touched).

The fake (tests/cgroup_fake.py) behaves like the kernel for the operations
Bulkhead uses and injects the failures the plan names: missing controllers or
files, partial or ignored writes, stale leaves, and kills that leave processes
behind. Real cgroup behavior is covered by the resource integration tests,
which run only inside a confirmed delegated parent.
"""

from __future__ import annotations

import errno
import os

import pytest

from bulkhead.cgroups import CgroupError, CgroupLimits, CgroupParent, configured_parent
from bulkhead.governor import CallGovernor, allocated_bytes
from bulkhead.manifest import Limits

from cgroup_fake import MOUNT, PARENT, FakeCgroupFS

LIMITS = CgroupLimits(memory_bytes=64 * 1024 * 1024, pids=32, cpu_quota_us=50_000)


def parent(fs=None, **options):
    return CgroupParent(PARENT, fs=fs or FakeCgroupFS(**options))


def leaf(fs=None, name="bh-0123456789abcdef"):
    owner = parent(fs)
    return owner, owner.create(name, LIMITS)


# The parent


@pytest.mark.parametrize("path", [MOUNT, "/sys/fs", "relative/path", "/tmp/not-a-cgroup", f"{MOUNT}/../etc"])
def test_the_parent_is_never_the_mount_root_or_outside_it(path):
    with pytest.raises(CgroupError) as raised:
        CgroupParent(path, fs=FakeCgroupFS())
    assert raised.value.code == "bulkhead.cgroup.parent"


def test_missing_controllers_are_refused_and_needed_ones_are_enabled():
    with pytest.raises(CgroupError) as raised:
        parent(controllers=("cpu", "io"))
    assert raised.value.code == "bulkhead.cgroup.controllers" and "memory" in str(raised.value)

    fs = FakeCgroupFS(enabled=("io",))
    CgroupParent(PARENT, fs=fs)
    assert set(fs.read(f"{PARENT}/cgroup.subtree_control").split()) == {"cpu", "io", "memory", "pids"}
    assert ("cgroup.subtree_control", "+cpu +memory +pids") in fs.writes  # only what is needed

    refusing = FakeCgroupFS()
    refusing.subtree_refuses = True
    with pytest.raises(CgroupError) as raised:
        CgroupParent(PARENT, fs=refusing)
    assert raised.value.code == "bulkhead.cgroup.controllers"


def test_the_parent_comes_only_from_configuration(monkeypatch):
    monkeypatch.delenv("BULKHEAD_CGROUP_PARENT", raising=False)
    assert configured_parent(fs=FakeCgroupFS()) is None
    monkeypatch.setenv("BULKHEAD_CGROUP_PARENT", PARENT)
    assert configured_parent(fs=FakeCgroupFS()).path.as_posix() == PARENT


def test_leaf_names_are_unique_and_reveal_nothing():
    names = {CgroupParent.leaf_name("ep_0123", "call_refund_customer_cus_101") for _ in range(50)}
    assert len(names) == 50
    for name in names:
        assert name.startswith("bh-") and len(name) == 19 and "refund" not in name and "ep_" not in name


# Limits


def test_limits_are_written_and_read_back():
    fs = FakeCgroupFS()
    _, created = leaf(fs)

    assert fs.read(created.path / "memory.max") == str(64 * 1024 * 1024)
    assert fs.read(created.path / "pids.max") == "32"
    assert fs.read(created.path / "cpu.max") == "50000 100000"
    assert fs.read(created.path / "memory.swap.max") == "0" and fs.read(created.path / "memory.oom.group") == "1"
    assert created.unavailable == []


def test_limits_follow_the_sandbox_spec():
    limits = CgroupLimits.from_limits(Limits(cpu_s=10, mem_mb=512, pids=64, disk_mb=200, out_kb=64), cpu_fraction=0.5)
    assert limits.as_dict() == {"memory_bytes": 512 * 1024 * 1024, "pids": 64, "cpu_quota_us": 50_000, "cpu_period_us": 100_000}
    with pytest.raises(ValueError):
        CgroupLimits(memory_bytes=1024, pids=1, cpu_quota_us=50_000)


def test_missing_optional_files_are_recorded_not_faked():
    fs = FakeCgroupFS(missing=("memory.swap.max", "memory.peak", "pids.peak"))
    _, created = leaf(fs)

    assert set(created.unavailable) >= {"memory.swap.max"}
    usage = created.usage()
    assert "peak_memory_kb" not in usage and "pids_peak" not in usage
    assert {"memory.peak", "pids.peak"} <= set(created.unavailable)


@pytest.mark.parametrize("name", ["memory.max", "pids.max", "cpu.max"])
def test_a_required_limit_that_does_not_stick_aborts_and_removes_the_leaf(name):
    fs = FakeCgroupFS()
    fs.ignore.add(name)
    owner = parent(fs)
    with pytest.raises(CgroupError) as raised:
        owner.create("bh-0123456789abcdef", LIMITS)
    assert raised.value.code == "bulkhead.cgroup.limit_write"
    assert owner.stale() == []  # the half-made leaf was removed


def test_a_failing_or_partial_write_is_an_error():
    fs = FakeCgroupFS()
    fs.fail["pids.max"] = OSError(errno.EINVAL, "Invalid argument")
    with pytest.raises(CgroupError) as raised:
        parent(fs).create("bh-0123456789abcdef", LIMITS)
    assert raised.value.code == "bulkhead.cgroup.limit_write"
    partial = FakeCgroupFS()
    partial.fail["cpu.max"] = CgroupError("partial write to cpu.max", code="bulkhead.cgroup.partial_write")
    with pytest.raises(CgroupError) as raised:
        parent(partial).create("bh-0123456789abcdef", LIMITS)
    assert raised.value.code == "bulkhead.cgroup.partial_write"


# Admission, kill, and removal


def test_admission_is_verified_against_the_kernels_view():
    fs = FakeCgroupFS()
    _, created = leaf(fs)
    created.admit(4242)
    assert fs.proc_cgroup(4242) == f"0:{created.relative}\n".replace("0:/", "0::/")
    assert created.populated()

    elsewhere = FakeCgroupFS()
    elsewhere.admit_elsewhere = True
    _, other = leaf(elsewhere)
    with pytest.raises(CgroupError) as raised:
        other.admit(4243)
    assert raised.value.code == "bulkhead.cgroup.admit"


def test_destroy_kills_survivors_and_removes_the_leaf():
    fs = FakeCgroupFS()
    owner, created = leaf(fs)
    fs.spawn_into(created.path, 100, 101, 102)  # a child and grandchildren forked inside

    assert created.destroy(sleep=lambda s: None) == "killed"
    assert ("cgroup.kill", "1") in fs.writes and owner.stale() == []
    _, empty = leaf(fs, name="bh-fedcba9876543210")
    assert empty.destroy(sleep=lambda s: None) == "clean"


def test_a_leaf_that_cannot_be_emptied_is_reported_leaked_and_kept():
    fs = FakeCgroupFS()
    owner, created = leaf(fs)
    fs.spawn_into(created.path, 100)
    fs.kill_ineffective = True
    clock = iter(range(0, 1000))

    assert created.wait_empty(0.5, sleep=lambda s: None, clock=lambda: next(clock)) is False
    assert created.destroy(sleep=lambda s: None, clock=lambda: next(clock)) == "leaked"
    assert owner.stale() == [created.name]  # left for reap_stale, never silently dropped


def test_removal_retries_while_busy():
    fs = FakeCgroupFS()
    _, created = leaf(fs)
    fs.spawn_into(created.path, 7)
    assert created.remove(sleep=lambda s: None) is True  # EBUSY -> cgroup.kill -> rmdir


def test_stale_leaves_from_a_crash_are_reaped_and_a_reused_name_starts_fresh():
    fs = FakeCgroupFS()
    owner, created = leaf(fs)
    fs.spawn_into(created.path, 55)
    fs.mkdir(f"{PARENT}/unrelated")  # never ours, never touched

    assert owner.reap_stale() == {created.name: "killed"}
    assert owner.stale() == [] and fs.exists(f"{PARENT}/unrelated")
    _, again = leaf(fs)
    fs.spawn_into(again.path, 56)
    fresh = owner.create(again.name, LIMITS)  # same name: the stale one is destroyed first
    assert not fresh.populated()


def test_a_kernel_without_cgroup_kill_is_an_error():
    fs = FakeCgroupFS(missing=("cgroup.kill",))
    _, created = leaf(fs)
    with pytest.raises(CgroupError) as raised:
        created.kill()
    assert raised.value.code == "bulkhead.cgroup.kill_unavailable"


def test_an_unreadable_events_file_counts_as_populated():
    fs = FakeCgroupFS(missing=("cgroup.events",))
    _, created = leaf(fs)
    assert created.populated() is True


# Usage


def test_usage_is_parsed_and_malformed_values_are_skipped():
    fs = FakeCgroupFS()
    _, created = leaf(fs)
    fs.set(created.path, "cpu.stat", "usage_usec 2500000\nthrottled_usec 400000\nnr_periods banana\n")
    fs.set(created.path, "memory.peak", "73400320\n")
    fs.set(created.path, "memory.events", "low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\n")
    fs.set(created.path, "pids.peak", "32")
    fs.set(created.path, "pids.events", "max 17\n")

    assert created.usage() == {
        "cpu_s": 2.5, "cpu_throttled_s": 0.4, "peak_memory_kb": 71680, "oom_kills": 1, "oom_killed": True,
        "pids_peak": 32, "pids_limit_hits": 17,
    }
    fs.set(created.path, "memory.peak", "not-a-number")
    assert "peak_memory_kb" not in created.usage()


# The governor


def test_the_governor_admits_kills_and_polls_through_the_leaf(tmp_path):
    fs = FakeCgroupFS()
    _, created = leaf(fs)
    now = [0.0]
    governor = CallGovernor(leaf=created, writable=[tmp_path], disk_bytes=10_000, cpu_s=1.0, clock=lambda: now[0])

    assert governor.needs_barrier and governor.tree_kill
    governor.admit(9)
    assert governor.populated() and governor.poll() is None
    fs.set(created.path, "cpu.stat", "usage_usec 1500000\n")
    now[0] += 1
    assert governor.poll() == "cpu"
    (tmp_path / "big").write_bytes(os.urandom(64 * 1024))
    fs.set(created.path, "cpu.stat", "usage_usec 0\n")
    now[0] += 1
    assert governor.poll() == "disk"
    assert governor.usage()["disk_peak_kb"] >= 64
    assert governor.close() == "killed" and not fs.exists(created.path)


def test_without_a_leaf_the_governor_still_watches_disk(tmp_path):
    governor = CallGovernor(writable=[tmp_path], disk_bytes=1 << 30, interval_s=0)
    assert not governor.needs_barrier and not governor.tree_kill
    assert governor.poll() is None and governor.close() == "clean"


def test_allocated_bytes_counts_blocks_without_following_links(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 10_000)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"y" * 10_000)
    (tmp_path / "link").symlink_to("/usr")  # never followed
    assert 20_000 <= allocated_bytes([tmp_path]) < 1_000_000
    assert allocated_bytes([tmp_path], max_entries=2) == float("inf")  # a many-file fill counts as over
