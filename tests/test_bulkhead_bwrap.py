"""Bulkhead part 4: the bubblewrap namespace/seccomp backend and its filesystem view (attacks run in part 7's suite).

Unit tests build argv lists and views without executing anything; mocked
construction tests prove the backend refuses to exist without probe evidence.
None of that is sandbox acceptance. The integration tests at the end run real
bubblewrap and are skipped (reported as unsupported) unless
``BULKHEAD_INTEGRATION=1`` is set on a Linux host whose probe proves every
feature the backend needs.
"""

from __future__ import annotations

import json
import os
import platform
import stat
import sys
from pathlib import Path

import pytest

from bulkhead.backends.base import Program, ProgramKind
from bulkhead.backends.bwrap import (
    BACKEND_NAME,
    REQUIRED_CAPABILITIES,
    SANDBOX_ENV,
    BackendUnavailable,
    BwrapBackend,
    BwrapConfig,
    build_argv,
    classify_setup_failure,
    missing_capabilities,
)
from bulkhead.capabilities import BWRAP_NAMESPACE_FLAGS, ChildRun
from bulkhead.manifest import Egress, Limits, Sandbox, SecuritySpec, Sink, Trust
from bulkhead.supervisor import ProcessOutcome
from bulkhead.types import CAPABILITY_NAMES, Capability, HostInfo, ProbeReport
from bulkhead.workspace import (
    MASKS,
    ViewEntry,
    ViewError,
    WorkRoot,
    Workspace,
    check_host_read,
    ensure_visible,
    plan_view,
)
from steadyhand.tools_exec import EXEC_SANDBOX, EXEC_SECURITY

from exec_support import request_for

ROOT = Path(__file__).parent.parent
PROBE_FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "bulkhead_probe_reports.json").read_text())
LINUX = sys.platform.startswith("linux")
INTEGRATION = os.environ.get("BULKHEAD_INTEGRATION") == "1"
MACHINE = platform.machine().lower()


# A fake host tree for view tests: merged /usr, a system python, and an /etc with secrets.


@pytest.fixture
def host(tmp_path):
    root = tmp_path / "host"
    for directory in ("usr/bin", "usr/lib/python3.12", "usr/lib64", "usr/sbin", "etc/ssl/certs", "home/user/.ssh", "opt/tool"):
        (root / directory).mkdir(parents=True)
    (root / "usr/bin/python3.12").write_text("#!fake\n")
    (root / "usr/bin/python3").symlink_to("python3.12")
    (root / "usr/bin/dash").write_text("#!fake\n")
    (root / "etc/shadow").write_text("root:secret\n")
    for name, target in (("bin", "usr/bin"), ("lib", "usr/lib"), ("lib64", "usr/lib64"), ("sbin", "usr/sbin")):
        (root / name).symlink_to(target)
    (root / "opt/tool/link").symlink_to("/etc/shadow")
    return root


@pytest.fixture
def work(tmp_path):
    root = WorkRoot(tmp_path / "work")
    files = {".env": b"API_KEY=secret\n", ".git/config": b"[core]\n", ".git/hooks/README": b"hooks\n", "README.txt": b"hi\n"}
    return Workspace.create(root, "ep_0123456789abcdef", files)


def view_for(sandbox, workspace, host):
    return plan_view(sandbox, workspace, home="/home/user", host_root=host)


# Work roots and workspaces


def test_the_work_root_is_private_and_holds_inaccessible_masks(tmp_path):
    root = WorkRoot(tmp_path / "work")

    assert stat.S_IMODE(os.stat(root.path).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(root.mask_dir).st_mode) == 0 and stat.S_ISDIR(os.lstat(root.mask_dir).st_mode)
    assert stat.S_IMODE(os.lstat(root.mask_file).st_mode) == 0 and os.lstat(root.mask_file).st_size == 0
    assert WorkRoot(tmp_path / "work").path == root.path  # reopening checks, never recreates


def test_a_shared_or_altered_work_root_is_refused(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    os.chmod(shared, 0o755)
    with pytest.raises(ViewError, match="0700"):
        WorkRoot(shared)
    root = WorkRoot(tmp_path / "work")
    os.chmod(root.mask_file, 0o600)
    with pytest.raises(ViewError, match="mask"):
        WorkRoot(tmp_path / "work")


def test_workspaces_copy_files_in_and_refuse_escapes(tmp_path, work):
    assert stat.S_IMODE(os.stat(work.path).st_mode) == 0o700 and (work.path / "out").is_dir()
    assert (work.path / ".env").read_bytes() == b"API_KEY=secret\n"
    assert os.stat(work.path / ".env").st_nlink == 1  # a copy, never a hardlink
    for bad in ("../escape", "/etc/passwd", "a/../../b", "", "a\nb"):
        with pytest.raises(ViewError):
            work.write(bad, b"x")
    (work.path / "linked").symlink_to(tmp_path)
    with pytest.raises(ViewError, match="symlink"):
        work.write("linked/file", b"x")
    with pytest.raises(FileExistsError):
        Workspace.create(work.root, work.episode_id)
    for bad_id in ("../x", MASKS, "", "a" * 65):
        with pytest.raises(ViewError):
            work.root.episode_path(bad_id)


def test_removing_a_workspace_restores_permissions_the_sandbox_removed(work):
    locked = work.path / "out" / "locked"
    locked.mkdir()
    (locked / "file").write_text("x")
    os.chmod(locked, 0)
    work.remove()
    assert not work.path.exists()


# The view


def test_the_exec_sandbox_view_is_minimal(host, work):
    view = view_for(EXEC_SANDBOX, work, host)

    assert view.entries == (
        ViewEntry("symlink", "/bin", "usr/bin"),
        ViewEntry("symlink", "/lib", "usr/lib"),
        ViewEntry("symlink", "/lib64", "usr/lib64"),
        ViewEntry("symlink", "/sbin", "usr/sbin"),
        ViewEntry("ro", "/usr", str(host / "usr")),
        ViewEntry("ro", "/workspace", str(work.path)),
        ViewEntry("rw", "/workspace/out", str(work.path / "out")),
        ViewEntry("mask_file", "/workspace/.env"),
        ViewEntry("mask_dir", "/workspace/.git"),
    )
    assert view.readable == ("/workspace", "/usr") and view.writable == ("/workspace/out",)
    assert view.visible("/usr/bin/python3.12") and view.visible("/workspace/README.txt")
    for hidden in ("/etc/shadow", "/home/user/.ssh", "/workspace/.git/hooks", "/workspace/.env", "/opt/tool", "/"):
        assert not view.visible(hidden), hidden


@pytest.mark.parametrize("path", ["/", "/home", "/home/user/project", "/etc", "/proc", "/sys/fs", "/dev", "/run/docker.sock", "/tmp/x", "/var/run", "/root", "/boot"])
def test_dangerous_host_paths_are_never_bound(path, host, work):
    with pytest.raises(ViewError) as raised:
        view_for(Sandbox(fs_read=(path,)), work, host)
    assert raised.value.code == "bulkhead.view.forbidden_path"


def test_the_home_directory_and_the_work_root_cannot_be_bound(work):
    with pytest.raises(ViewError, match="home"):
        check_host_read("/srv/alice", home="/srv/alice", work_root=work.root.path)
    with pytest.raises(ViewError):
        check_host_read(str(work.root.path.parent), home="/home/user", work_root=work.root.path)
    check_host_read("/etc/ssl", home="/home/user", work_root=work.root.path)  # a specific /etc subdirectory is fine


def test_writable_paths_must_be_inside_the_workspace(host, work):
    from types import SimpleNamespace

    from bulkhead.errors import SecuritySpecError

    with pytest.raises(SecuritySpecError):  # the spec refuses it first...
        Sandbox(fs_read=("/usr",), fs_write=("/usr/local",))
    unchecked = SimpleNamespace(fs_read=("/usr",), fs_write=("/usr/local",), fs_deny=())
    with pytest.raises(ViewError) as raised:  # ...and the view refuses it again
        view_for(unchecked, work, host)
    assert raised.value.code == "bulkhead.view.write_outside_workspace"


def test_host_symlinks_are_refused_unless_they_are_top_level_links_into_readable_paths(host, work):
    with pytest.raises(ViewError) as raised:
        view_for(Sandbox(fs_read=("/opt/tool/link",)), work, host)
    assert raised.value.code == "bulkhead.view.symlink"
    with pytest.raises(ViewError):
        view_for(Sandbox(fs_read=("/lib",)), work, host)  # /lib -> usr/lib, but /usr is not readable
    (work.path / "out").rmdir()
    (work.path / "out").symlink_to(host / "etc")
    with pytest.raises(ViewError) as raised:
        view_for(EXEC_SANDBOX, work, host)
    assert raised.value.code == "bulkhead.view.symlink"


def test_a_denied_symlink_is_refused_and_absent_reads_are_left_out(host, work):
    (work.path / ".env").unlink()
    (work.path / ".env").symlink_to("/etc/shadow")
    with pytest.raises(ViewError, match="symlink"):
        view_for(EXEC_SANDBOX, work, host)
    (work.path / ".env").unlink()
    view = view_for(Sandbox(fs_read=("$WORKSPACE", "/usr", "/nonexistent")), work, host)
    assert [entry.dest for entry in view.entries if entry.op == "ro"] == ["/usr", "/workspace"]


def test_a_denied_path_that_could_be_created_is_occupied(host, work):
    sandbox = Sandbox(fs_read=("$WORKSPACE",), fs_write=("$WORKSPACE/out",), fs_deny=("$WORKSPACE/out/secret", "$WORKSPACE/out/secret/inner"))
    view = view_for(sandbox, work, host)
    assert [entry for entry in view.entries if entry.op.startswith("mask")] == [ViewEntry("mask_dir", "/workspace/out/secret")]


def test_writable_directories_are_created_privately(host, work):
    view = view_for(Sandbox(fs_read=("$WORKSPACE",), fs_write=("$WORKSPACE/out/new/deeper",)), work, host)
    created = work.path / "out" / "new" / "deeper"
    assert created.is_dir() and stat.S_IMODE(os.stat(created).st_mode) == 0o700
    assert view.writable == ("/workspace/out/new/deeper",)


def test_interpreters_must_be_visible(host, work):
    view = view_for(EXEC_SANDBOX, work, host)
    assert ensure_visible(view, "/usr/bin/python3", host_root=host) == "/usr/bin/python3.12"
    assert ensure_visible(view, "/bin/dash", host_root=host) == "/usr/bin/dash"
    (host / "opt/python").write_text("#!fake\n")
    with pytest.raises(ViewError) as raised:
        ensure_visible(view, "/opt/python", host_root=host)
    assert raised.value.code == "bulkhead.view.interpreter"


# The argv


def argv_for(view, program, *, tmpfs_size=False):
    config = BwrapConfig("/usr/bin/bwrap", tmpfs_size=tmpfs_size)
    return build_argv(
        config, view, program, interpreter="/usr/bin/python3.12", shell="/usr/bin/dash",
        seccomp_fd=11, status_fd=12, tmp_bytes=200 * 1024 * 1024, mask_dir="/w/.bulkhead-masks/dir", mask_file="/w/.bulkhead-masks/file",
    )


def test_the_argv_is_fixed_and_builds_every_isolation_layer(host, work):
    program = Program.from_text(ProgramKind.PYTHON, "print('do-not-log-me')")
    argv = argv_for(view_for(EXEC_SANDBOX, work, host), program)

    assert argv[0] == "/usr/bin/bwrap"
    assert argv[1:1 + len(BWRAP_NAMESPACE_FLAGS)] == list(BWRAP_NAMESPACE_FLAGS)
    for flag in ("--unshare-user", "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts", "--die-with-parent", "--new-session"):
        assert flag in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    text = " ".join(argv)
    assert "--hostname bulkhead --proc /proc --dev /dev --tmpfs /tmp" in text
    assert f"--ro-bind {host / 'usr'} /usr" in text and f"--bind {work.path / 'out'} /workspace/out" in text
    assert "--ro-bind /w/.bulkhead-masks/dir /workspace/.git" in text and "--ro-bind /w/.bulkhead-masks/file /workspace/.env" in text
    assert "--symlink usr/bin /bin" in text
    assert "--ro-bind / /" not in text and "/etc" not in text and "/home" not in text
    assert argv[argv.index("--chdir") + 1] == "/workspace"
    assert argv[argv.index("--seccomp") + 1] == "11"
    tail = argv[argv.index("--") + 1:]
    assert tail[:4] == ["/usr/bin/python3.12", "-I", "-S", "-c"] and tail[5:] == ["12", "python", "/usr/bin/python3.12", "-I", "-"]
    assert not any("do-not-log-me" in item for item in argv)
    # Masks come after the binds they cover, and nested binds after their parents.
    assert argv.index("/workspace/.git") > argv.index("/workspace/out") > argv.index("/workspace")


def test_tmp_is_bounded_only_when_this_bubblewrap_supports_it(host, work):
    program = Program.from_text(ProgramKind.SHELL, "true")
    view = view_for(EXEC_SANDBOX, work, host)
    assert "--size" not in argv_for(view, program)
    sized = argv_for(view, program, tmpfs_size=True)
    assert sized[sized.index("--size") + 1] == str(200 * 1024 * 1024) and sized[sized.index("--size") + 2] == "--tmpfs"
    assert sized[-4:] == ["12", "shell", "/usr/bin/dash", "-c"]


def test_a_view_without_the_workspace_starts_in_tmp(host, work):
    argv = argv_for(view_for(Sandbox(fs_read=("/usr",)), work, host), Program.from_text("python", "pass"))
    assert argv[argv.index("--chdir") + 1] == "/tmp"


def test_the_sandbox_environment_is_minimal_and_secret_free():
    assert dict(SANDBOX_ENV) == {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/workspace", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TMPDIR": "/tmp"}


@pytest.mark.parametrize(
    ("stderr", "timed_out", "code"),
    [
        (b"bwrap: Unknown option --size\n", False, "bulkhead.setup.unsupported_option"),
        (b"bwrap: No permissions to creating new namespace, likely because the kernel does not allow non-privileged user namespaces.\n", False, "bulkhead.setup.userns_denied"),
        (b"bwrap: Can't bind mount /x on /newroot/x: No such file or directory\n", False, "bulkhead.setup.mount"),
        (b"bwrap: execvp /usr/bin/python3.12: No such file or directory\n", False, "bulkhead.setup.exec"),
        (b"bwrap: prctl(PR_SET_SECCOMP): Invalid argument\n", False, "bulkhead.setup.seccomp"),
        (b"", True, "bulkhead.setup.timeout"),
        (b"something else\n", False, "bulkhead.setup.failed"),
    ],
)
def test_setup_failures_get_stable_codes_and_bubblewraps_text_is_not_recorded(stderr, timed_out, code):
    observed = ProcessOutcome(False, None, None, timed_out, b"", stderr, 0, len(stderr), False, False, 0.1, "not_started", {}, "bulkhead.setup.failed")
    assert classify_setup_failure(observed) == code


# Construction: only with probe evidence, never a fallback


def fake_report(*, failing=(), architecture=MACHINE, bwrap_version="0.9.0"):
    facts = PROBE_FIXTURE["baseline_facts"]
    records = []
    for name in CAPABILITY_NAMES:
        if name in failing:
            records.append(Capability(name, "unavailable", "not_found", "fake failure", {}, 0.01))
        else:
            fact = dict(facts[name])
            if name == "bwrap.binary":
                fact["version"] = bwrap_version
            records.append(Capability(name, "ok", "ok", "fake", fact, 0.01))
    host = HostInfo("Linux", "6.8.0-test", architecture, "3.12.0", wsl=False, container=False, root=False)
    return ProbeReport.build(
        created_at="2026-11-09T12:00:00Z", host=host, python="/usr/bin/python3", helper="/opt/bulkhead/_probe_child.py",
        bwrap_path="/usr/bin/bwrap", capabilities=records, cgroup_mount=None, cgroup_parent=None,
    )


def fake_runner(version="0.9.0", help_text="--size BYTES  Set size of next tmpfs\n"):
    calls = []

    def runner(argv, *, timeout_s, env=None):
        calls.append(list(argv))
        if argv[1] == "--version":
            return ChildRun(0, f"bubblewrap {version}\n", "", False, 0.01)
        return ChildRun(0, help_text, "", False, 0.01)

    runner.calls = calls
    return runner


@pytest.mark.parametrize("missing", REQUIRED_CAPABILITIES)
def test_the_backend_refuses_to_exist_without_each_required_capability(tmp_path, missing):
    report = fake_report(failing=(missing,))
    assert missing in missing_capabilities(report)
    with pytest.raises(BackendUnavailable) as raised:
        BwrapBackend(report, tmp_path / "work", runner=fake_runner(), namespace_phase=True)
    assert missing in raised.value.missing
    assert not (tmp_path / "work").exists()  # nothing was set up


def test_a_stale_or_foreign_probe_report_is_refused(tmp_path):
    with pytest.raises(BackendUnavailable, match="architecture"):
        BwrapBackend(fake_report(architecture="s390x"), tmp_path / "w", runner=fake_runner(), namespace_phase=True)
    with pytest.raises(BackendUnavailable, match="different bubblewrap"):
        BwrapBackend(fake_report(bwrap_version="0.8.0"), tmp_path / "w", runner=fake_runner("0.9.0"), namespace_phase=True)
    with pytest.raises(BackendUnavailable, match="older"):
        BwrapBackend(fake_report(bwrap_version="0.3.3"), tmp_path / "w", runner=fake_runner("0.3.3"), namespace_phase=True)


needs_libseccomp = pytest.mark.skipif(not LINUX or MACHINE not in ("x86_64", "aarch64"), reason="unsupported: needs Linux x86_64/aarch64")


@needs_libseccomp
def test_a_proven_host_gets_a_backend_labeled_as_the_namespace_phase(tmp_path):
    runner = fake_runner()
    backend = BwrapBackend(fake_report(), tmp_path / "work", runner=runner, namespace_phase=True)

    assert backend.identity.name == BACKEND_NAME == "bwrap-namespace-phase"
    assert backend.identity.tier.value == "T2" and backend.identity.version == "5.0+bwrap.0.9.0"
    assert backend.cgroup_parent is None
    assert backend.identity.probe_sha256 == fake_report().probe_sha256
    assert backend.config.tmpfs_size and backend.config.bwrap == "/usr/bin/bwrap"
    assert runner.calls == [["/usr/bin/bwrap", "--version"], ["/usr/bin/bwrap", "--help"]]  # no sandbox was run
    assert not BwrapBackend(fake_report(), tmp_path / "w2", runner=fake_runner(help_text="no size"), namespace_phase=True).config.tmpfs_size


@needs_libseccomp
def test_full_t2_needs_every_t2_capability_and_a_cgroup_parent(tmp_path):
    from bulkhead.backends.bwrap import BACKEND_NAME_T2
    from bulkhead.cgroups import CgroupParent

    from cgroup_fake import PARENT, FakeCgroupFS

    with pytest.raises(BackendUnavailable) as raised:
        BwrapBackend(fake_report(), tmp_path / "w", runner=fake_runner())
    assert raised.value.missing == ("cgroup.parent",)
    parent = CgroupParent(PARENT, fs=FakeCgroupFS())
    for capability in ("cgroup.delegation", "cgroup.kill", "rlimits"):
        with pytest.raises(BackendUnavailable) as raised:
            BwrapBackend(fake_report(failing=(capability,)), tmp_path / "w", runner=fake_runner(), cgroup_parent=parent)
        assert capability in raised.value.missing
    backend = BwrapBackend(fake_report(), tmp_path / "w", runner=fake_runner(), cgroup_parent=parent)
    assert backend.identity.name == BACKEND_NAME_T2 == "bwrap-t2" and backend.cgroup_parent is parent


@needs_libseccomp
def test_requests_it_cannot_serve_are_refused_before_anything_spawns(tmp_path, monkeypatch):
    import bulkhead.backends.bwrap as module

    def no_spawn(*args, **kwargs):
        raise AssertionError("nothing may be spawned for a refused request")

    monkeypatch.setattr(module, "run_governed", no_spawn)
    # This bubblewrap was never shown to pass a bound Unix socket, so it cannot offer egress.
    backend = BwrapBackend(fake_report(failing=("bwrap.unix_socket",)), tmp_path / "work", runner=fake_runner(), namespace_phase=True)
    assert not backend.network
    networked = SecuritySpec(
        Sandbox(fs_read=("$WORKSPACE", "/usr"), net=Egress(allow=("pypi.org:443",)), limits=EXEC_SANDBOX.limits), Trust.UNTRUSTED, (Sink.NETWORK,)
    )
    escaping = SecuritySpec(Sandbox(fs_read=("$WORKSPACE", "/home"), limits=EXEC_SANDBOX.limits), Trust.UNTRUSTED)
    cases = [
        (request_for(Program.from_text("python", "print(1)")).__class__("ep_a", "c", "model", "run_python", "a" * 64, EXEC_SECURITY), "bulkhead.setup.no_program"),
        (request_for(Program.from_text("python", "print(1)"), spec=networked), "bulkhead.setup.network_unsupported"),
        (request_for(Program.from_text("python", "print(1)"), spec=escaping), "bulkhead.view.forbidden_path"),
    ]
    for request, code in cases:
        outcome = backend.execute(request, {})
        assert outcome.process is not None and not outcome.process.started
        assert outcome.process.setup_error == code
        assert outcome.audit["started"] is False and outcome.audit["cleanup_status"] == "not_started"
        assert outcome.audit["violations"] == [{"kind": "setup", "rule_id": code, "detail": "the sandbox was not set up, so the program did not run"}]
        assert outcome.audit["backend"] == "bwrap-namespace-phase"


@needs_libseccomp
def test_a_poisoned_workspace_refuses_every_later_call(tmp_path, monkeypatch):
    import bulkhead.backends.bwrap as module

    monkeypatch.setattr(module, "run_governed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    backend = BwrapBackend(fake_report(), tmp_path / "work", runner=fake_runner(), namespace_phase=True)
    backend.workspace("ep_poisoned").poisoned = True
    outcome = backend.execute(request_for(Program.from_text("python", "print(1)"), episode_id="ep_poisoned"), {})
    assert outcome.process.setup_error == "bulkhead.setup.workspace_unsafe"


# The attack fixtures themselves


# Integration: real bubblewrap (skipped unless BULKHEAD_INTEGRATION=1 and the probe proves the features)


@pytest.fixture(scope="module")
def t2_backend(tmp_path_factory):
    if not (LINUX and INTEGRATION):
        pytest.skip("unsupported: set BULKHEAD_INTEGRATION=1 on Linux to run the real bubblewrap backend")
    from bulkhead.probe import run_probe

    from bulkhead.cgroups import configured_parent

    report = run_probe(cgroup_parent=os.environ.get("BULKHEAD_CGROUP_PARENT"))
    parent = configured_parent()
    try:
        return BwrapBackend(report, tmp_path_factory.mktemp("bulkhead-work"), cgroup_parent=parent, namespace_phase=parent is None)
    except BackendUnavailable as exc:
        pytest.skip(f"unsupported: {exc}")


def small_spec(**limits):
    values = {"cpu_s": 10, "mem_mb": 256, "pids": 32, "disk_mb": 50, "out_kb": 4, "wall_s": 5} | limits
    return SecuritySpec(Sandbox(fs_read=EXEC_SANDBOX.fs_read, fs_write=EXEC_SANDBOX.fs_write, fs_deny=EXEC_SANDBOX.fs_deny, limits=Limits(**values)), Trust.UNTRUSTED)


def test_integration_benign_programs_run_inside_the_namespace_phase(t2_backend):
    python = t2_backend.execute(request_for(Program.from_text("python", "print(sum(range(10)))"), episode_id="ep_benign"), {})
    shell = t2_backend.execute(request_for(Program.from_text("shell", "echo hi > out/f && cat out/f; exit 3"), episode_id="ep_benign", call_id="c1"), {})

    assert python.audit["backend"] in ("bwrap-namespace-phase", "bwrap-t2") and python.audit["started"]
    assert (python.process.stdout, python.audit["exit_code"]) == ("45\n", 0)
    assert (shell.process.stdout, shell.audit["exit_code"]) == ("hi\n", 3)
    assert (t2_backend.workspace("ep_benign").path / "out" / "f").read_text() == "hi\n"


def test_integration_timeouts_and_output_caps(t2_backend):
    slow = t2_backend.execute(request_for(Program.from_text("shell", "sleep 30"), episode_id="ep_limits", spec=small_spec(wall_s=1)), {})
    loud = t2_backend.execute(request_for(Program.from_text("python", "print('x' * 100000)"), episode_id="ep_limits", call_id="c1", spec=small_spec()), {})

    assert slow.audit["timed_out"] and slow.audit["cleanup_status"] == "killed" and slow.audit["elapsed_s"] < 5
    assert loud.audit["stdout_truncated"] and len(loud.process.stdout) <= 4096 and loud.audit["stdout_bytes"] == 100001


def test_integration_namespace_capability_descriptor_and_environment_properties(t2_backend):
    program = (
        "import json, os, socket\n"
        "status = dict(l.split(':', 1) for l in open('/proc/self/status') if ':' in l)\n"
        "fds = []\n"
        "for n in range(3, 256):\n"
        "    try:\n        os.fstat(n); fds.append(n)\n    except OSError:\n        pass\n"
        "print(json.dumps({'hostname': socket.gethostname(), 'root': sorted(os.listdir('/')), 'pid': os.getpid(),\n"
        "  'cap': status['CapEff'].strip(), 'nnp': status['NoNewPrivs'].strip(), 'seccomp': status['Seccomp'].strip(),\n"
        "  'env': dict(os.environ), 'fds': fds, 'cwd': os.getcwd()}))\n"
    )
    outcome = t2_backend.execute(request_for(Program.from_text("python", program), episode_id="ep_properties"), {})
    facts = json.loads(outcome.process.stdout)

    assert facts["hostname"] == "bulkhead" and facts["pid"] < 10
    assert int(facts["cap"], 16) == 0 and facts["nnp"] == "1" and facts["seccomp"] == "2"
    assert facts["env"] == dict(SANDBOX_ENV) and facts["fds"] == [] and facts["cwd"] == "/workspace"
    assert not {"home", "root", "etc", "var", "run", "mnt", "opt", "srv", "boot"} & set(facts["root"])


# The attack corpus itself runs on this backend through the part 7 suite (tests/test_bulkhead_attack_suite.py).


def test_integration_network_goes_only_through_the_calls_proxy(t2_backend, tmp_path):
    """Real bubblewrap: the allowed fixture works through the bridge; everything else fails with evidence."""
    import http.server
    import threading

    from bulkhead.manifest import Egress

    if not t2_backend.network:
        pytest.skip("unsupported: the probe did not prove a bound Unix socket works in this bubblewrap")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "3")
            self.end_headers()
            self.wfile.write(b"ok\n")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    egress = Egress(allow=(f"127.0.0.1:{port}",), address_classes=("loopback",))
    spec = SecuritySpec(
        Sandbox(fs_read=EXEC_SANDBOX.fs_read, fs_write=EXEC_SANDBOX.fs_write, fs_deny=EXEC_SANDBOX.fs_deny, net=egress, limits=EXEC_SANDBOX.limits),
        Trust.UNTRUSTED, (Sink.FILESYSTEM, Sink.NETWORK),
    )
    program = f"""
import errno, http.client, json, socket
out = {{}}
c = http.client.HTTPConnection("127.0.0.1", 3128, timeout=5); c.set_tunnel("127.0.0.1", {port}); c.request("GET", "/")
out["allowed"] = c.getresponse().read().decode()
d = http.client.HTTPConnection("127.0.0.1", 3128, timeout=5); d.set_tunnel("exfil.test", 443)
try:
    d.request("GET", "/")
except OSError as exc:
    out["denied"] = str(exc)
def attempt(f):
    try:
        f(); return "ok"
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc))
out["direct"] = attempt(lambda: socket.create_connection(("127.0.0.1", {port}), timeout=2).close())
out["host_proxy_socket"] = attempt(lambda: socket.socket(socket.AF_UNIX).connect("/tmp/does-not-matter.sock"))
print(json.dumps(out, sort_keys=True))
"""
    try:
        outcome = t2_backend.execute(request_for(Program.from_text("python", program), episode_id="ep_network", spec=spec, timeout_s=30), {})
    finally:
        server.shutdown()
    report = json.loads(outcome.process.stdout)
    assert report["allowed"] == "ok\n" and "403" in report["denied"] and report["direct"] == "ECONNREFUSED"
    assert [(r["decision"], r["rule_id"]) for r in outcome.egress] == [("allow", "bulkhead.egress.allow"), ("deny", "bulkhead.egress.not_allowed")]
