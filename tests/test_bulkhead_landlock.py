"""Bulkhead part 5: Landlock rulesets and the tier-1 backend.

Rule planning is unit-tested against fake ABIs (0 to 7) and a temporary
workspace; applying is tested with a fake syscall. The tier-1 integration
tests run the real backend (real Landlock, seccomp, no_new_privs, rlimits)
wherever the part 1 probe proves every tier-1 requirement, and are skipped as
unsupported elsewhere. Tier 1 is a different boundary from tier 2; where an
attack is stopped differently, the fixture says so instead of reusing tier 2's
expectation.
"""

from __future__ import annotations

import errno
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bulkhead import landlock
from bulkhead.backends.base import BackendIdentity, Program
from bulkhead.backends.bwrap import BackendUnavailable
from bulkhead.backends.landlock import DEVICES, LandlockBackend, missing_for_t1
from bulkhead.backends.profiles import CGROUP_PROFILES, PROFILES, PROPERTIES, STATUSES, attack_matrix
from bulkhead.landlock import (
    FILE_RIGHTS,
    MIN_ABI,
    READ_ACCESS,
    READ_DIR,
    READ_FILE,
    TRUNCATE,
    WRITE_ACCESS,
    WRITE_FILE,
    LandlockError,
    Rule,
    Ruleset,
    covers,
    plan_ruleset,
)
from bulkhead.manifest import Egress, Sandbox, SecuritySpec, Sink, Trust
from bulkhead.policy import TaskPolicy
from bulkhead.types import SandboxTier
from steadyhand import EpisodeStatus, SteadyhandClient, build_registry, initial_messages, load_fixture
from steadyhand.replay import replay_episode, run_claimed_episode
from steadyhand.security import STORE_TOOLS, SecurityConfig
from steadyhand.tools_exec import EXEC_LIMITS, EXEC_SANDBOX
from toolsmith.protocols import QWEN3_PROFILE

from exec_support import ATTACKS, AttackContext, check_attack, load_attacks, request_for, snapshot

LINUX = sys.platform.startswith("linux")
MACHINE = platform.machine().lower()
TIER1 = json.loads((ATTACKS / "resource_and_tier1.json").read_text())
PART4 = load_attacks()


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    (root / "out").mkdir(parents=True)
    (root / ".git" / "hooks").mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]\n")
    (root / ".env").write_text("SECRET=1\n")
    (root / "README.txt").write_text("hi\n")
    (root / "data").mkdir()
    (root / "data" / "keep.txt").write_text("keep\n")
    (root / "data" / "private").mkdir()
    (root / "data" / "private" / "key").write_text("k\n")
    return root


def plan(sandbox, workspace, abi=7, **options):
    return plan_ruleset(sandbox, workspace, abi, home="/home/nobody", work_root=workspace.parent / "not-the-root", **options)


def rule_for(ruleset, path):
    matches = [rule for rule in ruleset.rules if rule.path == str(path)]
    return matches[0] if matches else None


# ABI handling


@pytest.mark.parametrize(
    ("abi", "fs_bits", "net", "scoped", "size"),
    [(0, 0, 0, 0, 8), (1, 13, 0, 0, 8), (2, 14, 0, 0, 8), (3, 15, 0, 0, 8), (4, 15, 3, 0, 16), (5, 16, 3, 0, 16), (6, 16, 3, 3, 24), (7, 16, 3, 3, 24)],
)
def test_rights_follow_the_abi(abi, fs_bits, net, scoped, size):
    assert landlock.fs_rights(abi) == (1 << fs_bits) - 1
    assert (landlock.net_rights(abi), landlock.scopes(abi), landlock.attr_size(abi)) == (net, scoped, size)


@pytest.mark.parametrize("abi", [0, 1, 2])
def test_abis_that_cannot_enforce_read_only_files_are_refused(abi, workspace):
    with pytest.raises(LandlockError) as raised:
        plan(EXEC_SANDBOX, workspace, abi)
    assert raised.value.code == "bulkhead.landlock.abi_too_old"
    assert MIN_ABI == 3


def test_newer_abis_add_tcp_denial_and_scopes(workspace):
    abi3, abi4, abi6 = (plan(EXEC_SANDBOX, workspace, abi) for abi in (3, 4, 6))
    assert not abi3.tcp_denied and not abi3.signals_scoped
    assert abi4.tcp_denied and not abi4.abstract_unix_scoped
    assert abi6.tcp_denied and abi6.signals_scoped and abi6.abstract_unix_scoped
    assert not abi3.handled_fs & landlock.IOCTL_DEV and abi6.handled_fs & landlock.IOCTL_DEV


# Rules


def test_reads_are_read_only_and_writes_are_confined_to_out(workspace):
    ruleset = plan(Sandbox(fs_read=("$WORKSPACE", "/usr"), fs_write=("$WORKSPACE/out",)), workspace)

    assert rule_for(ruleset, workspace).access == READ_ACCESS
    assert rule_for(ruleset, workspace / "out").access == WRITE_ACCESS
    assert rule_for(ruleset, "/usr").access == READ_ACCESS
    assert covers(ruleset, str(workspace / "README.txt"), READ_FILE)
    assert not covers(ruleset, str(workspace / "README.txt"), WRITE_FILE)
    assert not covers(ruleset, str(workspace / "README.txt"), TRUNCATE)
    assert covers(ruleset, str(workspace / "out" / "new.txt"), WRITE_FILE | TRUNCATE)
    assert not covers(ruleset, "/etc/passwd", READ_FILE)


def test_denied_subpaths_are_split_out_and_never_granted(workspace):
    sandbox = Sandbox(fs_read=("$WORKSPACE",), fs_write=("$WORKSPACE/out",), fs_deny=("$WORKSPACE/.git", "$WORKSPACE/.env", "$WORKSPACE/data/private"))
    ruleset = plan(sandbox, workspace)
    paths = {rule.path for rule in ruleset.rules}

    assert str(workspace) not in paths and str(workspace / "data") not in paths  # parents of denied entries get no rule
    assert {str(workspace / "README.txt"), str(workspace / "out"), str(workspace / "data" / "keep.txt")} <= paths
    for denied in (".git", ".git/config", ".git/hooks", ".env", "data/private", "data/private/key"):
        assert not covers(ruleset, str(workspace / denied), READ_FILE), denied
        assert not covers(ruleset, str(workspace / denied), READ_DIR), denied
    assert not covers(ruleset, str(workspace), READ_DIR)  # the cost: a directory holding a denied entry cannot be listed
    assert rule_for(ruleset, workspace / "README.txt").access == READ_ACCESS & FILE_RIGHTS


def test_a_denied_path_that_could_be_created_is_protected_by_splitting(workspace):
    sandbox = Sandbox(fs_read=("$WORKSPACE",), fs_write=("$WORKSPACE/out",), fs_deny=("$WORKSPACE/out/secret",))
    ruleset = plan(sandbox, workspace)
    assert rule_for(ruleset, workspace / "out") is None  # no rule on out itself, so nothing new appears in it
    assert not covers(ruleset, str(workspace / "out" / "secret"), WRITE_FILE)


def test_file_rules_never_carry_directory_rights(workspace):
    ruleset = plan(Sandbox(fs_read=("$WORKSPACE",), fs_deny=("$WORKSPACE/.env",)), workspace, extra=DEVICES)
    for rule in ruleset.rules:
        if not rule.directory:
            assert rule.access & ~FILE_RIGHTS == 0, rule
    assert rule_for(ruleset, "/dev/null").access == READ_FILE | WRITE_FILE
    assert rule_for(ruleset, "/dev/urandom").access == READ_FILE


@pytest.mark.parametrize("path", ["/", "/home", "/etc", "/proc", "/sys", "/tmp", "/root"])
def test_forbidden_host_paths_are_refused(path, workspace):
    with pytest.raises(LandlockError) as raised:
        plan(Sandbox(fs_read=(path,)), workspace)
    assert raised.value.code == "bulkhead.view.forbidden_path"


def test_absolute_writes_and_unsafe_symlinks_are_refused(workspace, tmp_path):
    with pytest.raises(LandlockError) as raised:
        plan(SimpleNamespace(fs_read=(), fs_write=("/usr/local",), fs_deny=()), workspace)
    assert raised.value.code == "bulkhead.landlock.write_outside_workspace"
    (workspace / "link").symlink_to(tmp_path)
    with pytest.raises(LandlockError) as raised:
        plan(Sandbox(fs_read=("$WORKSPACE/link",)), workspace)
    assert raised.value.code == "bulkhead.landlock.symlink"
    (workspace / ".env").unlink()
    (workspace / ".env").symlink_to("/etc/shadow")
    with pytest.raises(LandlockError):
        plan(EXEC_SANDBOX, workspace)


@pytest.mark.skipif(not os.path.islink("/lib") or not os.path.isdir("/usr"), reason="needs a merged-/usr host")
def test_merged_usr_links_are_covered_by_usr(workspace):
    ruleset = plan(Sandbox(fs_read=("/usr", "/lib")), workspace)
    assert "/lib" not in {rule.path for rule in ruleset.rules} and rule_for(ruleset, "/usr") is not None
    with pytest.raises(LandlockError):
        plan(Sandbox(fs_read=("/lib",)), workspace)  # /lib alone would grant a hierarchy the spec did not name


def test_a_ruleset_refuses_rights_its_abi_lacks_and_round_trips(workspace):
    with pytest.raises(LandlockError) as raised:
        Ruleset(3, landlock.fs_rights(3), 0, 0, (Rule("/usr", landlock.IOCTL_DEV, True),))
    assert raised.value.code == "bulkhead.landlock.unsupported_right"
    with pytest.raises(LandlockError) as raised:
        Ruleset(7, landlock.fs_rights(7), 0, 0, (Rule("/etc/hosts", READ_DIR, False),))
    assert raised.value.code == "bulkhead.landlock.file_rule"
    with pytest.raises(LandlockError):
        Ruleset(3, landlock.fs_rights(3), landlock.net_rights(4), 0, ())
    ruleset = plan(EXEC_SANDBOX, workspace, extra=DEVICES)
    assert Ruleset.from_dict(json.loads(json.dumps(ruleset.as_dict()))) == ruleset
    assert set(landlock.describe(ruleset)) == {"abi", "rules", "tcp_denied", "signals_scoped", "abstract_unix_scoped"}


# Applying, with a fake kernel


class FakeKernel:
    def __init__(self, abi=7, fail=None):
        self.abi = abi
        self.fail = fail
        self.calls = []

    def __call__(self, number, *args):
        self.calls.append(number)
        if number == landlock.SYS_CREATE_RULESET and args[0] is None:
            return self.abi
        if number == self.fail:
            raise OSError(errno.EINVAL, "Invalid argument")
        return os.open("/dev/null", os.O_RDONLY) if number == landlock.SYS_CREATE_RULESET else 0


def test_prepare_adds_every_rule_and_restrict_enforces(workspace):
    ruleset = plan(EXEC_SANDBOX, workspace)
    kernel = FakeKernel()
    fd = landlock.prepare(ruleset, syscall=kernel)
    landlock.restrict_self(fd, syscall=kernel)
    assert kernel.calls.count(landlock.SYS_ADD_RULE) == len(ruleset.rules) and kernel.calls[-1] == landlock.SYS_RESTRICT_SELF


@pytest.mark.parametrize(
    ("kernel", "code"),
    [(FakeKernel(abi=3), "bulkhead.landlock.abi_mismatch"), (FakeKernel(fail=landlock.SYS_ADD_RULE), "bulkhead.landlock.add_rule"),
     (FakeKernel(fail=landlock.SYS_RESTRICT_SELF), "bulkhead.landlock.restrict")],
)
def test_every_kernel_failure_is_a_typed_error(workspace, kernel, code):
    ruleset = plan(EXEC_SANDBOX, workspace)
    with pytest.raises(LandlockError) as raised:
        landlock.restrict_self(landlock.prepare(ruleset, syscall=kernel), syscall=kernel)
    assert raised.value.code == code


def test_a_rule_path_that_changed_type_is_refused(workspace):
    ruleset = plan(EXEC_SANDBOX, workspace)
    (workspace / "README.txt").unlink()
    (workspace / "README.txt").mkdir()
    with pytest.raises(LandlockError) as raised:
        landlock.prepare(ruleset, syscall=FakeKernel())
    assert raised.value.code == "bulkhead.landlock.rule_path"


# The helper refuses to exec after any failure


def test_the_helper_never_execs_after_a_setup_failure(tmp_path):
    from bulkhead.helper import config_memfd, helper_command

    marker = tmp_path / "ran"
    good = {
        "status_fd": 1, "barrier_fd": None, "seccomp_fd": None, "keep_fds": [], "landlock": None,
        "rlimits": {"cpu_s": 5, "fsize_bytes": 1 << 20, "nofile": 64, "core_bytes": 0, "address_space_bytes": None},
        "target": ["/bin/sh", "-c", f"touch {marker}"], "supervisor_pid": os.getpid(),
    }
    for change, code in [({"extra": 1}, "config_fields"), ({"supervisor_pid": 1}, "parent_gone"), ({"target": ["sh"]}, "target"),
                         ({"rlimits": {"cpu_s": 0}}, "rlimits")]:
        config = {**good, **change}
        fd = config_memfd(config)
        try:
            done = subprocess.run(helper_command(fd), pass_fds=(fd,), capture_output=True, timeout=30)
        finally:
            os.close(fd)
        assert done.returncode == 125 and f"bulkhead-helper: {code}".encode() in done.stderr, (change, done.stderr)
        assert not marker.exists()


# The profile matrix


def test_every_profile_describes_every_property_and_t0_runs_nothing():
    assert set(PROFILES) == {"bwrap-t2", "bwrap-namespace-phase", "landlock-t1-cgroup", "landlock-t1", "T0"}
    for profile in PROFILES.values():
        assert set(profile) == set(PROPERTIES)
    assert all("not a sandbox" in value for key, value in PROFILES["T0"].items() if key != "tier")
    with pytest.raises(ValueError, match="T0 is not a sandbox"):
        BackendIdentity("anything", "1", SandboxTier.T0)


def test_the_attack_matrix_never_claims_a_pass_and_covers_every_profile():
    matrix = attack_matrix(PART4, TIER1)
    assert len(matrix) == 30 and set(TIER1["tier1"]) == {attack["id"] for attack in PART4["attacks"]}
    for attack, row in matrix.items():
        assert set(row) == set(PROFILES), attack
        assert set(row.values()) <= STATUSES, attack
        assert row["T0"] == "not_applicable"
    assert matrix["fs.read_env"]["landlock-t1"] == "different_boundary" and matrix["fs.read_env"]["bwrap-t2"] == "blocked"
    assert matrix["resource.fork_bomb"]["landlock-t1"] == "not_applicable"  # no PID ceiling without a cgroup
    for profile in CGROUP_PROFILES:
        assert matrix["resource.fork_bomb"][profile] == "killed"


# Integration: the real tier-1 backend, wherever the probe proves tier 1


@pytest.fixture(scope="module")
def t1_report():
    if not LINUX:
        pytest.skip("unsupported: tier 1 needs Linux")
    from bulkhead.probe import run_probe

    report = run_probe()
    missing = missing_for_t1(report, cgroup=False)
    if missing:
        pytest.skip(f"unsupported: the probe did not prove {', '.join(missing)}")
    return report


@pytest.fixture(scope="module")
def t1_backend(t1_report, tmp_path_factory):
    return LandlockBackend(t1_report, tmp_path_factory.mktemp("bulkhead-t1"))


def test_integration_t1_runs_benign_programs_under_real_landlock(t1_backend):
    t1_backend.prepare("ep_t1_benign", {"README.txt": b"hello\n", ".env": b"SECRET=1\n"})
    python = t1_backend.execute(request_for(Program.from_text("python", "print(sum(range(10)))"), episode_id="ep_t1_benign"), {})
    shell = t1_backend.execute(request_for(Program.from_text("shell", "cat README.txt; echo made > out/f; cat out/f; exit 4"), episode_id="ep_t1_benign", call_id="c1"), {})

    assert python.audit["backend"] == "landlock-t1" and python.audit["tier"] == "T1"
    assert (python.process.stdout, python.audit["exit_code"], python.audit["cleanup_status"]) == ("45\n", 0, "clean")
    assert (shell.process.stdout, shell.audit["exit_code"]) == ("hello\nmade\n", 4)


def test_integration_t1_refuses_network_and_unenforceable_specs(t1_backend, monkeypatch):
    import bulkhead.backends.landlock as module

    monkeypatch.setattr(module, "run_governed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    networked = SecuritySpec(Sandbox(fs_read=("$WORKSPACE", "/usr"), net=Egress(allow=("pypi.org:443",)), limits=EXEC_LIMITS), Trust.UNTRUSTED, (Sink.NETWORK,))
    homeward = SecuritySpec(Sandbox(fs_read=("$WORKSPACE", "/usr", "/home"), limits=EXEC_LIMITS), Trust.UNTRUSTED)
    no_runtime = SecuritySpec(Sandbox(fs_read=("$WORKSPACE",), fs_write=("$WORKSPACE/out",), limits=EXEC_LIMITS), Trust.UNTRUSTED, (Sink.FILESYSTEM,))
    for spec, code in [(networked, "bulkhead.setup.network_unsupported"), (homeward, "bulkhead.view.forbidden_path"), (no_runtime, "bulkhead.view.interpreter")]:
        outcome = t1_backend.execute(request_for(Program.from_text("python", "print(1)"), episode_id="ep_t1_refused", spec=spec), {})
        assert outcome.process.setup_error == code and not outcome.audit["started"]


@pytest.mark.parametrize("attack", PART4["attacks"], ids=[attack["id"] for attack in PART4["attacks"]])
def test_integration_t1_the_first_twenty_attacks_fail(t1_backend, tmp_path, monkeypatch, attack):
    if MACHINE not in attack.get("architectures", [MACHINE]):
        pytest.skip(f"unsupported: {attack['id']} needs {attack['architectures']}")
    tier1 = TIER1["tier1"][attack["id"]]
    expected = {**attack, "expect": tier1["expect"], "expected_verdict": tier1["status"]}
    context = AttackContext.create(tmp_path, listeners=attack["canaries"].get("no_connections", False))
    monkeypatch.setenv("HF_TOKEN", context.token)
    episode = "ep_t1_" + attack["id"].replace(".", "_")
    workspace = t1_backend.prepare(episode, context.workspace_files(PART4))
    before = snapshot(context, workspace.path)
    try:
        outcome = t1_backend.execute(request_for(context.program(attack, workspace=str(workspace.path)), episode_id=episode, timeout_s=20), {})
        problems = check_attack(expected, outcome, context, before=before, workspace=workspace.path)
    finally:
        if context.listeners is not None:
            context.listeners.stop()
        t1_backend.cleanup(episode)
    assert problems == []


class FakeService:
    profile = QWEN3_PROFILE
    use_grammars = True

    def __init__(self, *responses):
        self.responses = list(responses)

    def chat(self, request):
        return self.responses.pop(0)


def completion(content=None, calls=()):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [{"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, n, a in calls]
    return {"choices": [{"index": 0, "message": message}], "usage": {"prompt_tokens": 40, "completion_tokens": 6}, "toolsmith": {"repair_attempts": 0}}


def test_integration_t1_runs_an_episode_end_to_end_and_replay_spawns_nothing(t1_backend, tmp_path, monkeypatch):
    policy = TaskPolicy(
        tools=(*STORE_TOOLS, "run_python", "run_shell"), fs_read=EXEC_SANDBOX.fs_read, fs_write=EXEC_SANDBOX.fs_write,
        fs_deny=EXEC_SANDBOX.fs_deny, limits=EXEC_LIMITS, required_tier=SandboxTier.T1,
    )
    result = run_claimed_episode(
        tmp_path, SteadyhandClient(FakeService(completion(calls=[("c1", "run_python", {"code": "print(6 * 7)"})]), completion("42."))),
        build_registry(exec_tools=True), load_fixture().clone(), initial_messages("Compute 6 * 7."),
        security=SecurityConfig(policy, available_tiers=(SandboxTier.T1,)), backend=t1_backend, sleep=lambda s: None,
    )
    records = [json.loads(line) for line in Path(result.journal_path).read_text().splitlines()]
    [run] = [record for record in records if record["event"] == "sandbox_execution"]

    assert result.status is EpisodeStatus.DONE
    assert run["backend"] == "landlock-t1" and run["tier"] == "T1" and run["result"]["value"]["stdout"] == "42\n"
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("replay spawned a process")))
    assert replay_episode(result.journal_path).status is EpisodeStatus.DONE


def test_integration_t1_never_stands_in_for_tier_2(t1_backend, tmp_path):
    from steadyhand import EpisodeJournal, run_episode

    assert t1_backend.identity.tier is SandboxTier.T1
    with pytest.raises(ValueError, match="available tiers"):
        run_episode(
            SteadyhandClient(FakeService()), build_registry(exec_tools=True), load_fixture().clone(), initial_messages("x"),
            journal=EpisodeJournal.create(tmp_path),
            security=SecurityConfig(TaskPolicy(tools=("run_python",), limits=EXEC_LIMITS), available_tiers=(SandboxTier.T2,)), backend=t1_backend,
        )


def test_landlock_backend_refuses_to_exist_without_proof(tmp_path):
    from test_bulkhead_bwrap import fake_report

    for capability in ("landlock.fs", "seccomp.socket_deny", "process.group_cleanup", "rlimits"):
        with pytest.raises(BackendUnavailable) as raised:
            LandlockBackend(fake_report(failing=(capability,)), tmp_path / "w", kernel_abi=7)
        assert capability in raised.value.missing
    with pytest.raises(BackendUnavailable) as raised:
        LandlockBackend(fake_report(failing=("cgroup.kill",)), tmp_path / "w", kernel_abi=7, cgroup_parent=object())
    assert "cgroup.kill" in raised.value.missing
