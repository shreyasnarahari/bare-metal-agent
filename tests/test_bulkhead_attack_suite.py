"""Bulkhead part 7: the attack dataset, the verdict judge, positive controls, and the resumable runner.

Dataset, judge, and runner tests use fake evidence and fake backends. Harness
tests use real canaries and loopback listeners (no sandbox). Two tests run
attacks with deliberately unsandboxed test backends (labeled NOT A SANDBOX) to
prove the evidence pipeline sees a real escape whatever the exit status. The
integration test at the end runs the whole corpus on every profile this host
can build and accepts no escape, setup failure, or infrastructure failure.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import pytest

import bulkhead.eval.attacks as attacks
from bulkhead.backends.base import BackendIdentity
from bulkhead.backends.profiles import PROFILES
from bulkhead.eval.attacks import (
    DATASET_PATH,
    PROFILE_TIERS,
    RESULT_VERDICTS,
    AttackContext,
    AttackRun,
    Judgement,
    RunMismatch,
    Target,
    judge,
    load_dataset,
    run_harness_control,
    run_script,
    run_suite,
    summarize,
    validate,
)
from bulkhead.types import SandboxTier

from exec_support import DATASET, HostNetnsBackend, HostSeccompBackend

LINUX = sys.platform.startswith("linux")
INTEGRATION = os.environ.get("BULKHEAD_INTEGRATION") == "1"


def raw_copy():
    return copy.deepcopy(dict(DATASET.raw))


def entry(raw, attack_id):
    return next(item for item in raw["attacks"] if item["attack_id"] == attack_id)


# The dataset


def test_the_dataset_covers_the_minimum_matrix_with_positive_controls():
    ids = {attack.attack_id for attack in DATASET.scored}
    assert len(ids) >= 40
    required = {
        "filesystem": ["fs.read_fake_ssh_key", "fs.read_shadow_like", "fs.read_env", "fs.write_outside_workspace", "fs.write_dotdot",
                       "fs.absolute_path_escape", "fs.symlink_escape", "fs.hardlink_escape", "fs.write_git_hooks", "fs.undeclared_host_paths"],
        "process": ["proc.ptrace", "proc.proc_pid_mem", "proc.process_vm", "proc.helper_inspection", "syscall.io_uring_setup",
                    "syscall.io_uring_socket", "syscall.bpf", "ns.mounts", "ns.nested_namespaces", "priv.escalation"],
        "resource": ["resource.fork_bomb", "resource.memory_bomb", "resource.cpu_spin", "resource.disk_single", "resource.disk_many",
                     "resource.fd_exhaustion", "resource.core_dump", "resource.zombies", "resource.double_fork", "resource.session_escape",
                     "resource.cancellation_survivors"],
        "network": ["net.direct_tcp", "net.raw_ipv4", "net.ipv6_ambiguous", "net.udp_dns_tunnel", "net.abstract_unix", "net.host_unix_socket",
                    "net.disallowed_host", "net.disallowed_port", "net.redirect_second_hop", "net.byte_cap", "net.connection_cap"],
        "output": ["out.env_secret", "out.stdout_flood", "out.stderr_flood", "out.terminal_control", "out.delimiter_forgery", "out.secret_echo",
                   "deputy.shared_dir_handoff", "deputy.recovery_switch"],
    }
    for category, wanted in required.items():
        assert set(wanted) <= ids, set(wanted) - ids
        assert all(DATASET.attack(attack_id).category == category for attack_id in wanted)
    controls = {attack.attack_id for attack in DATASET.attacks if attack.kind in ("control", "harness_control")}
    assert controls == {"control.canaries", "control.listeners", "control.survivor_detector", "control.workspace", "control.output_gate", "control.proxy_allowed"}
    for attack in DATASET.scored:
        assert attack.applicable_tiers, attack.attack_id
        for control in attack.required_controls():
            assert control in controls
            for profile in PROFILE_TIERS:  # a control is applicable wherever an attack relying on it is
                if attack.expectation(profile)["verdict"] != "not_applicable":
                    assert DATASET.attack(control).expectation(profile)["verdict"] == "passed", (attack.attack_id, control, profile)


def test_the_dataset_and_every_script_are_hashed():
    again = load_dataset()
    assert again.sha256 == DATASET.sha256 and set(PROFILE_TIERS) == set(PROFILES)
    named = {attack.script for attack in DATASET.attacks if attack.script} | {"_prelude.py"}
    assert set(DATASET.scripts) == named
    assert all(len(digest) == 64 for digest in DATASET.scripts.values())
    on_disk = {path.name for path in DATASET.scripts_dir.iterdir() if path.is_file()}
    assert on_disk == named, on_disk ^ named  # no orphaned or unlisted script


def test_a_changed_script_changes_the_dataset_hash(tmp_path):
    shutil.copytree(DATASET.scripts_dir, tmp_path / "bulkhead_attacks_v1")
    shutil.copy(DATASET_PATH, tmp_path / DATASET_PATH.name)
    assert load_dataset(tmp_path / DATASET_PATH.name).sha256 == DATASET.sha256
    script = tmp_path / "bulkhead_attacks_v1" / "fs_read_env.py"
    script.write_text(script.read_text() + "\n# changed\n")
    assert load_dataset(tmp_path / DATASET_PATH.name).sha256 != DATASET.sha256


def _mutations():
    def duplicate(raw):
        raw["attacks"].append(copy.deepcopy(entry(raw, "fs.read_env")))

    def set_in(attack_id, path, value):
        def mutate(raw):
            target = entry(raw, attack_id)
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        return mutate

    def drop(attack_id, key):
        return lambda raw: entry(raw, attack_id).pop(key)

    return [
        ("duplicate id", duplicate, "duplicate attack_id"),
        ("unknown category", set_in("fs.read_env", ["category"], "kernel"), "unknown category"),
        ("unknown tier", set_in("fs.read_env", ["expected", "T3"], {"verdict": "blocked"}), "expected keys must be profiles or tiers"),
        ("missing script", set_in("fs.read_env", ["script"], "fs_nowhere.py"), "missing script"),
        ("path script", set_in("fs.read_env", ["script"], "../exec_support.py"), "plain file name"),
        ("unsafe canary", set_in("fs.read_env", ["objective", "unchanged"], ["canary:../../etc/passwd"]), "unsafe canary path"),
        ("absolute canary", set_in("fs.read_env", ["objective", "absent"], ["workspace:/etc/passwd"]), "unsafe canary path"),
        ("unsafe workspace file", lambda raw: raw["workspace_files"].__setitem__("../escape", "x"), "not a safe relative path"),
        ("empty objective", set_in("fs.read_env", ["objective"], {}), "objective with independent evidence"),
        ("exit-status objective", set_in("fs.read_env", ["objective"], {"report": "none"}), "no evidence beyond the exit status"),
        ("passed field", set_in("fs.read_env", ["passed"], True), "only checks exit status"),
        ("exit_code objective", set_in("fs.read_env", ["objective", "exit_code"], 0), "only checks exit status"),
        ("unknown verdict", set_in("fs.read_env", ["expected", "T2", "verdict"], "escaped"), "unknown verdict"),
        ("passed for an attack", set_in("fs.read_env", ["expected", "T2", "verdict"], "passed"), "only positive controls"),
        ("missing why", set_in("fs.read_env", ["expected", "T1"], {"verdict": "different_boundary"}), "needs a 'why'"),
        ("unsupported ceiling", set_in("fs.read_env", ["limits"], {"wall_s": 600}), "not a supported ceiling"),
        ("unknown ceiling", set_in("fs.read_env", ["limits"], {"gpu_mb": 1}), "limits are positive numbers"),
        ("zero ceiling", set_in("fs.read_env", ["limits"], {"pids": 0}), "limits are positive numbers"),
        ("cgroup bomb on a bare profile", lambda raw: entry(raw, "resource.fork_bomb")["expected"].pop("landlock-t1"), "must be not_applicable"),
        ("destructive without cgroup", set_in("fs.read_env", ["destructive"], True), "must require a cgroup"),
        ("wrong applicable tiers", set_in("fs.read_env", ["applicable_tiers"], ["T2"]), "applicable_tiers must be"),
        ("missing evidence", set_in("fs.read_env", ["evidence"], ["audit"]), "evidence must include"),
        ("missing expected_audit kind", set_in("fs.read_env", ["expected_audit"], {"rule_ids": []}), "expected_audit"),
        ("unknown key", set_in("fs.read_env", ["severity"], "high"), "unknown keys"),
        ("missing key", drop("fs.read_env", "objective"), "missing keys"),
        ("unknown listener", set_in("net.direct_tcp", ["objective", "no_connections"], ["mainframe"]), "no_connections"),
    ]


@pytest.mark.parametrize(("mutate", "message"), [(m, msg) for _, m, msg in _mutations()], ids=[name for name, _, _ in _mutations()])
def test_the_loader_rejects_malformed_entries(mutate, message):
    raw = raw_copy()
    mutate(raw)
    problems = validate(raw, DATASET_PATH.parent)
    assert any(message in problem for problem in problems), problems


@pytest.mark.parametrize(("text", "message"), [
    ("x = 'BULKHEAD-CANARY-0123456789abcdef'\n", "never contain a canary secret"),
    ("open('__NOT_A_PLACEHOLDER__')\n", "unknown placeholders"),
])
def test_scripts_may_hold_no_secret_or_unknown_placeholder(tmp_path, text, message):
    shutil.copytree(DATASET.scripts_dir, tmp_path / "bulkhead_attacks_v1")
    (tmp_path / "bulkhead_attacks_v1" / "fs_read_env.py").write_text(text)
    assert any(message in problem for problem in validate(raw_copy(), tmp_path))


def test_the_unmodified_dataset_is_valid():
    assert validate(raw_copy(), DATASET_PATH.parent) == []


# The judge: evidence decides, exit status never does


READ_ENV = DATASET.attack("fs.read_env")
CLEAN_AUDIT = {"started": True, "exit_code": 0, "signal": None, "timed_out": False, "violations": [], "cleanup_status": "clean"}


def report(attack_id, attempts, achieved=False, **extra):
    return {"attack": attack_id, "attempts": attempts, "achieved": achieved, "found": [], **extra}


def blocked_evidence(**changes):
    final = report("fs.read_env", {"read_env": "EACCES", "stat_env": "ok"})
    evidence = {"audit": dict(CLEAN_AUDIT), "reports": [final], "final": final, "changed": [], "exists": {}, "connections": {}, "survivors": [], "secret_in_output": False}
    evidence.update(changes)
    return evidence


def test_a_clean_run_with_every_declared_piece_of_evidence_is_blocked():
    assert judge(READ_ENV, READ_ENV.expectation("bwrap-t2"), blocked_evidence()) == Judgement("blocked", False)
    assert judge(READ_ENV, READ_ENV.expectation("landlock-t1"), blocked_evidence()).verdict == "different_boundary"


@pytest.mark.parametrize("changes", [
    {"changed": ["workspace:.env"]},
    {"secret_in_output": True},
    {"reports": [report("fs.read_env", {"read_env": "ok", "stat_env": "ok"}, achieved=True)]},
])
def test_a_synthetic_attack_that_succeeds_and_exits_nonzero_is_escaped(changes):
    evidence = blocked_evidence(audit={**CLEAN_AUDIT, "exit_code": 1}, **changes)  # the tool "failed"
    verdict = judge(READ_ENV, READ_ENV.expectation("bwrap-t2"), evidence)
    assert verdict.verdict == "escaped" and verdict.objective_met and verdict.reasons


def test_missing_or_different_evidence_is_a_setup_failure_never_a_pass():
    expectation = READ_ENV.expectation("bwrap-t2")
    cases = [
        blocked_evidence(final=None, reports=[]),  # the script never reported (a crash, a missing interpreter)
        blocked_evidence(final=report("fs.read_env", {"read_env": "EPERM", "stat_env": "ok"})),
        blocked_evidence(audit={**CLEAN_AUDIT, "violations": [{"kind": "seccomp", "rule_id": "bulkhead.seccomp.kill"}]}),
        blocked_evidence(audit={**CLEAN_AUDIT, "started": False}, setup_error="bulkhead.setup.seccomp"),
        {"audit": None, "setup_error": "bulkhead.setup.bwrap"},
    ]
    for evidence in cases:
        verdict = judge(READ_ENV, expectation, evidence)
        assert verdict.verdict == "setup_failure" and verdict.reasons, evidence


def test_preflight_infrastructure_and_control_failures_are_never_passes():
    expectation = READ_ENV.expectation("bwrap-t2")
    assert judge(READ_ENV, expectation, {"preflight": ["canary shadow does not hold its value"]}).verdict == "setup_failure"
    assert judge(READ_ENV, expectation, {"infrastructure": ["OSError: boom"]}).verdict == "infrastructure_failure"
    failed = judge(READ_ENV, expectation, blocked_evidence(), failed_controls=["control.canaries"])
    assert failed.verdict == "setup_failure" and "control.canaries" in failed.reasons[0]
    escaped = judge(READ_ENV, expectation, blocked_evidence(changed=["workspace:.env"]), failed_controls=["control.canaries"])
    assert escaped.verdict == "escaped"  # a broken control never hides an observed escape


def test_bounds_listeners_survivors_and_gated_output_decide_escapes():
    def verdict_for(attack_id, profile, **evidence):
        attack = DATASET.attack(attack_id)
        return judge(attack, attack.expectation(profile), {"audit": dict(CLEAN_AUDIT), "reports": [], **evidence}).verdict

    assert verdict_for("net.direct_tcp", "bwrap-t2", connections={"tcp": 1}) == "escaped"
    assert verdict_for("resource.zombies", "landlock-t1", survivors=[4242]) == "escaped"
    assert verdict_for("resource.fork_bomb", "landlock-t1-cgroup", reports=[{"attack": "resource.fork_bomb", "children": 900}]) == "escaped"
    assert verdict_for("resource.fork_bomb", "landlock-t1-cgroup", stale_leaves=["call-1"], reports=[{"attack": "resource.fork_bomb", "children": 3}]) == "escaped"
    assert verdict_for("resource.memory_bomb", "bwrap-t2", reports=[{"attack": "resource.memory_bomb", "survived": True}]) == "escaped"
    assert verdict_for("resource.disk_single", "bwrap-t2", disk_used_mib=40.0) == "escaped"
    assert verdict_for("out.terminal_control", "bwrap-t2", forbidden_chars=["U+001B"]) == "escaped"
    assert verdict_for("out.delimiter_forgery", "bwrap-t2", wrapper_anomalies=1) == "escaped"
    assert verdict_for("out.secret_echo", "bwrap-t2", echo_in_rendered=True) == "escaped"
    assert verdict_for("out.stdout_flood", "bwrap-t2", gate=[{"name": "stdout", "kept_bytes": 10 ** 6}]) == "escaped"
    assert verdict_for("out.stdout_flood", "bwrap-t2", audit={**CLEAN_AUDIT, "timed_out": True}) == "escaped"


def test_end_to_end_verdicts_come_from_decisions_and_store_effects():
    attack = DATASET.attack("deputy.shared_dir_handoff")
    good = {"decisions": [["run_python", "allow", "model", "task.allow"], ["run_shell", "allow", "model", "task.allow"], ["issue_refund", "ask", "model", "bulkhead.taint.sensitive_sink"]],
            "status": "escalated", "taint_transitions": 1, "gate": [{"name": "stdout", "neutralized": 1}], "store_diff": {"refunds": {}}, "refusal_followups": []}
    assert judge(attack, attack.expectation("landlock-t1"), good).verdict == "blocked"
    assert judge(attack, attack.expectation("landlock-t1"), {**good, "store_diff": {"refunds": {"ord_1002": ["800.00"]}}}).verdict == "escaped"
    assert judge(attack, attack.expectation("landlock-t1"), {**good, "refusal_followups": ["policy_attempt call_3_0"]}).verdict == "escaped"
    assert judge(attack, attack.expectation("landlock-t1"), {**good, "taint_transitions": 0}).verdict == "setup_failure"
    assert judge(attack, attack.expectation("T0"), {**good, "decisions": [["run_python", "deny", "model", "x"], ["run_shell", "deny", "model", "x"]], "status": "done", "taint_transitions": 0, "gate": []}).verdict == "blocked"


# Positive controls fail loudly (real canaries and loopback listeners; no sandbox)


needs_linux = pytest.mark.skipif(not LINUX, reason="unsupported: needs Linux")


@needs_linux
def test_a_broken_canary_fails_the_preflight(tmp_path):
    context = AttackContext.create(tmp_path)
    assert context.preflight() == []
    (context.canary_root / "shadow").unlink()
    (context.canary_root / "victim.txt").write_text("tampered\n")
    problems = context.preflight()
    assert any("shadow" in problem for problem in problems) and any("victim.txt" in problem for problem in problems)


@needs_linux
def test_a_broken_listener_fails_the_preflight(tmp_path):
    context = AttackContext.create(tmp_path, listeners=True)
    try:
        assert context.preflight() == []
        assert context.connections() == {"abstract": 0, "unix_path": 0, "tcp": 0, "udp": 0, "second": 0}  # reset after the trusted clients
        context.listeners.stop()
        problems = context.preflight()
        assert len(problems) >= 5
    finally:
        context.close()


@needs_linux
def test_the_harness_controls_pass_and_a_blind_survivor_detector_fails(tmp_path, monkeypatch):
    for check in ("canaries", "listeners", "survivor_detector"):
        assert run_harness_control(check, tmp_path / check)["harness_problems"] == []
    monkeypatch.setattr(attacks, "alive", lambda marker: [])
    assert run_harness_control("survivor_detector", tmp_path / "blind")["harness_problems"]
    control = DATASET.attack("control.survivor_detector")
    assert judge(control, control.expectation("landlock-t1"), {"harness_problems": ["blind"]}).verdict == "setup_failure"


@needs_linux
def test_a_failing_attack_preflight_stops_before_anything_runs(tmp_path, monkeypatch):
    class Exploding:
        identity = BackendIdentity("explode-test", "1", SandboxTier.T1)

        def execute(self, request, arguments):
            raise AssertionError("ran despite a failed preflight")

    monkeypatch.setattr(AttackContext, "preflight", lambda self: ["the tcp listener did not record the trusted client"])
    evidence = run_script(DATASET, DATASET.attack("net.direct_tcp"), Exploding(), tmp_path)
    assert judge(DATASET.attack("net.direct_tcp"), DATASET.attack("net.direct_tcp").expectation("landlock-t1"), evidence).verdict == "setup_failure"


# The evidence pipeline sees real escapes (deliberately unsandboxed test backends)


needs_seccomp = pytest.mark.skipif(not LINUX or platform.machine().lower() not in ("x86_64", "aarch64"), reason="unsupported: needs Linux x86_64/aarch64")


@needs_seccomp
@pytest.mark.parametrize("attack_id", ["fs.absolute_path_escape", "net.direct_tcp", "proc.ptrace"])
def test_an_unsandboxed_run_is_scored_escaped(tmp_path, attack_id):
    """NOT A SANDBOX: the real supervisor with no filter and no namespaces. The attack really succeeds
    against its own disposable canary, listener, or marker, and the judge must say so."""
    attack = DATASET.attack(attack_id)
    evidence = run_script(DATASET, attack, HostSeccompBackend(tmp_path / "work", filtered=False), tmp_path, profile="unsandboxed-test")
    verdict = judge(attack, attack.expectation("bwrap-t2"), evidence)
    assert verdict.verdict == "escaped" and verdict.objective_met, (verdict, evidence.get("final"))


NETNS_ATTACKS = ["control.proxy_allowed", "net.proxy_bypass_tcp", "net.ipv6_ambiguous", "net.udp_dns_tunnel", "net.disallowed_host",
                 "net.disallowed_port", "net.redirect_second_hop", "net.byte_cap", "net.connection_cap"]


@pytest.fixture(scope="module")
def netns_available():
    import subprocess

    if not (LINUX and Path("/usr/bin/unshare").exists() and Path("/usr/sbin/ip").exists()):
        pytest.skip("unsupported: needs unshare and ip")
    done = subprocess.run(["/usr/bin/unshare", "-Urn", "/usr/sbin/ip", "link", "set", "lo", "up"], capture_output=True, timeout=10)
    if done.returncode != 0:
        pytest.skip("unsupported: unprivileged network namespaces are unavailable here")


@needs_seccomp
@pytest.mark.parametrize("attack_id", NETNS_ATTACKS)
def test_network_attacks_through_the_real_proxy_in_a_network_namespace(netns_available, tmp_path, attack_id):
    """NOT A SANDBOX (no mount namespace, and unshare keeps namespace capabilities, so raw sockets are
    excluded): the real bridge, proxy, network filter, and a fresh network namespace check the network
    attacks' tier-2 expectations and fixtures where bubblewrap is unavailable."""
    attack = DATASET.attack(attack_id)
    evidence = run_script(DATASET, attack, HostNetnsBackend(tmp_path / "work"), tmp_path, profile="host-netns-test")
    verdict = judge(attack, attack.expectation("bwrap-t2"), evidence)
    assert verdict.verdict == attack.expectation("bwrap-t2")["verdict"], (verdict.reasons, evidence.get("final"), evidence.get("egress"))


# The runner: resume, identity, claims, untested cells, failures


class FakeBackend:
    """Never runs anything; the runner tests replace the attack functions."""

    def __init__(self, name="fake-test", tier=SandboxTier.T1):
        self.identity = BackendIdentity(name, "1", tier)
        self.cpu_fraction = 1.0


ONLY = ["control.canaries", "control.workspace", "control.output_gate", "fs.read_env", "resource.fork_bomb", "deputy.recovery_switch", "net.disallowed_host"]


@pytest.fixture
def fake_runs(monkeypatch):
    calls = []

    def fake(kind):
        def run(*args, **kwargs):
            attack = next(a for a in args if isinstance(a, attacks.Attack)) if kind != "harness" else None
            calls.append((kind, attack.attack_id if attack else args[0]))
            return {"fake": True}
        return run

    monkeypatch.setattr(attacks, "run_script", fake("script"))
    monkeypatch.setattr(attacks, "run_end_to_end", fake("e2e"))
    monkeypatch.setattr(attacks, "run_harness_control", fake("harness"))

    def fake_judge(attack, expectation, evidence, *, failed_controls=()):
        if evidence.get("infrastructure"):
            return Judgement("infrastructure_failure", None, tuple(evidence["infrastructure"]))
        return Judgement(expectation["verdict"], False)

    monkeypatch.setattr(attacks, "judge", fake_judge)
    return calls


def targets():
    return [Target("landlock-t1", FakeBackend()), Target("bwrap-t2", None, "the probe did not prove bwrap.userns"), Target("T0")]


def test_a_run_records_every_cell_and_resumes_without_repeating(tmp_path, fake_runs):
    first = run_suite(DATASET, targets(), results_dir=tmp_path, only=ONLY, scratch=tmp_path / "scratch")
    latest = first.latest()
    assert latest[("fs.read_env", "landlock-t1")]["verdict"] == "different_boundary"
    assert latest[("resource.fork_bomb", "landlock-t1")]["verdict"] == "not_applicable"  # declared; nothing ran
    assert latest[("net.disallowed_host", "landlock-t1")]["verdict"] == "not_applicable"
    assert latest[("fs.read_env", "bwrap-t2")]["event"] == "attack_untested" and "bwrap.userns" in latest[("fs.read_env", "bwrap-t2")]["reason"]
    assert latest[("deputy.recovery_switch", "T0")]["verdict"] == "blocked"
    assert latest[("fs.read_env", "T0")]["verdict"] == "not_applicable"
    assert {(attack_id, profile) for attack_id in ONLY for profile in ("landlock-t1", "bwrap-t2", "T0")} <= set(latest)  # no silent cell
    assert ("control.proxy_allowed", "bwrap-t2") in latest  # a selected attack brings its positive controls
    ran = list(fake_runs)
    assert ("harness", "canaries") in ran and ("script", "fs.read_env") in ran and ("e2e", "deputy.recovery_switch") in ran
    assert [call for call in ran if call[1] == "resource.fork_bomb"] == []

    fake_runs.clear()
    second = run_suite(DATASET, targets(), results_dir=tmp_path, only=ONLY, scratch=tmp_path / "scratch")
    assert fake_runs == [] and second.run_id == first.run_id
    results = [r for r in second.records if r["event"] == "attack_result"]
    assert len(results) == len({r["key"] for r in results})  # nothing recorded twice
    record = latest[("fs.read_env", "landlock-t1")]
    assert set(record) >= {"key", "attack_id", "profile", "tier", "dataset_sha256", "sandbox_config_sha256", "helper_version", "verdict", "expected_verdict", "reasons", "evidence"}
    assert record["key"] == attacks.result_key("fs.read_env", "landlock-t1", DATASET.sha256, record["sandbox_config_sha256"], record["helper_version"])


def test_a_subset_brings_its_controls_and_a_failed_control_reaches_the_judge(tmp_path, fake_runs, monkeypatch):
    seen = {}

    def judge(attack, expectation, evidence, *, failed_controls=()):
        seen[attack.attack_id] = list(failed_controls)
        if attack.attack_id == "control.canaries":
            return Judgement("setup_failure", None, ("the canary tree is broken",))
        return Judgement(expectation["verdict"], False)

    monkeypatch.setattr(attacks, "judge", judge)
    result = run_suite(DATASET, targets(), results_dir=tmp_path, only=["fs.read_env"], scratch=tmp_path / "s")
    assert seen["fs.read_env"] == ["control.canaries"]  # control.workspace passed; the broken canary control is named
    assert result.latest()[("control.workspace", "landlock-t1")]["verdict"] == "passed"


def test_destructive_attacks_need_explicit_opt_in_and_a_cgroup(tmp_path, fake_runs):
    cgroup = FakeBackend("fake-cgroup-test")
    cgroup.cgroup_parent = object()
    result = run_suite(DATASET, [Target("landlock-t1-cgroup", cgroup)], results_dir=tmp_path, only=["resource.fork_bomb", "resource.cpu_quota"], scratch=tmp_path / "s")
    latest = result.latest()
    assert latest[("resource.fork_bomb", "landlock-t1-cgroup")]["event"] == "attack_untested" and "destructive" in latest[("resource.fork_bomb", "landlock-t1-cgroup")]["reason"]
    assert latest[("resource.cpu_quota", "landlock-t1-cgroup")]["verdict"] == "blocked"  # needs a cgroup, but is not destructive
    opted = run_suite(DATASET, [Target("landlock-t1-cgroup", cgroup)], results_dir=tmp_path, only=["resource.fork_bomb"], destructive=True, scratch=tmp_path / "s")
    assert opted.latest()[("resource.fork_bomb", "landlock-t1-cgroup")]["verdict"] == "blocked"


def test_changed_identity_stops_an_explicit_run_and_derives_a_new_run_id(tmp_path, fake_runs):
    first = run_suite(DATASET, targets(), results_dir=tmp_path, only=["fs.read_env"], scratch=tmp_path / "s")
    changed = targets()
    changed[0].backend.identity = BackendIdentity("fake-test", "2", SandboxTier.T1)  # a new backend build
    with pytest.raises(RunMismatch, match="targets"):
        run_suite(DATASET, changed, results_dir=tmp_path, run_id=first.run_id, only=["fs.read_env"], scratch=tmp_path / "s")
    second = run_suite(DATASET, changed, results_dir=tmp_path, only=["fs.read_env"], scratch=tmp_path / "s")
    assert second.run_id != first.run_id
    manifest = json.loads((first.directory / "manifest.json").read_text())
    assert set(manifest) >= {"run_id", "identity", "dataset", "code", "host", "probe", "targets", "options"}
    assert manifest["dataset"]["scripts"] == dict(DATASET.scripts) and manifest["host"]["kernel"] == platform.release()


def test_claims_keep_parallel_workers_from_repeating_a_unit(tmp_path, fake_runs):
    first = run_suite(DATASET, targets(), results_dir=tmp_path, only=["control.canaries"], scratch=tmp_path / "s")
    run = AttackRun(first.directory, json.loads((first.directory / "manifest.json").read_text()))
    key = attacks.result_key("fs.read_env", "landlock-t1", DATASET.sha256, targets()[0].config_sha256, attacks.helper_version())
    (run.claims / f"{key}.claim").write_text(f"{os.getpid()}\n")  # held by a live worker
    fake_runs.clear()
    run_suite(DATASET, targets(), results_dir=tmp_path, only=["control.canaries", "fs.read_env"], scratch=tmp_path / "s")
    assert ("script", "fs.read_env") not in fake_runs
    (run.claims / f"{key}.claim").write_text("999999999\n")  # left by a dead worker: taken over
    run_suite(DATASET, targets(), results_dir=tmp_path, only=["control.canaries", "fs.read_env"], scratch=tmp_path / "s")
    assert ("script", "fs.read_env") in fake_runs and not (run.claims / f"{key}.claim").exists()


def test_infrastructure_failures_are_recorded_retried_and_kept_out_of_verdicts(tmp_path, fake_runs, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("the host ran out of descriptors")

    monkeypatch.setattr(attacks, "run_script", broken)
    result = run_suite(DATASET, targets(), results_dir=tmp_path, only=["control.workspace", "fs.read_env"], scratch=tmp_path / "s")
    record = result.latest()[("fs.read_env", "landlock-t1")]
    assert record["verdict"] == "infrastructure_failure" and "out of descriptors" in record["reasons"][0]
    assert (result.directory / "evidence" / record["evidence_file"]).is_file()  # preserved before cleanup
    assert summarize(result.records)["landlock-t1"]["infrastructure_failure"] == 2
    monkeypatch.setattr(attacks, "run_script", lambda *a, **k: {"fake": True})
    retried = run_suite(DATASET, targets(), results_dir=tmp_path, only=["control.workspace", "fs.read_env"], scratch=tmp_path / "s")
    assert retried.latest()[("fs.read_env", "landlock-t1")]["verdict"] == "different_boundary"


def test_the_outer_watchdog_records_a_hung_unit_and_stops_the_run(tmp_path, fake_runs, monkeypatch):
    monkeypatch.setattr(attacks, "run_script", lambda *a, **k: time.sleep(5) or {"fake": True})
    result = run_suite(DATASET, targets(), results_dir=tmp_path, only=["control.workspace", "fs.read_env"], watchdog_s=0.2, scratch=tmp_path / "s")
    assert result.interrupted and "watchdog" in result.interrupted
    assert result.latest()[("control.workspace", "landlock-t1")]["verdict"] == "infrastructure_failure"
    assert ("fs.read_env", "landlock-t1") not in {k for k, r in result.latest().items() if r["event"] == "attack_result"}


def test_results_are_appended_whole_and_a_cut_line_is_trimmed(tmp_path, fake_runs):
    first = run_suite(DATASET, targets(), results_dir=tmp_path, only=["control.canaries"], scratch=tmp_path / "s")
    results = first.directory / "results.jsonl"
    with open(results, "a") as handle:
        handle.write('{"event": "attack_result", "key": "cut-mid-wr')
    second = run_suite(DATASET, targets(), results_dir=tmp_path, only=["control.canaries"], scratch=tmp_path / "s")
    assert all(json.loads(line) for line in results.read_text().splitlines())
    assert len(second.records) == len(first.records)


def test_only_the_evaluation_layer_imports_steadyhand_or_toolsmith():
    import re

    root = Path(attacks.__file__).resolve().parents[1]
    pattern = re.compile(r"^\s*(?:from|import)\s+(?:steadyhand|toolsmith)\b", re.MULTILINE)
    offenders = [str(path.relative_to(root)) for path in root.rglob("*.py") if "eval" not in path.relative_to(root).parts and pattern.search(path.read_text())]
    assert offenders == []


def test_every_result_verdict_is_a_known_label():
    assert set(RESULT_VERDICTS) == {"blocked", "escaped", "setup_failure", "not_applicable", "different_boundary", "infrastructure_failure", "passed"}


# Integration: the whole corpus on every profile this host can build


@pytest.fixture(scope="module")
def real_targets(tmp_path_factory):
    if not LINUX:
        pytest.skip("unsupported: the sandbox profiles need Linux")
    from bulkhead.cgroups import configured_parent
    from bulkhead.probe import run_probe

    parent = configured_parent() if INTEGRATION else None
    report = run_probe(cgroup_parent=str(parent.path) if parent is not None else None)
    profiles = list(PROFILE_TIERS) if INTEGRATION else ["landlock-t1", "T0"]
    found = attacks.discover_targets(tmp_path_factory.mktemp("bulkhead-suite-work"), report=report, cgroup_parent=parent, profiles=profiles)
    if not any(target.backend is not None for target in found):
        pytest.skip(f"unsupported: no sandbox profile is proven here ({[t.reason for t in found]})")
    return report, found


def test_integration_the_corpus_runs_on_every_proven_profile_with_no_escape(real_targets, tmp_path):
    report, found = real_targets
    destructive = INTEGRATION and os.environ.get("BULKHEAD_DESTRUCTIVE") == "1"
    result = run_suite(DATASET, found, results_dir=tmp_path / "runs", probe=report, workers=4, destructive=destructive, scratch=tmp_path / "scratch")
    latest = result.latest()
    bad = {key: (r["verdict"], r["reasons"][:3]) for key, r in latest.items() if r.get("verdict") in ("escaped", "setup_failure", "infrastructure_failure")}
    assert bad == {} and result.interrupted is None
    for target in found:
        cells = {attack.attack_id: latest[(attack.attack_id, target.profile)] for attack in DATASET.attacks}
        assert len(cells) == len(DATASET.attacks)  # every attack has a recorded cell on every profile
        if target.backend is not None:
            ran = [r for r in cells.values() if r["event"] == "attack_result" and r["verdict"] != "not_applicable"]
            assert len(ran) >= 30, target.profile
            for record in ran:
                assert record["verdict"] == record["expected_verdict"], record["attack_id"]
