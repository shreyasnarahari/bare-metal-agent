"""Bulkhead part 1: report format, tier selection, probe classification, and cleanup.

Unit tests use fake probe outcomes (tests/fixtures/bulkhead_probe_reports.json)
and fake child runners, so they pass on any host. Tests that start real
processes but change nothing (the bounded runner, the minimal child
environment) always run on Linux. Tests that depend on what the kernel offers
are opt-in with BULKHEAD_INTEGRATION=1 and skip with the probe's own reason
when a capability is absent; a skip is never a passing sandbox test.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bulkhead import capabilities as caps
from bulkhead import probe
from bulkhead.capabilities import (
    CHILD_ENV,
    HELPER,
    ChildRun,
    ProbeContext,
    classify_bwrap_failure,
    dependent,
    run_bounded,
    sanitize_detail,
)
from bulkhead.errors import ProbeReportError, UnsupportedTierError
from bulkhead.types import (
    CAPABILITY_NAMES,
    TIER_REQUIREMENTS,
    Capability,
    CapabilityStatus,
    HostInfo,
    ProbeReport,
    SandboxTier,
)

ROOT = Path(__file__).parent.parent
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "bulkhead_probe_reports.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
SCENARIOS = {scenario["name"]: scenario for scenario in FIXTURE["scenarios"]}
LINUX = sys.platform.startswith("linux")
INTEGRATION = os.environ.get("BULKHEAD_INTEGRATION") == "1"
HOST = HostInfo("Linux", "6.8.0-test", "x86_64", "3.12.0", wsl=False, container=False, root=False)
SENSITIVE = re.compile(r"(\.ssh\b|/etc/shadow|id_rsa|id_ed25519|\.aws\b|\.netrc|\.env(?![a-z_]))")


def scenario_capabilities(name, *, detail="probe detail", elapsed=0.01):
    overrides = SCENARIOS[name]["overrides"]
    records = []
    for capability in CAPABILITY_NAMES:
        if capability in overrides:
            override = overrides[capability]
            records.append(Capability(capability, override["status"], override["code"], f"{detail} ({override['code']})", {}, elapsed))
        else:
            records.append(Capability(capability, "ok", "ok", detail, FIXTURE["baseline_facts"][capability], elapsed))
    return records


def build(records, *, created_at="2026-11-09T12:00:00Z", host=HOST, python="/usr/bin/python3", helper="/opt/bulkhead/_probe_child.py", bwrap_path="/usr/bin/bwrap", cgroup_mount="/sys/fs/cgroup", cgroup_parent="/sys/fs/cgroup/delegated"):
    return ProbeReport.build(
        created_at=created_at,
        host=host,
        python=python,
        helper=helper,
        bwrap_path=bwrap_path,
        capabilities=records,
        cgroup_mount=cgroup_mount,
        cgroup_parent=cgroup_parent,
    )


def report_for(name, **options):
    return build(scenario_capabilities(name), **options)


# Tier selection from fake outcomes


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_each_scenario_selects_the_documented_tier(name):
    scenario = SCENARIOS[name]
    report = report_for(name)

    assert report.selected_tier.value == scenario["selected"]
    assert [evaluation.tier.value for evaluation in report.tiers if evaluation.satisfied] == scenario["satisfied"]
    assert [item.capability for item in report.evaluation("T2").root_causes()] == scenario["t2_root_causes"]
    # Every rejected tier records each failing requirement, with its status and code.
    for evaluation in report.tiers:
        expected = [capability for capability in TIER_REQUIREMENTS[evaluation.tier] if not report.capability(capability).ok]
        assert [item.capability for item in evaluation.missing] == expected
    assert ProbeReport.from_json(report.to_json()) == report


def test_every_capability_is_either_required_or_explicitly_informational():
    required = {name for names in TIER_REQUIREMENTS.values() for name in names}
    informational = set(CAPABILITY_NAMES) - required

    assert informational == {"bwrap.unix_socket", "userns.unshare", "landlock.tcp", "fs.openat2"}
    assert TIER_REQUIREMENTS[SandboxTier.T0] == ()


def test_require_confirms_a_tier_and_never_falls_back():
    report = report_for("wsl2_dev_host")

    assert report.require("T1").satisfied and report.require(SandboxTier.T0).satisfied
    with pytest.raises(UnsupportedTierError) as raised:
        report.require("T2")
    assert raised.value.tier is SandboxTier.T2
    assert {item.capability for item in raised.value.missing} >= {"bwrap.binary", "cgroup.delegation"}
    assert "bwrap.binary unavailable (not_found)" in str(raised.value) and "dependent" in str(raised.value)


def test_a_higher_selected_tier_does_not_imply_a_lower_one():
    report = report_for("landlock_abi_0_with_full_t2")

    assert report.selected_tier is SandboxTier.T2 and not report.supports("T1")
    with pytest.raises(UnsupportedTierError, match="landlock.abi unsupported"):
        report.require("T1")


def test_risky_execution_rejects_t0_by_default():
    proven = report_for("everything_proven")
    only_t0 = report_for("seccomp_failure")

    with pytest.raises(UnsupportedTierError, match="NOT a sandbox"):
        proven.require_risky_execution("T0")
    with pytest.raises(UnsupportedTierError):
        only_t0.require_risky_execution("T1")
    assert proven.require_risky_execution("T2").satisfied
    assert not SandboxTier.T0.is_sandbox and SandboxTier.T1.is_sandbox and SandboxTier.T2.is_sandbox
    assert SandboxTier.T0.label == "NOT a sandbox"


# Rendering


def test_t0_is_rendered_as_not_a_sandbox_and_never_as_a_higher_tier():
    text = probe.render_report(report_for("seccomp_failure"))

    selected = [line for line in text.splitlines() if line.startswith("Selected tier:")]
    assert selected == [
        "Selected tier: T0 ... NOT a sandbox. Exec and network tools are denied; "
        "only trusted in-memory tools and diagnostics may run."
    ]
    assert "T1 ... " not in text and "T2 ... " not in text
    assert "seccomp.filter failed (verification_failed)" in text


def test_t1_rendering_lists_the_missing_namespace_and_cgroup_properties():
    text = probe.render_report(report_for("wsl2_dev_host"))

    assert "Selected tier: T1 ... Landlock sandbox. Missing T2 namespace/cgroup properties: bwrap.binary" in text
    assert "rejected: bwrap.binary unavailable (not_found); cgroup.delegation unavailable (not_configured) (+6 dependent)" in text
    assert "bubblewrap at /usr/bin/bwrap is not usable" in text


def test_t2_rendering_names_the_exact_capabilities():
    text = probe.render_report(report_for("everything_proven"))

    assert "Selected tier: T2 ... Namespace sandbox: fresh ipc, mnt, net, pid, user, uts namespaces" in text
    assert "cgroup v2 (cpu, memory, pids) with cgroup.kill" in text
    assert "Selection is evidence, not permission to execute" in text


# probe_sha256


def test_hash_ignores_timestamps_paths_details_and_timings():
    first = build(scenario_capabilities("wsl2_dev_host", detail="ran in /tmp/bulkhead-probe-aaaa", elapsed=0.5))
    second = build(
        scenario_capabilities("wsl2_dev_host", detail="ran in /tmp/bulkhead-probe-bbbb", elapsed=1.25),
        created_at="2026-12-01T08:30:00Z",
        host=HostInfo("Linux", "6.18.0-other", "x86_64", "3.13.1", wsl=True, container=True, root=True),
        python="~/project/.venv/bin/python",
        helper="~/project/src/bulkhead/_probe_child.py",
        bwrap_path=None,
        cgroup_mount=None,
        cgroup_parent="/somewhere/else",
    )

    assert first.probe_sha256 == second.probe_sha256
    assert first.created_at != second.created_at


def test_hash_changes_with_semantic_evidence():
    base = report_for("wsl2_dev_host")
    records = scenario_capabilities("wsl2_dev_host")
    changed_status = [
        Capability(r.name, "failed", "verification_failed", r.detail) if r.name == "landlock.fs" else r for r in records
    ]
    changed_facts = [Capability(r.name, r.status, r.code, r.detail, {"abi": 6}) if r.name == "landlock.abi" else r for r in records]
    other_arch = HostInfo("Linux", "6.8.0", "aarch64", "3.12.0", wsl=False, container=False, root=False)

    assert build(changed_status).probe_sha256 != base.probe_sha256
    assert build(changed_facts).probe_sha256 != base.probe_sha256
    assert build(records, host=other_arch).probe_sha256 != base.probe_sha256


def test_golden_report_parses_and_pins_the_hash_scheme():
    golden = FIXTURE["golden_report"]
    report = ProbeReport.from_dict(golden)

    assert report.as_dict() == golden
    assert report.selected_tier is SandboxTier.T1
    assert report.probe_sha256 == report_for("wsl2_dev_host").probe_sha256


# Strict parsing


def mutate(path, value):
    def apply(document):
        target = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return apply


def delete(path):
    def apply(document):
        target = document
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return apply


def landlock_index():
    return CAPABILITY_NAMES.index("landlock.abi")


REFUSALS = {
    "unknown top-level field": (mutate(["extra"], 1), "unknown field"),
    "unknown capability field": (mutate(["capabilities", 0, "score"], 1), "unknown field"),
    "unknown host field": (mutate(["host", "hostname"], "box"), "unknown field"),
    "unknown nested tool field": (mutate(["tools", "bwrap", "setuid"], False), "unknown field"),
    "unknown tier field": (mutate(["tiers", "T1", "reason"], "x"), "unknown field"),
    "unknown selected tier": (mutate(["selected_tier"], "T3"), "unknown sandbox tier"),
    "unknown tier key": (mutate(["tiers", "T3"], {"label": "x", "satisfied": True, "missing": []}), "unknown field"),
    "unknown status": (mutate(["capabilities", 0, "status"], "maybe"), "unknown capability status"),
    "unknown status in tier evidence": (mutate(["tiers", "T2", "missing", 0, "status"], "maybe"), "unknown capability status"),
    "unknown capability": (mutate(["capabilities", 0, "name"], "bwrap.magic"), "unknown capability"),
    "missing capability": (lambda d: d["capabilities"].pop(), "missing capability"),
    "duplicate capability": (lambda d: d["capabilities"].append(copy.deepcopy(d["capabilities"][0])), "duplicate capability"),
    "missing field": (delete(["probe_sha256"]), "missing field"),
    "nan elapsed": (mutate(["capabilities", 0, "elapsed_s"], math.nan), "finite"),
    "infinite elapsed": (mutate(["capabilities", 0, "elapsed_s"], math.inf), "finite"),
    "float fact": (mutate(["capabilities", landlock_index(), "facts", "abi"], 7.0), "facts hold"),
    "bool as integer": (mutate(["host", "wsl"], 1), "true or false"),
    "integer schema as bool": (mutate(["schema_version"], True), "schema version"),
    "future schema": (mutate(["schema_version"], 2), "schema version"),
    "ok code on a failure": (mutate(["capabilities", 0, "code"], "ok"), "exactly when"),
    "multi-line detail": (mutate(["capabilities", 0, "detail"], "line one\nline two"), "control characters"),
    "overlong detail": (mutate(["capabilities", 0, "detail"], "x" * 201), "longer than"),
    "claimed T2": (mutate(["selected_tier"], "T2"), "inconsistent tier evidence"),
    "claimed T2 satisfied": (mutate(["tiers", "T2", "satisfied"], True), "inconsistent tier evidence"),
    "dropped a missing requirement": (lambda d: d["tiers"]["T2"]["missing"].pop(), "inconsistent tier evidence"),
    "tampered hash": (mutate(["probe_sha256"], "0" * 64), "does not match"),
    "malformed hash": (mutate(["probe_sha256"], "abc"), "64 lowercase hex"),
    "edited landlock summary": (mutate(["landlock", "abi"], 8), "contradicts"),
    "edited cgroup summary": (mutate(["cgroup", "delegated"], True), "contradicts"),
    "edited tool version": (mutate(["tools", "libseccomp", "version"], "9.9.9"), "contradicts"),
    "reordered records": (lambda d: d["capabilities"].reverse(), "canonical order"),
    "bad timestamp": (mutate(["created_at"], "yesterday"), "UTC timestamp"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_parsing_refuses_malformed_or_inconsistent_reports(case):
    document = report_for("wsl2_dev_host").as_dict()
    change, message = REFUSALS[case]
    change(document)

    with pytest.raises(ProbeReportError, match=message):
        ProbeReport.from_dict(document)


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version": NaN}',
        '{"schema_version": Infinity}',
        '{"schema_version": 1, "schema_version": 1}',
        '{"schema_version": 1e999}',
        "not json",
    ],
)
def test_json_text_with_non_finite_numbers_or_duplicate_keys_is_refused(text):
    with pytest.raises(ProbeReportError):
        ProbeReport.from_json(text)


# Writing and loading


def test_write_report_is_atomic_and_round_trips(tmp_path):
    report = report_for("wsl2_dev_host")
    path = probe.write_report(report, tmp_path / "nested" / "probe.json")

    assert probe.load_report(path) == report
    assert [entry.name for entry in path.parent.iterdir()] == ["probe.json"]  # no temporary file left


def test_write_report_refuses_a_report_containing_a_secret(tmp_path):
    records = [
        Capability(r.name, r.status, r.code, "token hf-secret-value leaked") if r.name == "fs.openat2" else r
        for r in scenario_capabilities("wsl2_dev_host")
    ]

    with pytest.raises(ValueError, match="secret"):
        probe.write_report(build(records), tmp_path / "probe.json", secrets=("hf-secret-value", ""))
    assert not (tmp_path / "probe.json").exists()


# CLI (run_probe faked so the exit codes are host-independent)


@pytest.fixture
def fake_probe(monkeypatch):
    calls = []

    def fake_run_probe(**options):
        calls.append(options)
        return report_for("wsl2_dev_host")

    monkeypatch.setattr(probe, "run_probe", fake_run_probe)
    return calls


def test_cli_require_tier_mismatch_exits_nonzero_without_fallback(fake_probe, capsys):
    assert probe.main(["--require-tier", "T2"]) == probe.EXIT_TIER_UNSUPPORTED

    out, err = capsys.readouterr()
    assert "Selected tier: T1 ... Landlock sandbox" in out
    assert "Required tier not met: T2 (Namespace sandbox)" in err and "bwrap.binary" in err


def test_cli_confirms_a_supported_tier_and_writes_json(fake_probe, capsys, tmp_path):
    assert probe.main(["--require-tier", "T1", "--json", str(tmp_path / "probe.json")]) == 0

    out, _ = capsys.readouterr()
    assert "Required tier T1 (Landlock sandbox) is supported." in out
    assert probe.load_report(tmp_path / "probe.json").selected_tier is SandboxTier.T1


def test_cli_takes_the_cgroup_parent_only_from_explicit_configuration(fake_probe, monkeypatch):
    monkeypatch.delenv(caps.CGROUP_PARENT_ENV, raising=False)
    probe.main([])
    monkeypatch.setenv(caps.CGROUP_PARENT_ENV, "/sys/fs/cgroup/delegated")
    probe.main(["--bwrap", "/opt/bwrap"])

    assert fake_probe[0]["cgroup_parent"] is None and fake_probe[0]["bwrap"] is None
    assert fake_probe[1]["cgroup_parent"] == "/sys/fs/cgroup/delegated" and fake_probe[1]["bwrap"] == "/opt/bwrap"


@pytest.mark.parametrize("argv", [["--require-tier", "T3"], ["--timeout", "nan"], ["--timeout", "0"]])
def test_cli_rejects_unknown_tiers_and_bad_timeouts(fake_probe, argv, capsys):
    with pytest.raises(SystemExit) as raised:
        probe.main(argv)
    assert raised.value.code == 2


# Probe classification with fake child runners


def child_output(results):
    return ChildRun(0, json.dumps({"results": results}) + "\n", "", False, 0.01)


def ok_result(detail="fine", **facts):
    return {"status": "ok", "code": "ok", "detail": detail, "facts": facts}


class FakeRunner:
    """Answers each helper operation (or ``bwrap --version``) from a table; records every argv."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, argv, *, timeout_s):
        self.calls.append(list(argv))
        operation = "bwrap --version" if argv[1:] == ["--version"] else argv[argv.index(str(HELPER)) + 1]
        answer = self.answers[operation]
        return answer(argv) if callable(answer) else answer

    def operations(self):
        return ["bwrap --version" if argv[1:] == ["--version"] else argv[argv.index(str(HELPER)) + 1] for argv in self.calls]


def context(tmp_path, answers=None, **options):
    runner = FakeRunner(answers or {})
    return ProbeContext(workdir=tmp_path, runner=runner, tcp_port=1, unix_socket=tmp_path / "proxy.sock", **options), runner


def test_missing_bwrap_is_unavailable_and_its_dependents_are_not_run(tmp_path):
    ctx, runner = context(tmp_path, bwrap=None)

    binary = caps.probe_bwrap_binary(ctx)
    namespaces = caps.probe_bwrap_namespaces(ctx, binary)
    mounts = caps.probe_bwrap_mounts(ctx, namespaces)

    assert (binary.status, binary.code) == (CapabilityStatus.UNAVAILABLE, "not_found")
    assert "never installs" in binary.detail
    assert (namespaces.code, mounts.code) == ("dependency_unavailable", "dependency_unavailable")
    assert runner.calls == []


def test_bwrap_version_is_recorded(tmp_path):
    fake_bwrap = tmp_path / "bwrap"
    fake_bwrap.write_text("#!/bin/sh\n")
    fake_bwrap.chmod(0o755)
    ctx, _ = context(tmp_path, {"bwrap --version": ChildRun(0, "bubblewrap 0.9.0\n", "", False, 0.01)}, bwrap=str(fake_bwrap))

    binary = caps.probe_bwrap_binary(ctx)

    assert binary.ok and binary.facts == {"version": "0.9.0", "setuid": False}


@pytest.mark.parametrize(
    ("stderr", "sysctls", "status", "code"),
    [
        ("bwrap: setting up uid map: Permission denied\n", {}, "denied", "eperm"),
        ("bwrap: Creating new namespace failed: Operation not permitted\n", {}, "denied", "eperm"),
        ("bwrap: setting up uid map: Permission denied\n", {"kernel/apparmor_restrict_unprivileged_userns": "1"}, "denied", "apparmor_userns_restricted"),
        ("bwrap: No permissions to creating new namespace, likely because the kernel does not allow non-privileged user namespaces.\n", {"user/max_user_namespaces": "0"}, "denied", "userns_disabled"),
        ("bwrap: No permissions to creating new namespace\n", {"kernel/unprivileged_userns_clone": "0"}, "denied", "userns_disabled"),
        ("bwrap: Unknown option --unshare-cgroup-try\n", {}, "unsupported", "unknown_option"),
        ("bwrap: Can't mount proc on /newroot/proc: Device or resource busy\n", {}, "failed", "bwrap_failed"),
        ("", {}, "failed", "bwrap_failed"),
    ],
)
def test_bwrap_failures_are_told_apart(stderr, sysctls, status, code):
    got_status, got_code, detail = classify_bwrap_failure(stderr, sysctls)

    assert (got_status.value, got_code) == (status, code)
    assert "\n" not in detail


def test_bwrap_namespace_probe_reports_an_apparmor_refusal(tmp_path):
    proc = tmp_path / "proc"
    (proc / "sys" / "kernel").mkdir(parents=True)
    (proc / "sys" / "kernel" / "apparmor_restrict_unprivileged_userns").write_text("1\n")
    (proc / "self").symlink_to("/proc/self")  # real namespace links, fake sysctls
    failure = ChildRun(1, "", "bwrap: setting up uid map: Permission denied\n", False, 0.01)
    ctx, runner = context(tmp_path, {"bwrap-namespaces": failure}, bwrap="/usr/bin/bwrap", proc_root=proc)
    binary = Capability("bwrap.binary", "ok", "ok", "bubblewrap 0.9.0", {"version": "0.9.0"})

    record = caps.probe_bwrap_namespaces(ctx, binary)

    assert (record.status, record.code) == (CapabilityStatus.DENIED, "apparmor_userns_restricted")
    argv = runner.calls[0]
    assert argv[0] == "/usr/bin/bwrap" and "--unshare-user" in argv and "--die-with-parent" in argv
    assert argv[argv.index("--") + 1 :][:2] == [ctx.python, "-I"]  # the helper runs isolated


def test_a_timed_out_child_is_a_failed_record(tmp_path):
    ctx, _ = context(tmp_path, {"no-new-privs": ChildRun(None, "", "", True, 10.0)})

    record = caps.probe_no_new_privs(ctx)

    assert (record.status, record.code) == (CapabilityStatus.FAILED, "timeout")
    assert record.elapsed_s == 10.0


@pytest.mark.parametrize(
    "run",
    [
        ChildRun(0, "everything is fine\n", "", False, 0.01),
        ChildRun(0, json.dumps({"results": {"rlimits": ok_result()}}), "", False, 0.01),  # wrong record
        ChildRun(0, json.dumps({"results": {"no_new_privs": {"status": "ok", "code": "ok"}}}), "", False, 0.01),
        ChildRun(0, json.dumps({"results": {"no_new_privs": {**ok_result(), "status": "great"}}}), "", False, 0.01),
        ChildRun(-9, "", "Killed\n", False, 0.01),
    ],
)
def test_garbled_or_crashed_child_output_is_never_ok(tmp_path, run):
    ctx, _ = context(tmp_path, {"no-new-privs": run})

    record = caps.probe_no_new_privs(ctx)

    assert record.status is CapabilityStatus.FAILED and record.code in ("no_result", "child_failed")


def test_child_details_are_sanitized(tmp_path):
    home = os.path.expanduser("~")
    detail = f"opened {tmp_path}/landlock/denied/canary and {home}/project\nwith\x1b[31m color " + "x" * 300
    ctx, _ = context(tmp_path, {"no-new-privs": child_output({"no_new_privs": ok_result(detail)})})

    record = caps.probe_no_new_privs(ctx)

    assert str(tmp_path) not in record.detail and "<tmp>/landlock/denied/canary" in record.detail
    assert (home in ("", "/")) or (home not in record.detail and "~/project" in record.detail)
    assert "\n" not in record.detail and "\x1b" not in record.detail and len(record.detail) <= 200


def test_landlock_abi_0_skips_the_enforcement_probes(tmp_path):
    abi0 = {"status": "unsupported", "code": "abi_0", "detail": "Landlock reported ABI 0", "facts": {}}
    ctx, runner = context(tmp_path, {"landlock-abi": child_output({"landlock.abi": abi0})})

    records = {record.name: record for record in caps.probe_landlock(ctx)}

    assert records["landlock.abi"].status is CapabilityStatus.UNSUPPORTED
    assert records["landlock.fs"].code == records["landlock.tcp"].code == "dependency_unavailable"
    assert runner.operations() == ["landlock-abi"]


def test_landlock_tcp_needs_abi_4_and_is_not_attempted_below_it(tmp_path):
    answers = {
        "landlock-abi": child_output({"landlock.abi": ok_result("Landlock ABI 3", abi=3)}),
        "landlock-fs": child_output({"landlock.fs": ok_result("enforced", abi=3, tested=["read_denied"])}),
    }
    ctx, runner = context(tmp_path, answers)

    records = {record.name: record for record in caps.probe_landlock(ctx)}

    assert records["landlock.fs"].ok
    assert (records["landlock.tcp"].status, records["landlock.tcp"].code) == (CapabilityStatus.UNSUPPORTED, "abi_too_old")
    assert "landlock-tcp" not in runner.operations()


def test_seccomp_failure_and_missing_libseccomp_are_recorded(tmp_path):
    failed = {"status": "failed", "code": "verification_failed", "detail": "filter had no effect", "facts": {}}
    answers = {
        "seccomp-filter": child_output({"seccomp.filter": failed}),
        "seccomp-socket-deny": child_output({"seccomp.socket_deny": ok_result()}),
    }
    ctx, _ = context(tmp_path, answers)
    library = Capability("libseccomp.library", "ok", "ok", "libseccomp 2.5.5", {"version": "2.5.5"})
    missing = Capability("libseccomp.library", "unavailable", "not_found", "libseccomp not found")

    filter_record, _ = caps.probe_seccomp(ctx, library)
    skipped = caps.probe_seccomp(ctx, missing)

    assert (filter_record.status, filter_record.code) == (CapabilityStatus.FAILED, "verification_failed")
    assert [record.code for record in skipped] == ["dependency_unavailable", "dependency_unavailable"]


def test_an_unexpected_probe_exception_becomes_a_failed_record(tmp_path, monkeypatch):
    def explode(ctx):
        raise RuntimeError(f"boom in {ctx.workdir}")

    monkeypatch.setattr(caps, "PROBES", ((("rlimits",), lambda ctx, done: explode(ctx)),))
    ctx, _ = context(tmp_path)

    [record] = caps.collect_capabilities(ctx)

    assert (record.status, record.code) == (CapabilityStatus.FAILED, "probe_error")
    assert "boom in <tmp>" in record.detail


def test_dependent_records_name_their_prerequisite():
    prerequisite = Capability("cgroup.delegation", "denied", "not_writable", "read-only")

    record = dependent("cgroup.leaf", prerequisite)

    assert record.status is CapabilityStatus.UNAVAILABLE
    assert record.detail == "requires cgroup.delegation, which is denied (not_writable)"


# cgroup v2 against a fake tree (a plain directory must never pass as cgroupfs)


def fake_cgroupfs(tmp_path, *, enabled="cpu memory pids", mountinfo=None):
    mount = tmp_path / "cgroupfs"
    parent = mount / "delegated"
    parent.mkdir(parents=True)
    (mount / "cgroup.controllers").write_text("cpu io memory pids\n")
    (parent / "cgroup.controllers").write_text("cpu io memory pids\n")
    (parent / "cgroup.subtree_control").write_text(enabled + "\n")
    proc_self = tmp_path / "proc" / "self"
    proc_self.mkdir(parents=True)
    line = f"30 25 0:26 / {mount} rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw,nsdelegate\n"
    (proc_self / "mountinfo").write_text(line if mountinfo is None else mountinfo)
    return mount, parent, tmp_path / "proc"


def test_cgroup_mount_needs_a_cgroup2_entry_in_mountinfo(tmp_path):
    _, _, proc = fake_cgroupfs(tmp_path, mountinfo="22 1 0:20 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu\n")
    ctx, _ = context(tmp_path, proc_root=proc)

    record = caps.probe_cgroup_mount(ctx)

    assert (record.status, record.code) == (CapabilityStatus.UNAVAILABLE, "not_mounted")
    assert "only cgroup v1" in record.detail and ctx.cgroup_mount is None


def test_cgroup_mount_without_controllers_file_is_not_trusted(tmp_path):
    empty = tmp_path / "not-cgroupfs"
    empty.mkdir()
    _, _, proc = fake_cgroupfs(tmp_path, mountinfo=f"30 25 0:26 / {empty} rw - cgroup2 cgroup2 rw\n")
    ctx, _ = context(tmp_path, proc_root=proc)

    assert caps.probe_cgroup_mount(ctx).code == "not_readable"


def test_cgroup_needs_an_explicitly_configured_parent(tmp_path):
    _, _, proc = fake_cgroupfs(tmp_path)
    ctx, _ = context(tmp_path, proc_root=proc)

    mount = caps.probe_cgroup_mount(ctx)
    records = caps.probe_cgroup_tree(ctx, mount)

    assert mount.ok and mount.facts == {"nsdelegate": True}
    assert (records[0].status, records[0].code) == (CapabilityStatus.UNAVAILABLE, "not_configured")
    assert [record.code for record in records[1:]] == ["dependency_unavailable"] * 3


def test_the_cgroup_root_is_refused_even_when_configured(tmp_path):
    mount, _, proc = fake_cgroupfs(tmp_path)
    ctx, _ = context(tmp_path, proc_root=proc, cgroup_parent=mount)

    records = caps.probe_cgroup_tree(ctx, caps.probe_cgroup_mount(ctx))

    assert (records[0].status, records[0].code) == (CapabilityStatus.DENIED, "root_cgroup_refused")


def test_a_parent_outside_the_cgroup2_mount_is_refused(tmp_path):
    _, _, proc = fake_cgroupfs(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    ctx, _ = context(tmp_path, proc_root=proc, cgroup_parent=outside)

    records = caps.probe_cgroup_tree(ctx, caps.probe_cgroup_mount(ctx))

    assert records[0].code == "outside_cgroup2_mount"


@pytest.mark.skipif(LINUX and os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_read_only_cgroup_parent_is_denied(tmp_path):
    _, parent, proc = fake_cgroupfs(tmp_path)
    parent.chmod(0o555)
    try:
        ctx, _ = context(tmp_path, proc_root=proc, cgroup_parent=parent)
        records = caps.probe_cgroup_tree(ctx, caps.probe_cgroup_mount(ctx))
    finally:
        parent.chmod(0o755)

    assert (records[0].status, records[0].code) == (CapabilityStatus.DENIED, "not_writable")
    assert report_for("read_only_cgroup").selected_tier is SandboxTier.T1


def test_a_plain_directory_is_not_a_cgroup_and_no_leaf_is_left_behind(tmp_path):
    _, parent, proc = fake_cgroupfs(tmp_path)
    ctx, _ = context(tmp_path, proc_root=proc, cgroup_parent=parent)

    delegation, leaf, limits, kill = caps.probe_cgroup_tree(ctx, caps.probe_cgroup_mount(ctx))

    assert delegation.ok and delegation.facts["enabled"] == ["cpu", "memory", "pids"]
    assert (leaf.status, leaf.code) == (CapabilityStatus.FAILED, "not_a_cgroup")
    assert limits.code == kill.code == "dependency_unavailable"
    assert sorted(entry.name for entry in parent.iterdir()) == ["cgroup.controllers", "cgroup.subtree_control"]


# Real processes that change nothing on the host


def _dead(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:  # a zombie waiting for a reaper that is not us is already dead
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].startswith("Z")
    except (OSError, IndexError):
        return True


@pytest.mark.skipif(not LINUX, reason="process groups and /proc are Linux-specific")
def test_bounded_runner_timeout_kills_the_child_and_grandchild(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import os, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        f"    open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "time.sleep(60)\n"
    )
    started = time.monotonic()

    run = run_bounded([sys.executable, "-I", "-c", script], timeout_s=1.0)

    assert run.timed_out and run.returncode is None and time.monotonic() - started < 15
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    grandchild = int(pid_file.read_text())
    while not _dead(grandchild) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _dead(grandchild)


@pytest.mark.skipif(not LINUX, reason="Linux-only probe runner")
def test_children_get_only_the_minimal_environment(monkeypatch):
    monkeypatch.setenv("BULKHEAD_TEST_SECRET", "canary-value-not-a-real-secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")

    run = run_bounded([sys.executable, "-I", "-c", "import json, os; print(json.dumps(sorted(os.environ)))"], timeout_s=10)

    assert json.loads(run.stdout) == sorted(CHILD_ENV)
    assert "canary-value" not in run.stdout


# The probe layer's boundaries


def test_the_probe_layer_imports_neither_toolsmith_nor_steadyhand():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import bulkhead, bulkhead.probe, bulkhead.capabilities, bulkhead.types, bulkhead.errors;"
        "print([m for m in sys.modules if m.split('.')[0] in ('toolsmith', 'steadyhand')])"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "src")], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"


def test_the_child_helper_imports_only_the_standard_library():
    tree = ast.parse(HELPER.read_text(encoding="utf-8"))
    modules = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}

    assert modules - {"__future__"} <= set(sys.stdlib_module_names)


def test_probe_sources_name_no_sensitive_paths_and_read_one_environment_variable():
    sources = {path.name: path.read_text(encoding="utf-8") for path in (ROOT / "src" / "bulkhead").glob("*.py")}

    assert {name: SENSITIVE.findall(text) for name, text in sources.items() if SENSITIVE.search(text)} == {}
    environ_uses = [(name, line.strip()) for name, text in sources.items() for line in text.splitlines() if "os.environ" in line]
    # One variable only: the explicitly configured cgroup parent (read by the probe CLI and by the backends).
    assert sorted(environ_uses) == [("cgroups.py", "value = os.environ.get(CGROUP_PARENT_ENV)"), ("probe.py", "default=os.environ.get(CGROUP_PARENT_ENV) or None,")]
    assert "sudo" not in "".join(sources.values()).replace("never uses sudo", "").replace("``sudo``", "")


def test_sanitize_detail_caps_length_and_removes_control_characters():
    text = sanitize_detail("a\tb\nc\x00d " + "y" * 400, {"/tmp/x": "<tmp>"})

    assert text.startswith("a b c d ") and len(text) == 200 and text.endswith("…")


# Opt-in integration: a real probe on this host


integration = pytest.mark.skipif(
    not (INTEGRATION and LINUX), reason="set BULKHEAD_INTEGRATION=1 on Linux to run the real probe"
)


@pytest.fixture(scope="module")
def real_probe():
    return probe.run_probe()


@integration
def test_real_probe_writes_a_valid_secret_free_report(tmp_path, monkeypatch):
    monkeypatch.setenv("BULKHEAD_TEST_SECRET", "canary-value-not-a-real-secret")
    before = {entry.name for entry in Path("/tmp").iterdir() if entry.name.startswith("bulkhead-probe-")}

    report = probe.run_probe()
    path = probe.write_report(report, tmp_path / "probe.json", secrets=("canary-value-not-a-real-secret",))

    assert probe.load_report(path) == report
    assert {entry.name for entry in Path("/tmp").iterdir() if entry.name.startswith("bulkhead-probe-")} <= before
    assert not SENSITIVE.search(path.read_text())


@integration
def test_real_probe_hash_is_stable_across_runs(real_probe):
    assert probe.run_probe().probe_sha256 == real_probe.probe_sha256


@integration
def test_real_probe_opens_no_sensitive_file_and_hands_children_only_its_own_paths(monkeypatch):
    opened, argvs = [], []

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(args[0]))

    def recording_runner(argv, *, timeout_s):
        argvs.append(list(argv))
        return run_bounded(argv, timeout_s=timeout_s)

    sys.addaudithook(lambda event, args: audit(event, args) if recording else None)
    recording = True
    try:
        probe.run_probe(runner=recording_runner)
    finally:
        recording = False

    assert [path for path in opened if SENSITIVE.search(path)] == []
    for argv in argvs:
        if str(HELPER) in argv and argv[-1].startswith("{"):
            paths = [value for value in json.loads(argv[-1]).values() if isinstance(value, str) and value.startswith("/")]
            assert all("/bulkhead-probe-" in path or path.startswith("/tmp/") for path in paths), paths


@pytest.mark.parametrize(
    "name", sorted({name for names in TIER_REQUIREMENTS.values() for name in names}) + ["landlock.tcp", "fs.openat2"]
)
@integration
def test_real_capability_is_proven_or_skipped_with_its_reason(real_probe, name):
    record = real_probe.capability(name)
    if not record.ok:
        pytest.skip(f"{name} is {record.status.value} ({record.code}) on this host: {record.detail}")
    assert record.code == "ok"


@integration
def test_real_cgroup_leaf_is_removed_under_a_configured_parent():
    parent = os.environ.get("BULKHEAD_TEST_CGROUP_PARENT")
    if not parent:
        pytest.skip("set BULKHEAD_TEST_CGROUP_PARENT to a delegated cgroup v2 parent to test leaf cleanup")

    report = probe.run_probe(cgroup_parent=parent)

    assert not list(Path(parent).glob("bulkhead-probe-*"))
    if not report.capability("cgroup.kill").ok:
        pytest.skip(f"cgroup.kill not proven: {report.capability('cgroup.kill').detail}")
