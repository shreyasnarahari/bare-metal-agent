"""Bulkhead part 3: replay that re-derives every decision but never repeats an external effect."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from bulkhead.manifest import Egress, Sandbox, SecuritySpec
from bulkhead.replay import (
    AUDIT_FIELDS,
    BackendIdentity,
    ExecutionOutcome,
    ExecutionRequest,
    RecordedExecutionBackend,
    RecordedExecutionMismatch,
    requires_backend,
)
from bulkhead.taint import Sink
from bulkhead.types import SandboxTier
from steadyhand import EpisodeStatus, SteadyhandClient, build_registry, initial_messages, load_fixture
from steadyhand.journal import JournalFormatError, ReplayMismatch, comparable, manifest_path_for, read_journal
from steadyhand.manifest import load_manifest
from steadyhand.replay import live_replay_report, registry_for, replay_episode, run_claimed_episode
from steadyhand.security import SecurityConfig, approval_lineage, approval_requests
from toolsmith.protocols import QWEN3_PROFILE

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "bulkhead_episodes"
LEGACY = ROOT / "tests" / "fixtures" / "steadyhand_episodes"
REFUND = {"order_id": "ord_1001", "amount_inr": 120, "reason": "Dented lid"}


def fixture(label):
    for path in sorted(FIXTURES.glob("ep_*.jsonl")):
        if load_manifest(manifest_path_for(path)).run_label == label:
            return path
    raise LookupError(label)


def child_fixture():
    return next(path for path in sorted(FIXTURES.glob("ep_*.jsonl")) if (load_manifest(manifest_path_for(path)).run_label or "").startswith("approved-"))


def copy_episode(path, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for source in (path, manifest_path_for(path)):
        shutil.copy(source, directory / source.name)
    return directory / path.name


def rewrite(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records))


@pytest.fixture
def no_side_effects(monkeypatch):
    """Fail the test if anything spawns a process or opens a network connection."""

    def refuse(*args, **kwargs):
        raise AssertionError("replay tried to spawn a process or open a connection")

    monkeypatch.setattr(subprocess.Popen, "__init__", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# Which tools replay may re-execute


def test_only_in_memory_tools_are_re_executed_by_replay():
    sandboxed = SecuritySpec(sandbox=Sandbox(net=Egress(allow=("pypi.org:443",))), sinks=(Sink.NETWORK,))
    store_write = SecuritySpec(sandbox=None, sinks=(Sink.EXTERNAL_EFFECT,))

    assert requires_backend(None, exec_effect=True) and requires_backend(store_write, exec_effect=True)
    assert requires_backend(sandboxed, exec_effect=False)
    assert not requires_backend(store_write, exec_effect=False) and not requires_backend(None, exec_effect=False)
    for name in build_registry().names():
        assert not requires_backend(build_registry().security(name), exec_effect=False)


def test_backend_identity_is_strict_and_never_t0():
    identity = BackendIdentity("bwrap", "0.9.0", SandboxTier.T2, "b" * 64)

    assert BackendIdentity.from_dict(identity.as_dict()) == identity
    for bad in (("bad name", "1", "T2", None), ("bwrap", "", "T2", None), ("bwrap", "1", "T0", None), ("bwrap", "1", "T2", "xyz")):
        with pytest.raises(ValueError):
            BackendIdentity(*bad)


def recorded_run(**changes):
    record = {
        "event": "sandbox_execution", "seq": 4, "tool_call_id": "call_shell", "via": "model", "tool": "run_shell",
        "result": {"ok": True, "handler_called": True, "value": {"stdout": "x"}},
        **{field: None for field in AUDIT_FIELDS},
    }
    record.update(changes)
    return record


def test_the_recorded_backend_serves_runs_in_order_and_checks_each_request():
    spec = SecuritySpec(sandbox=Sandbox(), sinks=())
    request = ExecutionRequest("ep_x", "call_shell", "model", "run_shell", "c" * 64, spec)
    good = recorded_run(config_sha256=request.config_sha256, request_sha256=request.sha256)
    backend = RecordedExecutionBackend([{"event": "tool_result"}, good], BackendIdentity("bwrap", "1", "T2"))

    outcome = backend.execute(request, {"command": "never used"})
    assert outcome.result == good["result"] and backend.remaining == 0
    with pytest.raises(RecordedExecutionMismatch) as raised:
        backend.execute(request, {})
    assert raised.value.field == "sandbox_execution"
    for field, value in [("request_sha256", "0" * 64), ("tool_call_id", "call_other"), ("via", "policy"), ("config_sha256", "1" * 64), ("tool", "run_python")]:
        changed = RecordedExecutionBackend([{**good, field: value}], BackendIdentity("bwrap", "1", "T2"))
        with pytest.raises(RecordedExecutionMismatch) as raised:
            changed.execute(request, {})
        assert (raised.value.field, raised.value.seq) == (field, 4)


# Golden secured episodes


def test_every_golden_secured_episode_replays_without_side_effects(no_side_effects):
    statuses = {}
    for path in sorted(FIXTURES.glob("ep_*.jsonl")):
        manifest = load_manifest(manifest_path_for(path))
        statuses[manifest.run_label if not manifest.parent_episode_id else "approval_child"] = replay_episode(path).status.value
    assert statuses == {
        "clean": "done", "tainted": "escalated", "same_turn_taint": "escalated", "denied": "done",
        "approval_parent": "escalated", "approval_child": "done", "sandboxed_exec": "escalated",
    }


def test_the_sandboxed_golden_holds_its_run_and_replays_it_from_the_recording(no_side_effects):
    path = fixture("sandboxed_exec")
    records = read_journal(path)
    manifest = load_manifest(manifest_path_for(path))
    [run] = [record for record in records if record["event"] == "sandbox_execution"]

    assert manifest.backend == {"name": "scripted-test", "version": "1", "tier": "T2", "probe_sha256": None}
    assert run["result"]["ok"] and "IGNORE PREVIOUS INSTRUCTIONS" in run["result"]["value"]["stdout"]["content"]
    assert registry_for(manifest.schema_sha256).names()[-2:] == ("run_python", "run_shell")
    assert manifest.security_specs["run_shell"]["sandbox"]["net"]["allow"] == []
    assert replay_episode(path).status is EpisodeStatus.ESCALATED


def test_old_golden_episodes_replay_byte_for_byte_without_new_events(no_side_effects):
    for path in sorted(LEGACY.glob("ep_*.jsonl")):
        before = path.read_bytes()
        replay_episode(path)
        assert path.read_bytes() == before
        assert b"policy_decision" not in before and b"taint_transition" not in before


# Editing any deterministic field is caught at that field


def tampered(tmp_path, label, edit):
    path = copy_episode(fixture(label), tmp_path / "copy")
    records = read_journal(path)
    seq = edit(records)
    rewrite(path, records)
    return path, seq


def first(records, event, **match):
    return next(r for r in records if r["event"] == event and all(r.get(k) == v for k, v in match.items()))


def test_a_consistently_edited_verdict_is_a_replay_mismatch(tmp_path):
    def edit(records):
        decision = first(records, "policy_decision", tool="issue_refund")
        decision.update(verdict="deny", rule_id="task.deny.tool", reason_code="tool_denied")
        return decision["seq"]

    path, seq = tampered(tmp_path, "clean", edit)
    with pytest.raises(ReplayMismatch) as raised:
        replay_episode(path)
    assert raised.value.seq == seq and raised.value.field in ("/reason_code", "/rule_id", "/verdict")


def test_an_inconsistently_edited_verdict_is_refused_on_read(tmp_path):
    def edit(records):
        decision = first(records, "policy_decision", tool="issue_refund")
        decision["verdict"] = "deny"
        return decision["seq"]

    path, seq = tampered(tmp_path, "clean", edit)
    with pytest.raises(JournalFormatError) as raised:
        replay_episode(path)
    assert (raised.value.seq, raised.value.field) == (seq, "verdict")


@pytest.mark.parametrize(
    ("label", "event", "match", "field", "value"),
    [
        ("tainted", "policy_decision", {"tool": "issue_refund"}, "taint_before", "trusted"),
        ("tainted", "policy_decision", {"tool": "issue_refund"}, "normalized_request_sha256", "0" * 64),
        ("tainted", "taint_transition", {}, "source_seq", 2),
        ("tainted", "approval_requested", {}, "request_sha256", "0" * 64),
        ("sandboxed_exec", "tool_result", {"tool": "run_shell"}, "content", '{"ok": true, "value": "edited"}'),
        ("clean", "tool_result", {"tool": "issue_refund"}, "content", '{"ok": true, "value": {}}'),
    ],
)
def test_editing_a_deterministic_security_field_names_it(tmp_path, label, event, match, field, value):
    def edit(records):
        record = first(records, event, **match)
        record[field] = value
        return record["seq"]

    path, seq = tampered(tmp_path, label, edit)
    with pytest.raises(ReplayMismatch) as raised:
        replay_episode(path)
    assert raised.value.seq == seq and raised.value.field == f"/{field}"


def test_a_dropped_taint_transition_is_caught(tmp_path):
    def edit(records):
        index = next(i for i, r in enumerate(records) if r["event"] == "taint_transition")
        dropped = records.pop(index)
        for later in records[index:]:
            later["seq"] -= 1
        return dropped["seq"]

    path, seq = tampered(tmp_path, "tainted", edit)
    with pytest.raises(ReplayMismatch) as raised:
        replay_episode(path)
    assert raised.value.seq == seq


def test_an_edited_request_hash_is_caught_at_the_recorded_run(tmp_path, no_side_effects):
    def edit(records):
        run = first(records, "sandbox_execution")
        run["request_sha256"] = "0" * 64
        return run["seq"]

    path, seq = tampered(tmp_path, "sandboxed_exec", edit)
    with pytest.raises(ReplayMismatch) as raised:
        replay_episode(path)
    assert (raised.value.seq, raised.value.field) == (seq, "request_sha256")


def test_volatile_measurements_may_differ(tmp_path, no_side_effects):
    def edit(records):
        run = first(records, "sandbox_execution")
        run.update(elapsed_s=9.75, resource_usage={"cpu_s": 3.5, "peak_memory_kb": 99999, "pids_peak": 7})
        return run["seq"]

    path, _ = tampered(tmp_path, "sandboxed_exec", edit)
    assert replay_episode(path).status is EpisodeStatus.ESCALATED


# Approvals keep their linkage and never rewrite the parent


def test_the_approved_child_reconstructs_its_link_to_the_parent():
    parent, child = fixture("approval_parent"), child_fixture()
    before = parent.read_bytes(), manifest_path_for(parent).read_bytes()

    lineage = approval_lineage(child)
    [request] = approval_requests(parent)

    assert approval_lineage(parent) is None
    assert lineage.parent_episode_id == read_journal(parent)[0]["episode_id"]
    assert lineage.run_label == f"approved-{lineage.grant.approval_id}"
    assert (lineage.grant.tool, lineage.grant.arguments_sha256) == (request.tool, request.arguments_sha256)
    assert lineage.grant.normalized_grant == request.normalized_grant
    replay_episode(child)
    assert (parent.read_bytes(), manifest_path_for(parent).read_bytes()) == before


# Explicit live replay: a separate report, never the original journal


class EchoBackend:
    """Returns the recorded run's outcome with changes, as a live backend re-execution would."""

    identity = BackendIdentity("live-test", "2", SandboxTier.T2, None)

    def __init__(self, recorded, gate=None, **changes):
        self.recorded = recorded
        self.gate = gate
        self.changes = changes

    def execute(self, request, arguments):
        audit = {field: self.recorded[field] for field in AUDIT_FIELDS}
        audit.update(self.changes)
        return ExecutionOutcome(self.recorded["result"], audit, gate=self.gate)


def test_live_replay_writes_only_its_report_and_tolerates_volatile_differences(tmp_path):
    path = fixture("sandboxed_exec")
    run = first(read_journal(path), "sandbox_execution")
    gate = first(read_journal(path), "output_gate")["streams"]
    before = path.read_bytes(), manifest_path_for(path).read_bytes()

    same = live_replay_report(path, EchoBackend(run, gate, elapsed_s=4.2, resource_usage={"cpu_s": 1.0}), tmp_path / "same.json")
    changed = live_replay_report(path, EchoBackend(run, gate, exit_code=1), tmp_path / "changed.json")

    assert same["matched"] and not same["security_failure"] and same["live_backend"]["name"] == "live-test"
    assert json.loads((tmp_path / "same.json").read_text()) == same
    assert not changed["matched"] and not changed["security_failure"]
    assert changed["first_mismatch"]["kind"] == "sandbox_outcome" and changed["first_mismatch"]["field"] == "/exit_code"
    assert (path.read_bytes(), manifest_path_for(path).read_bytes()) == before
    with pytest.raises(ValueError, match="overwrite"):
        live_replay_report(path, EchoBackend(run), path)


def test_live_replay_flags_a_changed_security_decision_as_a_security_failure(tmp_path):
    path = copy_episode(fixture("sandboxed_exec"), tmp_path / "copy")
    records = read_journal(path)
    run = first(records, "sandbox_execution")
    decision = first(records, "policy_decision", tool="issue_refund")
    decision.update(verdict="allow", rule_id="task.allow", reason_code="within_task_policy")
    rewrite(path, records)

    report = live_replay_report(path, EchoBackend(run, first(records, "output_gate")["streams"]), tmp_path / "report.json")

    assert report["security_failure"] and report["first_mismatch"]["kind"] == "security_decision"
    assert report["first_mismatch"]["seq"] == decision["seq"]


# Resume serves the recorded prefix's runs instead of repeating them


class FakeService:
    profile = QWEN3_PROFILE
    use_grammars = True

    def __init__(self, *responses):
        self.responses = list(responses)

    def chat(self, request):
        return self.responses.pop(0)


class RefusingBackend:
    identity = BackendIdentity("scripted-test", "1", SandboxTier.T2, None)

    def execute(self, request, arguments):
        raise AssertionError("resume re-ran a recorded sandbox call")


@pytest.fixture
def generator():
    """The golden generator module (its policies, completions, and scripted backend); main() is not run."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("bulkhead_episode_generator", FIXTURES / "regenerate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_interrupted_sandboxed_episode_resumes_without_re_running_its_prefix(tmp_path, generator):
    original = fixture("sandboxed_exec")
    crashed = copy_episode(original, tmp_path / "crashed")
    records = read_journal(original)
    cut = next(r["seq"] for r in records if r["event"] == "taint_transition") + 1
    rewrite(crashed, records[:cut])
    manifest = json.loads(manifest_path_for(crashed).read_text())
    manifest["completed"] = False
    manifest_path_for(crashed).write_text(json.dumps(manifest))
    options = dict(
        run_label="sandboxed_exec",
        security=SecurityConfig(generator.EXEC_POLICY, available_tiers=(SandboxTier.T2,)),
        sleep=lambda seconds: None,
    )

    resumed = run_claimed_episode(
        tmp_path / "crashed", SteadyhandClient(FakeService(generator.completion(calls=[generator.REFUND_CALL]))),
        build_registry(exec_tools=True), load_fixture().clone(), initial_messages(generator.GOAL),
        backend=RefusingBackend(), **options,
    )

    assert resumed.status is EpisodeStatus.ESCALATED
    assert [comparable(r) for r in read_journal(crashed)] == [comparable(r) for r in records]
    other = copy_episode(original, tmp_path / "other")
    rewrite(other, records[:cut])
    manifest_path_for(other).write_text(json.dumps(manifest))

    class OtherBackend(RefusingBackend):
        identity = BackendIdentity("scripted-test", "2", SandboxTier.T2, None)

    # Backend identity is not part of the id, but resume checks it: a sweep can never switch
    # backend or tier midway; a deliberate change needs a new run label.
    with pytest.raises(ReplayMismatch) as raised:
        run_claimed_episode(
            tmp_path / "other", SteadyhandClient(FakeService()), build_registry(exec_tools=True),
            load_fixture().clone(), initial_messages(generator.GOAL), backend=OtherBackend(), **options,
        )
    assert raised.value.field == "backend"


# Secrets never reach new records


def test_secrets_do_not_appear_in_journals_manifests_or_grants(tmp_path, monkeypatch, generator):
    secret, key = "hf_canarysecretvalue123", "local-canary-key-456"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("LOCAL_KEY", key)
    result = run_claimed_episode(
        tmp_path, SteadyhandClient(FakeService(
            generator.completion(calls=[("call_shell", "run_shell", {"command": "env"})]),
            generator.completion(calls=[generator.REFUND_CALL]),
        )),
        build_registry(exec_tools=True), load_fixture().clone(), initial_messages(generator.GOAL),
        run_label="secrets", security=SecurityConfig(generator.EXEC_POLICY, available_tiers=(SandboxTier.T2,)),
        backend=generator.ScriptedBackend("PATH=/usr/bin\n"), sleep=lambda seconds: None, secrets=(secret, key),
    )
    from bulkhead.approvals import ScriptedApproval, ScriptedApprover
    from steadyhand.security import resolve_approvals

    [grant] = resolve_approvals(result.journal_path, ScriptedApprover((ScriptedApproval("issue_refund", REFUND),)), now=generator.NOW)
    texts = [result.journal_path.read_text(), manifest_path_for(result.journal_path).read_text(), json.dumps(grant.as_dict())]
    for text in texts:
        assert secret not in text and key not in text and "[redacted]" not in text  # never there, not merely scrubbed
