"""Bulkhead part 3: strict payloads for every security event, on write and on read."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from bulkhead import audit
from bulkhead.audit import SECURITY_EVENT_FIELDS, VOLATILE_SECURITY_FIELDS, AuditError
from steadyhand.journal import EVENT_FIELDS, VOLATILE_FIELDS, EpisodeJournal, JournalFormatError, read_journal

ROOT = Path(__file__).parent.parent
SHA = "a" * 64
GRANT = {"tool": "issue_refund", "arguments_sha256": SHA, "rules": ["task.widen.bound"], "widen": {"bounds": {}}}
VALID = {
    "policy_decision": {
        "step": 0, "tool_call_id": "call_0_0", "via": "model", "tool": "issue_refund", "arguments_sha256": SHA,
        "verdict": "ask", "rule_id": "task.widen.bound", "reason_code": "bound_exceeded", "taint_before": "trusted",
        "policy_sha256": SHA, "grant_id": None, "normalized_request_sha256": SHA,
    },
    "taint_transition": {
        "step": 1, "tool_call_id": "call_1_0", "source_tool": "get_customer", "before": "trusted", "after": "untrusted", "source_seq": 7,
    },
    "approval_requested": {
        "step": 0, "tool_call_id": "call_0_0", "rule_id": "task.widen.bound", "request_sha256": SHA,
        "normalized_grant": GRANT, "parent_episode_id": None, "approval_mode": "deny_all",
    },
    "sandbox_execution": {
        "step": 0, "tool_call_id": "call_0_0", "via": "model", "tool": "run_shell", "backend": "bwrap", "tier": "T2",
        "config_sha256": SHA, "request_sha256": SHA, "started": True, "exit_code": 0, "signal": None, "timed_out": False,
        "cleanup_status": "clean", "stdout_bytes": 3, "stderr_bytes": 0, "stdout_truncated": False, "stderr_truncated": False,
        "violations": [{"kind": "seccomp", "rule_id": "bulkhead.seccomp.ptrace", "detail": "ptrace refused"}],
        "resource_usage": {"cpu_s": 0.02, "peak_memory_kb": 2048, "pids_peak": 2, "oom_killed": False}, "elapsed_s": 0.05,
        "result": {"ok": True, "handler_called": True, "value": {"exit_code": 0, "stdout": "ok\n"}},
    },
    "output_gate": {
        "step": 0, "tool_call_id": "call_0_0", "via": "model", "tool": "run_shell",
        "streams": [
            {"name": "stdout", "raw_bytes": 9000, "kept_bytes": 4096, "invalid_utf8": True, "stripped": 3, "neutralized": 1, "redacted": 0, "truncated": True},
            {"name": "stderr", "raw_bytes": 0, "kept_bytes": 0, "invalid_utf8": False, "stripped": 0, "neutralized": 0, "redacted": 0, "truncated": False},
        ],
    },
    "egress_decision": {
        "step": 0, "tool_call_id": "call_0_0", "request_id": "req_1", "authority": "pypi.org:443", "decision": "allow",
        "rule_id": "task.allow", "resolved_address_class": "public", "bytes_in": 512, "bytes_out": 120, "truncated": False,
        "elapsed_s": 0.3,
    },
}


def test_every_security_event_is_registered_with_the_journal():
    assert set(VALID) == set(SECURITY_EVENT_FIELDS)
    for event, fields in SECURITY_EVENT_FIELDS.items():
        assert EVENT_FIELDS[event] == frozenset(fields)
    for event, volatile in VOLATILE_SECURITY_FIELDS.items():
        assert VOLATILE_FIELDS[event] == volatile and volatile <= set(SECURITY_EVENT_FIELDS[event])


def test_only_measurements_are_volatile():
    assert VOLATILE_SECURITY_FIELDS == {
        "sandbox_execution": {"elapsed_s", "resource_usage"},
        "egress_decision": {"elapsed_s", "bytes_in", "bytes_out"},
    }
    assert "policy_decision" not in VOLATILE_FIELDS and "taint_transition" not in VOLATILE_FIELDS


@pytest.mark.parametrize("event", sorted(VALID))
def test_valid_payloads_build(event):
    assert audit.build(event, **VALID[event]) == VALID[event]


@pytest.mark.parametrize("event", sorted(VALID))
def test_missing_and_unknown_fields_are_refused(event):
    for field in VALID[event]:
        payload = {key: value for key, value in VALID[event].items() if key != field}
        with pytest.raises(AuditError) as raised:
            audit.validate(event, payload)
        assert raised.value.field == field
    with pytest.raises(AuditError) as raised:
        audit.validate(event, {**VALID[event], "raw_arguments": "{}"})
    assert raised.value.field == "raw_arguments"


BAD_VALUES = [
    ("policy_decision", "verdict", "maybe"),
    ("policy_decision", "rule_id", "task.allow_everything"),
    ("policy_decision", "via", "someone"),
    ("policy_decision", "taint_before", "grey"),
    ("policy_decision", "arguments_sha256", "ABC"),
    ("policy_decision", "grant_id", "approval-123"),
    ("policy_decision", "step", -1),
    ("policy_decision", "step", True),
    ("policy_decision", "tool", "rm -rf /"),
    ("taint_transition", "before", "unknown"),
    ("taint_transition", "source_seq", 1.5),
    ("approval_requested", "approval_mode", "auto_approve"),
    ("approval_requested", "rule_id", "task.allow"),
    ("approval_requested", "normalized_grant", {"tool": "issue_refund", "scope": "all"}),
    ("sandbox_execution", "tier", "T3"),
    ("sandbox_execution", "tier", "T0"),
    ("sandbox_execution", "cleanup_status", "probably"),
    ("sandbox_execution", "signal", 99),
    ("sandbox_execution", "elapsed_s", math.nan),
    ("sandbox_execution", "elapsed_s", math.inf),
    ("sandbox_execution", "resource_usage", {"gpu_s": 1}),
    ("sandbox_execution", "resource_usage", {"cpu_s": -1}),
    ("sandbox_execution", "violations", [{"kind": "x", "rule_id": "y"}]),
    ("sandbox_execution", "violations", [{"kind": "x", "rule_id": "y", "detail": "z"}] * 17),
    ("sandbox_execution", "violations", [{"kind": "x", "rule_id": "y", "detail": "line\nbreak"}]),
    ("sandbox_execution", "result", {"ok": True, "handler_called": True}),
    ("sandbox_execution", "result", {"ok": False, "handler_called": True, "value": 1}),
    ("egress_decision", "authority", "PyPI.org:443"),
    ("egress_decision", "authority", "https://pypi.org/simple?token=abc"),
    ("egress_decision", "decision", "maybe"),
    ("egress_decision", "resolved_address_class", "somewhere"),
    ("egress_decision", "bytes_in", -5),
    ("output_gate", "streams", []),
    ("output_gate", "streams", [{"name": "stdin", "raw_bytes": 0, "kept_bytes": 0, "invalid_utf8": False, "stripped": 0, "neutralized": 0, "redacted": 0, "truncated": False}]),
    ("output_gate", "streams", [{"name": "stdout", "raw_bytes": 1, "kept_bytes": 2, "invalid_utf8": False, "stripped": 0, "neutralized": 0, "redacted": 0, "truncated": False}]),
    ("output_gate", "streams", [{"name": "stdout", "raw_bytes": 1, "kept_bytes": 1, "invalid_utf8": False, "stripped": 0, "neutralized": 0, "redacted": 0, "truncated": False, "content": "x"}]),
]


@pytest.mark.parametrize(("event", "field", "value"), BAD_VALUES)
def test_invalid_values_and_enum_labels_are_refused(event, field, value):
    with pytest.raises(AuditError) as raised:
        audit.validate(event, {**VALID[event], field: value})
    assert raised.value.field.split(".")[0].split("[")[0] in (field, "fields")


def test_non_json_and_unbounded_values_are_refused():
    huge = copy.deepcopy(VALID["sandbox_execution"])
    huge["result"]["value"] = {"stdout": "x" * (audit.MAX_RECORD_CHARS + 1)}
    with pytest.raises(AuditError, match="exceeds"):
        audit.validate("sandbox_execution", huge)
    with pytest.raises(AuditError, match="not JSON"):
        audit.validate("sandbox_execution", {**VALID["sandbox_execution"], "result": {"ok": True, "handler_called": True, "value": {1, 2}}})
    with pytest.raises(AuditError, match="not JSON"):
        audit.validate("sandbox_execution", {**VALID["sandbox_execution"], "result": {"ok": True, "handler_called": True, "value": math.nan}})


@pytest.mark.parametrize(
    ("event", "changes", "field"),
    [
        ("policy_decision", {"verdict": "allow"}, "verdict"),
        ("policy_decision", {"reason_code": "within_task_policy"}, "reason_code"),
        ("policy_decision", {"grant_id": "apr_0123456789abcdef"}, "grant_id"),
        ("policy_decision", {"verdict": "allow", "rule_id": "approval.grant", "reason_code": "approved_by_grant"}, "grant_id"),
        ("taint_transition", {"before": "untrusted", "after": "trusted"}, "after"),
        ("taint_transition", {"before": "trusted", "after": "trusted"}, "after"),
        ("sandbox_execution", {"started": False}, "started"),
        ("sandbox_execution", {"started": False, "exit_code": None, "stdout_bytes": 0}, "result"),
    ],
)
def test_inconsistent_combinations_are_refused(event, changes, field):
    with pytest.raises(AuditError) as raised:
        audit.validate(event, {**VALID[event], **changes})
    assert raised.value.field == field


# Through the journal


def test_the_journal_refuses_to_write_an_invalid_security_record(tmp_path):
    journal = EpisodeJournal(tmp_path / "ep_x.jsonl", episode_id="ep_x")
    journal.taint_transition(**VALID["taint_transition"])

    with pytest.raises(AuditError):
        journal.policy_decision(**{**VALID["policy_decision"], "verdict": "allow"})
    with pytest.raises(AuditError):
        journal.egress_decision(**{**VALID["egress_decision"], "authority": "https://pypi.org/x"})
    assert len((tmp_path / "ep_x.jsonl").read_text().splitlines()) == 1  # nothing invalid reached the file


def write_journal(path, records):
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def a_journal(tmp_path, *security_records):
    started = {
        "event": "episode_started", "episode_id": "ep_x", "seq": 0, "recorded_at": "t", "journal_version": 1,
        "settings": {}, "fixture_sha256": None, "store_sha256": SHA, "budgets": {}, "guard_limit": 3,
        "paraphrase_final": False, "tool_names": [], "schema_sha256": SHA, "policy_sha256": SHA, "messages": [],
    }
    records = [started] + [
        {"event": event, "episode_id": "ep_x", "seq": index + 1, "recorded_at": "t", **fields}
        for index, (event, fields) in enumerate(security_records)
    ]
    path = tmp_path / "ep_x.jsonl"
    write_journal(path, records)
    return path, records


def test_a_journal_with_every_security_event_reads_back(tmp_path):
    path, records = a_journal(tmp_path, *VALID.items())

    assert read_journal(path) == records


@pytest.mark.parametrize(
    ("event", "field", "value"),
    [("policy_decision", "verdict", "maybe"), ("taint_transition", "after", "trusted"), ("sandbox_execution", "tier", "T9"),
     ("approval_requested", "normalized_grant", {}), ("egress_decision", "decision", "perhaps")],
)
def test_reading_a_tampered_security_record_names_the_field(tmp_path, event, field, value):
    path, records = a_journal(tmp_path, (event, {**VALID[event], field: value}))

    with pytest.raises(JournalFormatError) as raised:
        read_journal(path)
    assert raised.value.seq == 1 and raised.value.field.startswith(field)


def test_unknown_missing_fields_and_sequence_gaps_are_refused_on_read(tmp_path):
    path, records = a_journal(tmp_path, ("taint_transition", VALID["taint_transition"]), ("policy_decision", VALID["policy_decision"]))
    for mutate in (
        lambda r: r[1].update(raw_url="https://x/?k=1"),
        lambda r: r[2].pop("rule_id"),
        lambda r: r[2].update(seq=5),
    ):
        tampered = copy.deepcopy(records)
        mutate(tampered)
        write_journal(path, tampered)
        with pytest.raises(JournalFormatError):
            read_journal(path)


def test_old_journals_without_security_events_still_read():
    for path in sorted((ROOT / "tests" / "fixtures" / "steadyhand_episodes").glob("ep_*.jsonl")):
        records = read_journal(path)
        assert not {record["event"] for record in records} & set(SECURITY_EVENT_FIELDS)
        assert "bulkhead" not in records[0]


def test_payloads_hold_no_raw_arguments_urls_or_command_lines():
    secured = ROOT / "tests" / "fixtures" / "bulkhead_episodes"
    for path in secured.glob("ep_*.jsonl"):
        for record in read_journal(path):
            if record["event"] in SECURITY_EVENT_FIELDS:
                text = json.dumps({k: v for k, v in record.items() if k != "result"})
                assert "cat out/receipt.txt" not in text and "Dented lid" not in text and "https://" not in text
