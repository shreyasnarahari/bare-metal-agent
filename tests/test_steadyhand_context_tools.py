"""``read_blob`` as a governed tool, and the server-only execution context that reaches it."""

import json

import pytest

from steadyhand import SteadyhandClient, initial_messages, load_fixture, run_episode
from steadyhand.executor import SequentialExecutor
from steadyhand.journal import EpisodeJournal, read_journal
from steadyhand.registry import SteadyhandRegistry
from steadyhand.replay import registry_for
from steadyhand.security import SecurityConfig, store_task_policy
from steadyhand.tools import build_registry
from steadyhand.tools_context import READ_BLOB_MAX_BYTES, READ_BLOB_POLICY, READ_BLOB_SECURITY, check_read_blob, read_blob
from steadyhand.types import EpisodeStatus, ToolCall
from steadyhand.validate import validate_call
from tidemark.blobs import BlobStore
from tidemark.resources import JournalView, ToolExecutionContext

from tidemark_support import FakeService, attempt, completion, shaping_session

PAYLOAD = ("evidence line é\n" * 200).encode()


@pytest.fixture
def blobs(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    handle = store.put("ep-ns", PAYLOAD, media_type="text/plain; charset=utf-8").record.handle
    return store, handle


def context_for(store, namespace="ep-ns"):
    return ToolExecutionContext(episode_id="ep_1", step=0, tool="read_blob", tool_call_id="c1", journal=JournalView([]),
                                namespace=namespace, blobs=store.namespace(namespace))


def call(arguments, call_id="c1"):
    return ToolCall(call_id, "read_blob", json.dumps(arguments))


def test_the_schema_is_closed_and_the_execution_context_is_hidden():
    registry = build_registry(context_tools=True)
    schema = registry.get("read_blob").parameters
    assert set(schema["properties"]) == {"handle", "offset", "max_bytes"} and schema["required"] == ["handle"]
    assert schema["additionalProperties"] is False and "execution_context" not in json.dumps(registry.schemas())
    verdict = validate_call(call({"handle": "blob://sha256:" + "0" * 64, "execution_context": {}}), registry, load_fixture())
    assert not verdict.ok and verdict.error.details[0].field == "/execution_context"


def test_the_default_registry_and_its_hashes_are_unchanged_and_replay_finds_the_context_registry():
    assert build_registry().schemas_sha256() == build_registry(context_tools=False).schemas_sha256()
    assert "read_blob" not in build_registry().names()
    with_blobs = build_registry(context_tools=True)
    assert with_blobs.names()[-1] == "read_blob"
    assert registry_for(with_blobs.schemas_sha256()).names() == with_blobs.names()
    assert registry_for(build_registry(exec_tools=True, context_tools=True).schemas_sha256()).names()[-1] == "read_blob"


def test_policy_and_security_are_a_retry_safe_read_with_untrusted_output_and_no_alternative():
    assert READ_BLOB_POLICY.idempotent and READ_BLOB_POLICY.alternatives == () and READ_BLOB_POLICY.verify is None
    assert "not_found" not in READ_BLOB_POLICY.retry.on and "invalid_arguments" not in READ_BLOB_POLICY.retry.on
    assert READ_BLOB_SECURITY.output_trust.value == "untrusted" and READ_BLOB_SECURITY.sinks == ()
    assert READ_BLOB_SECURITY.sandbox is None


@pytest.mark.parametrize("arguments,field", [
    ({"handle": "blob://sha256:abc"}, "/handle"),
    ({"handle": "/etc/passwd"}, "/handle"),
    ({"handle": "blob://sha256:" + "0" * 64, "offset": -1}, "/offset"),
    ({"handle": "blob://sha256:" + "0" * 64, "offset": 1.0}, "/offset"),
    ({"handle": "blob://sha256:" + "0" * 64, "max_bytes": 0}, "/max_bytes"),
    ({"handle": "blob://sha256:" + "0" * 64, "max_bytes": READ_BLOB_MAX_BYTES + 1}, "/max_bytes"),
])
def test_handle_syntax_and_ceilings_are_semantic_checks(arguments, field):
    verdict = validate_call(call(arguments), build_registry(context_tools=True), load_fixture())
    assert not verdict.ok and verdict.error.error_type.value == "invalid_arguments"
    assert verdict.error.details[0].field == field and check_read_blob(load_fixture(), arguments)


def test_the_handler_reads_a_range_through_the_server_supplied_namespace(blobs):
    store, handle = blobs
    registry = build_registry(context_tools=True)
    result = SequentialExecutor().run(call({"handle": handle, "max_bytes": 20}), registry, load_fixture(),
                                      execution_context=context_for(store))
    assert result.ok and result.value["content"].encode() == PAYLOAD[:result.value["end_offset"]]
    assert result.value["next_offset"] == result.value["end_offset"] and result.value["truncated"] is True
    assert set(result.value) == set(registry.result_schema("read_blob")["properties"])


def test_without_a_store_unknown_handles_other_namespaces_and_bad_offsets_are_typed(blobs):
    store, handle = blobs
    registry, fixture, executor = build_registry(context_tools=True), load_fixture(), SequentialExecutor()
    missing = executor.run(call({"handle": handle}), registry, fixture)
    assert not missing.ok and missing.error.error_type.value == "handler_error"
    other = executor.run(call({"handle": handle}), registry, fixture, execution_context=context_for(store, "other-ns"))
    assert other.error.error_type.value == "not_found" and other.error.details[0].field == "/handle"
    assert not other.error.retryable and "Copy the handle exactly" in other.error.hint
    past = executor.run(call({"handle": handle, "offset": len(PAYLOAD) + 5}), registry, fixture,
                        execution_context=context_for(store))
    assert past.error.error_type.value == "invalid_arguments" and past.error.details[0].field == "/offset"


def test_handlers_that_do_not_name_the_context_never_receive_it(blobs):
    store, _ = blobs
    seen = {}

    def plain_tool(store, value: str) -> dict:
        """Echo."""
        seen["arguments"] = value
        return {"value": value}

    registry = SteadyhandRegistry()
    registry.register(plain_tool, READ_BLOB_POLICY)
    result = SequentialExecutor().run(ToolCall("c", "plain_tool", '{"value": "x"}'), registry.freeze(), load_fixture(),
                                      execution_context=context_for(store))
    assert result.ok and result.value == {"value": "x"} and seen == {"arguments": "x"}


GOAL = "Read the stored evidence."


def read_turns(handle, offset=0):
    return [
        completion(calls=[("c1", "read_blob", {"handle": handle, "offset": offset, "max_bytes": 64})], attempts=[attempt(prompt_n=40)]),
        completion("Done.", attempts=[attempt(prompt_n=12)]),
    ]


def run_with(tmp_path, turns, *, security=None, namespace="ep-ns", registry=None, recovery=None):
    session = shaping_session(tmp_path / "blobs", namespace)
    journal = EpisodeJournal.create(tmp_path / "journals")
    result = run_episode(SteadyhandClient(FakeService(*turns)), registry or build_registry(context_tools=True),
                         load_fixture().clone(), initial_messages(GOAL), journal=journal, context=session,
                         security=security, recovery=recovery)
    return result, read_journal(journal.path)


def test_validate_guard_authorize_then_handler_and_the_result_taints(tmp_path, blobs):
    _, handle = blobs
    security = SecurityConfig(task_policy=store_task_policy(available_tools=("read_blob",)))
    result, records = run_with(tmp_path, read_turns(handle), security=security)
    assert result.status is EpisodeStatus.DONE
    events = [r["event"] for r in records]
    decision = events.index("policy_decision")
    tool = events.index("tool_result")
    taint = events.index("taint_transition")
    assert decision < tool < taint and records[decision]["verdict"] == "allow" and records[decision]["tool"] == "read_blob"
    assert records[tool]["ok"] and records[taint]["after"] == "untrusted" and records[taint]["source_seq"] == records[tool]["seq"]
    assert "observation_shaped" not in events  # a blob read is never shaped again


def test_a_task_policy_must_list_read_blob_explicitly(tmp_path, blobs):
    _, handle = blobs
    unlisted, records = run_with(tmp_path / "unlisted", read_turns(handle), security=SecurityConfig(task_policy=store_task_policy()))
    tool = next(r for r in records if r["event"] == "tool_result")
    assert tool["error"]["error_type"] == "permission_required" and not tool["handler_called"]
    assert unlisted.status is EpisodeStatus.ESCALATED
    denied_policy = store_task_policy(deny_tools=("read_blob",))
    _, records = run_with(tmp_path / "denied", read_turns(handle), security=SecurityConfig(task_policy=denied_policy))
    tool = next(r for r in records if r["event"] == "tool_result")
    assert tool["error"]["error_type"] == "permission_denied" and not tool["handler_called"]


def test_a_missing_security_spec_remains_a_denial(tmp_path, blobs):
    _, handle = blobs
    from steadyhand.tools import RESULT_SCHEMAS, TOOL_SECURITY, TOOL_SPECS
    from steadyhand.tools_context import check_read_blob as semantic

    registry = SteadyhandRegistry()
    for handler, policy, check in TOOL_SPECS:
        registry.register(handler, policy, check, RESULT_SCHEMAS[handler.__name__], security=TOOL_SECURITY[handler.__name__])
    registry.register(read_blob, READ_BLOB_POLICY, semantic)  # no SecuritySpec
    security = SecurityConfig(task_policy=store_task_policy(available_tools=("read_blob",)))
    _, records = run_with(tmp_path, read_turns(handle), security=security, registry=registry.freeze())
    tool = next(r for r in records if r["event"] == "tool_result")
    assert tool["error"]["error_type"] == "permission_denied"


def test_another_namespaces_handle_is_not_found_and_recovery_cannot_switch_tools(tmp_path, blobs):
    _, handle = blobs
    from steadyhand.recovery import strategy_for

    result, records = run_with(tmp_path, read_turns(handle), namespace="someone-else", recovery=strategy_for("switch"))
    tool = next(r for r in records if r["event"] == "tool_result")
    assert tool["error"]["error_type"] == "not_found"
    assert not [r for r in records if r["event"] == "policy_attempt"]  # no alternative exists to switch to


def test_the_secured_executor_forwards_the_context_only_after_allow(tmp_path, blobs):
    store, handle = blobs
    from steadyhand.security import SecuredExecutor, SecurityGate

    registry = build_registry(context_tools=True)
    journal = EpisodeJournal.create(tmp_path / "j")
    allowed = SecurityGate(SecurityConfig(task_policy=store_task_policy(available_tools=("read_blob",))), registry,
                           journal=journal, episode_id=journal.episode_id)
    result = SecuredExecutor(SequentialExecutor(), allowed).run(call({"handle": handle}), registry, load_fixture(),
                                                                execution_context=context_for(store))
    assert result.ok
    denied = SecurityGate(SecurityConfig(task_policy=store_task_policy(deny_tools=("read_blob",))), registry, journal=journal,
                          episode_id=journal.episode_id)
    refused = SecuredExecutor(SequentialExecutor(), denied).run(call({"handle": handle}, "c2"), registry, load_fixture(),
                                                                execution_context=context_for(store))
    assert refused.error.error_type.value == "permission_denied" and not refused.handler_called


def test_the_journal_view_is_read_only():
    records = [{"event": "x", "seq": 0, "nested": {"a": 1}}]
    view = JournalView(records)
    copy = view[0]
    copy["nested"]["a"] = 2
    assert records[0]["nested"]["a"] == 1 and len(view) == 1 and list(view)[0]["event"] == "x"
