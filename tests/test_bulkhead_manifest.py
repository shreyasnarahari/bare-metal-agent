"""Bulkhead part 2: security metadata (paths, egress, limits, sandboxes, specs) and its registration."""

from __future__ import annotations

import copy
import math

import pytest

from bulkhead.errors import SecuritySpecError
from bulkhead.manifest import (
    DEFAULT_LIMITS,
    GLOBAL_MAXIMA,
    WORKSPACE,
    Bounds,
    Egress,
    Limits,
    Sandbox,
    SecuritySpec,
    Sink,
    Trust,
    beneath,
    destination_authority,
    normalize_argument_path,
    normalize_authority,
    normalize_path,
)
from steadyhand import Effect, PolicyError, SteadyhandRegistry, ToolPolicy, build_registry
from steadyhand.store import Store
from steadyhand.tools import TOOL_SECURITY
from steadyhand.tools_exec import EXEC_POLICY, EXEC_SECURITY

EXAMPLE = SecuritySpec(
    sandbox=Sandbox(
        fs_read=("$WORKSPACE", "/usr", "/lib", "/lib64", "/etc/ssl"),
        fs_write=("$WORKSPACE/out",),
        fs_deny=("$WORKSPACE/.git", "$WORKSPACE/.env"),
        net=Egress(allow=("pypi.internal.corp:443",)),
        limits=Limits(cpu_s=10, mem_mb=512, pids=64, disk_mb=200, out_kb=64),
        secrets=(),
    ),
    output_trust=Trust.UNTRUSTED,
    sinks=(Sink.FILESYSTEM, Sink.NETWORK),
)


# Paths


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$WORKSPACE", "$WORKSPACE"),
        ("$WORKSPACE/", "$WORKSPACE"),
        ("$WORKSPACE//out///", "$WORKSPACE/out"),
        ("/usr/lib/", "/usr/lib"),
        ("//usr//lib", "/usr/lib"),
        ("/", "/"),
    ],
)
def test_paths_normalize_separators(text, expected):
    assert normalize_path(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "relative/path", "$WORKSPACE/../etc", "/usr/./lib", "/usr/..", "$HOME/x", "$WORKSPACE/$OTHER", "C:\\temp", "/tmp/\x00x", 7, None],
)
def test_ambiguous_paths_are_refused_not_resolved(text):
    with pytest.raises(SecuritySpecError):
        normalize_path(text)


def test_argument_paths_are_relative_to_the_workspace():
    assert normalize_argument_path("out/report.txt") == "$WORKSPACE/out/report.txt"
    assert normalize_argument_path("/usr/lib") == "/usr/lib"
    with pytest.raises(SecuritySpecError):
        normalize_argument_path("../escape")


def test_beneath_is_component_wise_and_roots_are_separate():
    assert beneath("$WORKSPACE/out/x", "$WORKSPACE/out") and beneath("$WORKSPACE", "$WORKSPACE")
    assert not beneath("$WORKSPACE/outside", "$WORKSPACE/out")  # not a string-prefix match
    assert beneath("/etc/ssl", "/") and not beneath("$WORKSPACE/x", "/")


# Network authorities


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PyPI.org:443", "pypi.org:443"),
        ("pypi.org.:443", "pypi.org:443"),
        ("10.0.0.1:8080", "10.0.0.1:8080"),
        ("[2001:DB8::0:1]:443", "[2001:db8::1]:443"),
        ("localhost:3128", "localhost:3128"),
    ],
)
def test_authorities_normalize(text, expected):
    assert normalize_authority(text) == expected


@pytest.mark.parametrize(
    "text",
    ["pypi.org", "pypi.org:0", "pypi.org:65536", "pypi.org:0443", "*.pypi.org:443", "https://pypi.org:443", "user@pypi.org:443",
     "pypi.org:443/simple", "-bad.org:443", "a..b:443", "[::1:443", "pypi org:443"],
)
def test_authorities_are_exact_host_port_values(text):
    with pytest.raises(SecuritySpecError):
        normalize_authority(text)


def test_destinations_come_from_urls_or_authorities():
    assert destination_authority("https://pypi.org/simple/") == "pypi.org:443"
    assert destination_authority("http://Example.COM:8080/x?y=1") == "example.com:8080"
    assert destination_authority("http://[::1]/") == "[::1]:80"
    assert destination_authority("pypi.org:443") == "pypi.org:443"
    for bad in ("ftp://pypi.org/", "https://u:p@pypi.org/", "https:///nohost", "http://pypi.org:99999/"):
        with pytest.raises(SecuritySpecError):
            destination_authority(bad)


# Limits, egress, sandbox


def test_limits_are_positive_finite_and_globally_capped():
    limits = Limits(cpu_s=10.0, mem_mb=512, pids=64, disk_mb=200, out_kb=64)

    assert limits.cpu_s == 10 and isinstance(limits.cpu_s, int)  # integer-valued floats hash like ints
    assert limits.stdout_kb == limits.stderr_kb == 64  # per-stream subcaps default to the combined cap
    for name, value in [("cpu_s", 0), ("mem_mb", -1), ("pids", math.inf), ("cpu_s", math.nan), ("mem_mb", 1.5),
                        ("disk_mb", True), ("out_kb", GLOBAL_MAXIMA["out_kb"] + 1)]:
        options = {"cpu_s": 10, "mem_mb": 512, "pids": 64, "disk_mb": 200, "out_kb": 64, name: value}
        with pytest.raises(SecuritySpecError):
            Limits(**options)
    with pytest.raises(SecuritySpecError):
        Limits(cpu_s=10, mem_mb=512, pids=64, disk_mb=200, out_kb=64, stdout_kb=65)


def test_empty_egress_means_no_network():
    assert not Egress().enabled and Egress().allow == ()
    assert not EXEC_SECURITY.sandbox.net.enabled
    assert Egress(allow=("b.example:443", "A.example:443", "a.example:443")).allow == ("a.example:443", "b.example:443")


def test_writes_must_stay_in_the_workspace_and_denies_win():
    sandbox = EXAMPLE.sandbox

    assert sandbox.writable("$WORKSPACE/out/result.txt")
    assert not sandbox.writable("$WORKSPACE/src/main.py")  # readable, not writable
    assert sandbox.readable("$WORKSPACE/src/main.py") and sandbox.readable("/usr/lib/libc.so")
    assert not sandbox.readable("$WORKSPACE/.git/config") and not sandbox.readable("$WORKSPACE/.env")
    write_everywhere = Sandbox(fs_write=("$WORKSPACE",), fs_deny=("$WORKSPACE/.git",))
    assert write_everywhere.writable("$WORKSPACE/a") and not write_everywhere.writable("$WORKSPACE/.git/hooks/x")
    with pytest.raises(SecuritySpecError, match="inside"):
        Sandbox(fs_write=("/tmp/out",))


def test_secrets_are_rejected_without_echoing_them():
    with pytest.raises(SecuritySpecError) as raised:
        Sandbox(secrets=("AWS_SECRET_ACCESS_KEY=abc123",))
    assert "abc123" not in str(raised.value) and "AWS_SECRET" not in str(raised.value)


def test_sinks_must_cover_what_the_sandbox_allows():
    with pytest.raises(SecuritySpecError, match="network sink"):
        SecuritySpec(sandbox=Sandbox(net=Egress(allow=("pypi.org:443",))), sinks=())
    with pytest.raises(SecuritySpecError, match="filesystem sink"):
        SecuritySpec(sandbox=Sandbox(fs_write=("$WORKSPACE/out",)), sinks=(Sink.NETWORK,))
    with pytest.raises(SecuritySpecError, match="give it a sandbox"):
        SecuritySpec(sandbox=None, sinks=(Sink.NETWORK,))
    with pytest.raises(SecuritySpecError):
        SecuritySpec(sandbox=None, sinks=("telepathy",))


def test_spec_canonical_json_round_trips_and_hashes_stably():
    same = SecuritySpec(
        sandbox=Sandbox(
            fs_read=("/etc/ssl/", "/lib64", "//lib", "/usr", "$WORKSPACE"),
            fs_write=("$WORKSPACE/out/",),
            fs_deny=("$WORKSPACE/.env", "$WORKSPACE/.git"),
            net=Egress(allow=("PYPI.internal.corp:443",)),
            limits=Limits(cpu_s=10.0, mem_mb=512, pids=64, disk_mb=200, out_kb=64),
        ),
        sinks=(Sink.NETWORK, Sink.FILESYSTEM, Sink.NETWORK),
    )

    assert same == EXAMPLE and same.sha256() == EXAMPLE.sha256()
    assert SecuritySpec.from_dict(EXAMPLE.as_dict()) == EXAMPLE
    assert EXAMPLE.as_dict()["sinks"] == ["filesystem", "network"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra=1),
        lambda d: d["sandbox"].update(extra=1),
        lambda d: d["sandbox"]["limits"].update(cpu_s=float("nan")),
        lambda d: d["sandbox"]["net"].update(proxy="x"),
        lambda d: d.update(output_trust="mostly"),
        lambda d: d["sandbox"].update(secrets=["TOKEN"]),
        lambda d: d["sandbox"].pop("fs_deny"),
    ],
)
def test_spec_loading_is_strict(mutate):
    document = copy.deepcopy(EXAMPLE.as_dict())
    mutate(document)
    with pytest.raises(SecuritySpecError):
        SecuritySpec.from_dict(document)


def test_bounds_need_a_side_and_an_order():
    assert Bounds(1, 5).contains(5) and not Bounds(1, 5).contains(5.01) and Bounds(None, 3).contains(-100)
    assert Bounds(1, 5).widened_to(9) == Bounds(1, 9) and Bounds(None, 5).widened_to(-3) == Bounds(None, 5)
    for bad in ((None, None), (5, 1), (math.nan, None), (True, None)):
        with pytest.raises(SecuritySpecError):
            Bounds(*bad)


# Registration in Steadyhand


def probe_tool(store: Store, order_id: str) -> dict:
    """A stand-in exec tool."""
    raise AssertionError("never called")


def test_exec_tools_need_a_sandboxed_spec_and_no_automatic_recovery():
    registry = SteadyhandRegistry()
    with pytest.raises(PolicyError, match="sandbox"):
        registry.register(probe_tool, EXEC_POLICY)
    with pytest.raises(PolicyError, match="sandbox"):
        registry.register(probe_tool, EXEC_POLICY, security=SecuritySpec(sandbox=None))
    with pytest.raises(PolicyError, match="non-idempotent|idempotent"):
        ToolPolicy(Effect.EXEC, 5.0, idempotent=True)
    with pytest.raises(PolicyError, match="exec tool cannot have automatic retries"):
        from steadyhand.policy import RetrySpec

        ToolPolicy(Effect.EXEC, 5.0, idempotent=False, retry=RetrySpec(on=("timeout",), max_attempts=1))
    with pytest.raises(PolicyError, match="verify tool or alternatives"):
        ToolPolicy(Effect.EXEC, 5.0, idempotent=False, alternatives=("other",))
    assert not Effect.EXEC.idempotent and Effect.READ.idempotent and not Effect.NON_IDEMPOTENT_WRITE.idempotent


def test_registry_keeps_security_in_a_side_table():
    registry = build_registry()
    exec_registry = build_registry(exec_tools=True)

    assert registry.security("issue_refund") == TOOL_SECURITY["issue_refund"]
    assert registry.security("get_order").sinks == () and registry.security("issue_refund").sinks == (Sink.EXTERNAL_EFFECT,)
    assert not any(spec.sandboxed for spec in map(registry.security, registry.names()))
    assert exec_registry.security("run_shell") == EXEC_SECURITY
    # Security metadata never reaches the model, and the default registry keeps its golden hashes.
    assert "sandbox" not in str(registry.schemas()) and "fs_read" not in str(exec_registry.schemas())
    assert registry.schemas_sha256() == "589d6e5e44640999ebdc8a17346e9a845d7be88d2cfcad3492bcb3d075cabb14"
    assert registry.policies_sha256() == "5e8004bc2a9fb4cef5f888edc29d9309e0ab6632786a9bc0ad0805915b1325e1"
    assert registry.security_spec_sha256() != exec_registry.security_spec_sha256()
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(probe_tool, ToolPolicy(Effect.READ, 5.0, idempotent=True))
    with pytest.raises(KeyError):
        registry.security("no_such_tool")


def test_a_tool_registered_without_a_spec_has_none():
    registry = SteadyhandRegistry()
    registry.register(probe_tool, ToolPolicy(Effect.READ, 5.0, idempotent=True))

    assert registry.security("probe_tool") is None
    with pytest.raises(TypeError):
        SteadyhandRegistry().register(probe_tool, ToolPolicy(Effect.READ, 5.0, idempotent=True), security={"sandbox": None})


def test_defaults_are_conservative():
    assert DEFAULT_LIMITS.ceilings()["out_kb"] == 64 and WORKSPACE == "$WORKSPACE"
    assert SecuritySpec(sandbox=None).output_trust is Trust.UNTRUSTED  # outside content until reviewed
