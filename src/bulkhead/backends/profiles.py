"""What each execution profile enforces, side by side, so tiers and phases are never confused.

``PROFILES`` describes every profile a backend can report in
``sandbox_execution.backend`` (plus T0, which runs nothing). ``attack_matrix``
turns the attack fixtures into a table of how each attack is expected to end
under each profile:

- ``blocked`` / ``killed``: the attack's objective fails the way the fixture says;
- ``different_boundary``: it fails, but by a different mechanism than tier 2
  (typically access denied where tier 2 makes the path invisible);
- ``not_applicable``: the profile cannot host the attack safely or meaningfully
  (a fork bomb without a PID ceiling, any exec on T0).

The matrix is an expectation, not a result. Whether an expectation has been
verified on a given host is decided only by running the integration tests
there; a skipped profile is "pending", never "passed".
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

PROPERTIES = ("tier", "filesystem", "network", "processes", "memory", "pids", "cpu", "disk", "signals", "proc", "sockets")

PROFILES: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "bwrap-t2": MappingProxyType({
        "tier": "T2",
        "filesystem": "mount namespace: undeclared paths are absent; denied paths are masked",
        "network": "fresh network namespace; no egress (the part 6 proxy is the only way out)",
        "processes": "PID namespace; cgroup.kill ends the whole call tree",
        "memory": "cgroup memory.max; an OOM kills the whole call (memory.oom.group)",
        "pids": "cgroup pids.max",
        "cpu": "cgroup cpu.max quota and a total cpu_s ceiling over the tree",
        "disk": "sized /tmp tmpfs where bubblewrap has --size; RLIMIT_FSIZE per file; monitored workspace total",
        "signals": "confined to the PID namespace",
        "proc": "the sandbox's own /proc only",
        "sockets": "AF_UNIX only, inside the fresh network namespace",
    }),
    "bwrap-namespace-phase": MappingProxyType({
        "tier": "T2 (namespace phase, not the complete profile)",
        "filesystem": "mount namespace: undeclared paths are absent; denied paths are masked",
        "network": "fresh network namespace; no egress",
        "processes": "PID namespace, process group, and session; no cgroup",
        "memory": "none",
        "pids": "none",
        "cpu": "RLIMIT_CPU per process",
        "disk": "sized /tmp tmpfs where supported; RLIMIT_FSIZE per file; monitored workspace total",
        "signals": "confined to the PID namespace",
        "proc": "the sandbox's own /proc only",
        "sockets": "AF_UNIX only, inside the fresh network namespace",
    }),
    "landlock-t1-cgroup": MappingProxyType({
        "tier": "T1",
        "filesystem": "Landlock: access is denied but names may be visible; a directory holding a denied entry cannot be listed",
        "network": "none: Landlock denies TCP bind/connect and seccomp refuses socket()",
        "processes": "process group (setsid/setpgid refused) and cgroup.kill",
        "memory": "cgroup memory.max",
        "pids": "cgroup pids.max",
        "cpu": "cgroup cpu.max quota and a total cpu_s ceiling over the tree",
        "disk": "RLIMIT_FSIZE per file; monitored workspace total; no private /tmp",
        "signals": "scoped to the sandbox on Landlock ABI 6+, otherwise unscoped",
        "proc": "not granted",
        "sockets": "none (socketpair only)",
    }),
    "landlock-t1": MappingProxyType({
        "tier": "T1",
        "filesystem": "Landlock: access is denied but names may be visible; a directory holding a denied entry cannot be listed",
        "network": "none: Landlock denies TCP bind/connect and seccomp refuses socket()",
        "processes": "process group only (setsid/setpgid refused); no cgroup",
        "memory": "RLIMIT_AS per-process backup only",
        "pids": "none",
        "cpu": "RLIMIT_CPU per process",
        "disk": "RLIMIT_FSIZE per file; monitored workspace total; no private /tmp",
        "signals": "scoped to the sandbox on Landlock ABI 6+, otherwise unscoped",
        "proc": "not granted",
        "sockets": "none (socketpair only)",
    }),
    "T0": MappingProxyType({name: ("T0" if name == "tier" else "not a sandbox: exec and network tools are denied") for name in PROPERTIES}),
})
CGROUP_PROFILES = frozenset({"bwrap-t2", "landlock-t1-cgroup"})
STATUSES = frozenset({"blocked", "killed", "different_boundary", "not_applicable"})


def attack_matrix(part4: Mapping[str, Any], part5: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Each attack id to its expected status under each profile (see the module docstring)."""
    rows: dict[str, dict[str, str]] = {}
    for attack in part4["attacks"]:
        tier2 = "killed" if attack["expected_verdict"] == "killed" else "blocked"
        tier1 = part5["tier1"][attack["id"]]
        tier1_status = "different_boundary" if tier1["boundary"] == "different" else tier1["status"]
        rows[attack["id"]] = {
            "bwrap-t2": tier2,
            "bwrap-namespace-phase": tier2,
            "landlock-t1-cgroup": tier1_status,
            "landlock-t1": tier1_status,
            "T0": "not_applicable",
        }
    for attack in part5["resource"]:
        row = {}
        for profile in PROFILES:
            if profile == "T0" or (attack["requires_cgroup"] and profile not in CGROUP_PROFILES):
                row[profile] = "not_applicable"
            else:
                row[profile] = "killed" if attack["expected"].get("exhausted") or attack["expected"].get("signal") else "blocked"
        rows[attack["id"]] = row
    return rows


__all__ = ["CGROUP_PROFILES", "PROFILES", "PROPERTIES", "STATUSES", "attack_matrix"]
