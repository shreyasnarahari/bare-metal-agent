"""Value types for the Bulkhead capability probe: tiers, capability records, and the report.

A ``ProbeReport`` is evidence, not configuration. Its derived fields (tool
versions, the Landlock and cgroup summaries, the per-tier requirement results,
the selected tier, and ``probe_sha256``) are always recomputed from the
capability records. ``ProbeReport.from_dict`` rebuilds a report from its
inputs and refuses one whose stored fields disagree with that rebuild, so a
hand-edited ``selected_tier`` or a stale hash is an error, never a tier.

``probe_sha256`` covers only semantic evidence: schema version, architecture,
and each capability's name, status, code, and facts. Timestamps, host names,
temporary paths, free-text details, interpreter paths, and timings are left
out, so two runs on the same host with the same capabilities hash the same.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import ProbeReportError, UnsupportedTierError

SCHEMA_VERSION = 1
MAX_DETAIL_CHARS = 200
MAX_PATH_CHARS = 300
MAX_FACT_STRING_CHARS = 120
MAX_FACT_LIST_ITEMS = 32

_CODE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}")
_FACT_KEY = re.compile(r"[a-z][a-z0-9_]{0,39}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class SandboxTier(str, Enum):
    """How strong an isolation boundary this host can enforce. Higher is stronger."""

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"

    @property
    def rank(self) -> int:
        return int(self.value[1:])

    @property
    def label(self) -> str:
        """The only wording a report or a log may use for this tier."""
        return TIER_LABELS[self]

    @property
    def is_sandbox(self) -> bool:
        """T0 is never a sandbox, whatever timeouts or rlimits are in force."""
        return self is not SandboxTier.T0


TIER_LABELS = {
    SandboxTier.T2: "Namespace sandbox",
    SandboxTier.T1: "Landlock sandbox",
    SandboxTier.T0: "NOT a sandbox",
}
# Highest first: selection takes the first tier whose requirements all hold.
TIERS_DESCENDING = (SandboxTier.T2, SandboxTier.T1, SandboxTier.T0)


class CapabilityStatus(str, Enum):
    """The outcome of one probe operation."""

    OK = "ok"  # the operation ran and its effect was verified
    UNAVAILABLE = "unavailable"  # a binary, library, or configuration is absent, or a prerequisite failed
    DENIED = "denied"  # the kernel or a security module refused (EPERM/EACCES, AppArmor, read-only cgroup)
    UNSUPPORTED = "unsupported"  # the kernel or tool lacks the feature (ENOSYS, old ABI, unknown option)
    FAILED = "failed"  # the operation ran but crashed, timed out, or did not have the verified effect


# Every report has exactly these records, in this order.
CAPABILITY_NAMES = (
    "bwrap.binary",
    "bwrap.namespaces",
    "bwrap.mounts",
    "bwrap.netns",
    "bwrap.unix_socket",
    "userns.unshare",
    "libseccomp.library",
    "seccomp.filter",
    "seccomp.socket_deny",
    "no_new_privs",
    "landlock.abi",
    "landlock.fs",
    "landlock.tcp",
    "rlimits",
    "process.group_cleanup",
    "cgroup.v2_mount",
    "cgroup.delegation",
    "cgroup.leaf",
    "cgroup.limits",
    "cgroup.kill",
    "fs.openat2",
)

CAPABILITY_DESCRIPTIONS = {
    "bwrap.binary": "bubblewrap binary runs",
    "bwrap.namespaces": "fresh user/mount/PID/net/IPC/UTS namespaces, fresh /proc, no_new_privs",
    "bwrap.mounts": "writable bind, read-only bind, masked subpath",
    "bwrap.netns": "network namespace cannot reach an outside listener",
    "bwrap.unix_socket": "bind-mounted Unix socket reachable while TCP stays blocked",
    "userns.unshare": "unprivileged user+network namespace without bubblewrap",
    "libseccomp.library": "system libseccomp loads",
    "seccomp.filter": "filter blocks a harmless syscall, allows others",
    "seccomp.socket_deny": "filter denies AF_INET/AF_INET6/AF_PACKET sockets, allows AF_UNIX",
    "no_new_privs": "PR_SET_NO_NEW_PRIVS sticks",
    "landlock.abi": "Landlock ABI query",
    "landlock.fs": "Landlock denies a canary outside the allowed directory",
    "landlock.tcp": "Landlock denies TCP connect (ABI 4+)",
    "rlimits": "RLIMIT_NOFILE/RLIMIT_FSIZE enforced, RLIMIT_CORE zero",
    "process.group_cleanup": "process-group kill reaps a child and grandchild",
    "cgroup.v2_mount": "cgroup2 mounted",
    "cgroup.delegation": "configured delegated parent is writable",
    "cgroup.leaf": "leaf created under the parent and removed",
    "cgroup.limits": "memory/pids/cpu limits written and read back",
    "cgroup.kill": "cgroup.kill removes a child and a session-escaped grandchild",
    "fs.openat2": "openat2 RESOLVE_BENEATH/RESOLVE_NO_SYMLINKS",
}

# Selection is the highest tier whose every requirement is ``ok``. Records not
# listed for any tier (bwrap.unix_socket, userns.unshare, landlock.tcp,
# fs.openat2) are evidence for later parts and never change the tier.
TIER_REQUIREMENTS: dict[SandboxTier, tuple[str, ...]] = {
    SandboxTier.T2: (
        "bwrap.binary",
        "bwrap.namespaces",
        "bwrap.mounts",
        "bwrap.netns",
        "libseccomp.library",
        "seccomp.filter",
        "no_new_privs",
        "rlimits",
        "cgroup.v2_mount",
        "cgroup.delegation",
        "cgroup.leaf",
        "cgroup.limits",
        "cgroup.kill",
    ),
    # Tier 1 has no namespaces, so its exec profile has no direct network:
    # seccomp must deny inet sockets. Landlock TCP rules are port rules, not
    # an egress boundary, and are recorded without being required.
    SandboxTier.T1: (
        "landlock.abi",
        "landlock.fs",
        "libseccomp.library",
        "seccomp.filter",
        "seccomp.socket_deny",
        "no_new_privs",
        "rlimits",
        "process.group_cleanup",
    ),
    SandboxTier.T0: (),
}

# Landlock records shown in the report summary, under the feature they test.
LANDLOCK_FEATURES = {"landlock.fs": "filesystem", "landlock.tcp": "tcp_connect"}


# JSON helpers. The probe layer imports nothing from toolsmith or steadyhand, so
# these mirror toolsmith.json_utils and toolsmith.evaluation.models.


def strict_json_loads(text: str) -> Any:
    """``json.loads`` that refuses NaN/Infinity, overflowing floats, and duplicate keys."""

    def reject_constant(constant: str) -> None:
        raise ProbeReportError(f"non-finite JSON number {constant}")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ProbeReportError("JSON numbers must be finite")
        return number

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProbeReportError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(text, parse_constant=reject_constant, parse_float=finite_float, object_pairs_hook=unique_object)
    except ProbeReportError:
        raise
    except (json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise ProbeReportError(f"not valid JSON: {exc}") from exc


def canonical_json(value: Any) -> str:
    """Compact JSON with sorted keys; NaN is refused."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    """sha256 of ``canonical_json``: the same value always gives the same hash."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# Field checks shared by every record. Each raises ProbeReportError naming the field.


def _check_keys(value: Any, expected: Iterable[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeReportError("must be a JSON object", field=where)
    expected = tuple(expected)
    unknown = sorted(set(value) - set(expected))
    missing = [key for key in expected if key not in value]
    if unknown:
        raise ProbeReportError(f"unknown field(s) {unknown}", field=where)
    if missing:
        raise ProbeReportError(f"missing field(s) {missing}", field=where)
    return value


def _text(value: Any, where: str, *, limit: int, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise ProbeReportError("must be a string", field=where)
    if len(value) > limit:
        raise ProbeReportError(f"longer than {limit} characters", field=where)
    if _CONTROL.search(value):
        raise ProbeReportError("contains control characters", field=where)
    return value


def _flag(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise ProbeReportError("must be true or false", field=where)
    return value


def _fact_scalar(value: Any, where: str) -> Any:
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value
    if isinstance(value, str):
        return _text(value, where, limit=MAX_FACT_STRING_CHARS)
    # Floats are refused outright: they are where NaN and unstable hashes come from.
    raise ProbeReportError("facts hold strings, integers, booleans, null, or lists of those", field=where)


def _check_facts(facts: Any, where: str) -> dict[str, Any]:
    if not isinstance(facts, Mapping):
        raise ProbeReportError("must be a JSON object", field=where)
    checked: dict[str, Any] = {}
    for key in sorted(facts):
        if not isinstance(key, str) or not _FACT_KEY.fullmatch(key):
            raise ProbeReportError(f"bad fact name {key!r}", field=where)
        value = facts[key]
        if isinstance(value, (list, tuple)):
            if len(value) > MAX_FACT_LIST_ITEMS:
                raise ProbeReportError(f"more than {MAX_FACT_LIST_ITEMS} items", field=f"{where}.{key}")
            checked[key] = [_fact_scalar(item, f"{where}.{key}") for item in value]
        else:
            checked[key] = _fact_scalar(value, f"{where}.{key}")
    return checked


def _status(value: Any, where: str) -> CapabilityStatus:
    if isinstance(value, CapabilityStatus):
        return value
    try:
        return CapabilityStatus(value)
    except ValueError:
        raise ProbeReportError(f"unknown capability status {value!r}", field=where) from None


def _tier(value: Any, where: str) -> SandboxTier:
    if isinstance(value, SandboxTier):
        return value
    try:
        return SandboxTier(value)
    except ValueError:
        raise ProbeReportError(f"unknown sandbox tier {value!r}", field=where) from None


@dataclass(frozen=True)
class Capability:
    """One probed capability: what happened, a stable code, and a short sanitized detail."""

    name: str
    status: CapabilityStatus
    code: str
    detail: str
    facts: Mapping[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        where = f"capabilities.{self.name}"
        if self.name not in CAPABILITY_NAMES:
            raise ProbeReportError(f"unknown capability {self.name!r}", field="capabilities")
        object.__setattr__(self, "status", _status(self.status, f"{where}.status"))
        if not isinstance(self.code, str) or not _CODE.fullmatch(self.code):
            raise ProbeReportError(f"bad code {self.code!r}", field=f"{where}.code")
        if (self.status is CapabilityStatus.OK) != (self.code == "ok"):
            raise ProbeReportError("code is 'ok' exactly when status is ok", field=f"{where}.code")
        _text(self.detail, f"{where}.detail", limit=MAX_DETAIL_CHARS)
        object.__setattr__(self, "facts", _check_facts(self.facts, f"{where}.facts"))
        elapsed = self.elapsed_s
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ProbeReportError("must be a finite, non-negative number", field=f"{where}.elapsed_s")
        object.__setattr__(self, "elapsed_s", float(elapsed))

    @property
    def ok(self) -> bool:
        return self.status is CapabilityStatus.OK

    def semantic(self) -> dict[str, Any]:
        """The part of the record that goes into ``probe_sha256``."""
        return {"name": self.name, "status": self.status.value, "code": self.code, "facts": dict(self.facts)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "code": self.code,
            "detail": self.detail,
            "facts": {key: list(value) if isinstance(value, list) else value for key, value in self.facts.items()},
            "elapsed_s": self.elapsed_s,
        }

    @classmethod
    def from_dict(cls, value: Any, where: str = "capabilities") -> Capability:
        _check_keys(value, ("name", "status", "code", "detail", "facts", "elapsed_s"), where)
        return cls(
            name=value["name"],
            status=_status(value["status"], f"{where}.status"),
            code=value["code"],
            detail=value["detail"],
            facts=value["facts"],
            elapsed_s=value["elapsed_s"],
        )


@dataclass(frozen=True)
class MissingRequirement:
    """A requirement of some tier that this host did not prove."""

    capability: str
    status: CapabilityStatus
    code: str
    detail: str

    @classmethod
    def of(cls, capability: Capability) -> MissingRequirement:
        return cls(capability.name, capability.status, capability.code, capability.detail)

    def summary(self) -> str:
        return f"{self.capability} {self.status.value} ({self.code})"

    def as_dict(self) -> dict[str, Any]:
        return {"capability": self.capability, "status": self.status.value, "code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class TierEvaluation:
    """Whether every requirement of one tier holds, and which do not."""

    tier: SandboxTier
    missing: tuple[MissingRequirement, ...]

    @property
    def satisfied(self) -> bool:
        return not self.missing

    def root_causes(self) -> tuple[MissingRequirement, ...]:
        """Missing requirements that failed on their own, not because a prerequisite did."""
        return tuple(item for item in self.missing if item.code != "dependency_unavailable")

    def summary(self) -> str:
        """Root causes, then how many requirements were skipped because of them."""
        causes = self.root_causes()
        text = "; ".join(item.summary() for item in causes)
        skipped = len(self.missing) - len(causes)
        return f"{text} (+{skipped} dependent)" if skipped else text

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.tier.label,
            "satisfied": self.satisfied,
            "missing": [item.as_dict() for item in self.missing],
        }


def evaluate_tiers(capabilities: Sequence[Capability]) -> tuple[TierEvaluation, ...]:
    """Evaluate every tier against the records, highest first."""
    by_name = {capability.name: capability for capability in capabilities}
    return tuple(
        TierEvaluation(
            tier,
            tuple(MissingRequirement.of(by_name[name]) for name in TIER_REQUIREMENTS[tier] if not by_name[name].ok),
        )
        for tier in TIERS_DESCENDING
    )


def select_tier(evaluations: Sequence[TierEvaluation]) -> SandboxTier:
    """The highest satisfied tier. T0 has no requirements, so there is always one."""
    return next(evaluation.tier for evaluation in evaluations if evaluation.satisfied)


@dataclass(frozen=True)
class HostInfo:
    """What the probe ran on. Not part of ``probe_sha256`` except the architecture."""

    os: str
    kernel_release: str
    architecture: str
    python_version: str
    wsl: bool
    container: bool
    root: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "os": self.os,
            "kernel_release": self.kernel_release,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "wsl": self.wsl,
            "container": self.container,
            "root": self.root,
        }

    @classmethod
    def from_dict(cls, value: Any) -> HostInfo:
        keys = ("os", "kernel_release", "architecture", "python_version", "wsl", "container", "root")
        _check_keys(value, keys, "host")
        return cls(
            os=_text(value["os"], "host.os", limit=64),
            kernel_release=_text(value["kernel_release"], "host.kernel_release", limit=128),
            architecture=_text(value["architecture"], "host.architecture", limit=32),
            python_version=_text(value["python_version"], "host.python_version", limit=32),
            wsl=_flag(value["wsl"], "host.wsl"),
            container=_flag(value["container"], "host.container"),
            root=_flag(value["root"], "host.root"),
        )


@dataclass(frozen=True)
class ToolInfo:
    """Interpreter, helper, and binary paths (home shown as ``~``) and the versions the probes found."""

    python: str
    helper: str
    bwrap_path: str | None
    bwrap_version: str | None
    libseccomp_available: bool
    libseccomp_version: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "helper": self.helper,
            "bwrap": {"path": self.bwrap_path, "version": self.bwrap_version},
            "libseccomp": {"available": self.libseccomp_available, "version": self.libseccomp_version},
        }


@dataclass(frozen=True)
class LandlockSummary:
    """The ABI the kernel advertised and the features actually exercised."""

    abi: int | None
    tested: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"abi": self.abi, "tested": list(self.tested)}


@dataclass(frozen=True)
class CgroupSummary:
    """The cgroup2 mount, the configured parent, and what the leaf test proved."""

    mount: str | None
    parent: str | None
    delegated: bool
    controllers: tuple[str, ...]
    leaf_created: bool
    limits_written: bool
    kill_verified: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "mount": self.mount,
            "parent": self.parent,
            "delegated": self.delegated,
            "controllers": list(self.controllers),
            "leaf_created": self.leaf_created,
            "limits_written": self.limits_written,
            "kill_verified": self.kill_verified,
        }


REPORT_KEYS = (
    "schema_version",
    "created_at",
    "host",
    "tools",
    "capabilities",
    "landlock",
    "cgroup",
    "tiers",
    "selected_tier",
    "probe_sha256",
)


def semantic_hash(architecture: str, capabilities: Sequence[Capability]) -> str:
    """``probe_sha256``: schema, architecture, and each record's name/status/code/facts."""
    return canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "architecture": architecture,
            "capabilities": [capability.semantic() for capability in sorted(capabilities, key=lambda c: c.name)],
        }
    )


def _fact(capabilities: Mapping[str, Capability], name: str, key: str) -> Any:
    capability = capabilities[name]
    return capability.facts.get(key) if capability.ok else None


@dataclass(frozen=True)
class ProbeReport:
    """The versioned result of one probe run. Build it with ``ProbeReport.build``."""

    schema_version: int
    created_at: str
    host: HostInfo
    tools: ToolInfo
    capabilities: tuple[Capability, ...]
    landlock: LandlockSummary
    cgroup: CgroupSummary
    tiers: tuple[TierEvaluation, ...]
    selected_tier: SandboxTier
    probe_sha256: str

    @classmethod
    def build(
        cls,
        *,
        created_at: str,
        host: HostInfo,
        python: str,
        helper: str,
        bwrap_path: str | None,
        capabilities: Iterable[Capability],
        cgroup_mount: str | None,
        cgroup_parent: str | None,
    ) -> ProbeReport:
        """Assemble a report and derive every summary, the tier, and the hash from the records."""
        if not isinstance(created_at, str) or not _TIMESTAMP.fullmatch(created_at):
            raise ProbeReportError("must be a UTC timestamp like 2026-11-09T12:00:00Z", field="created_at")
        _text(python, "tools.python", limit=MAX_PATH_CHARS)
        _text(helper, "tools.helper", limit=MAX_PATH_CHARS)
        _text(bwrap_path, "tools.bwrap.path", limit=MAX_PATH_CHARS, optional=True)
        _text(cgroup_mount, "cgroup.mount", limit=MAX_PATH_CHARS, optional=True)
        _text(cgroup_parent, "cgroup.parent", limit=MAX_PATH_CHARS, optional=True)

        records = tuple(capabilities)
        names = [record.name for record in records]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ProbeReportError(f"duplicate capability record(s) {duplicates}", field="capabilities")
        missing = [name for name in CAPABILITY_NAMES if name not in names]
        if missing:
            raise ProbeReportError(f"missing capability record(s) {missing}", field="capabilities")
        order = {name: index for index, name in enumerate(CAPABILITY_NAMES)}
        records = tuple(sorted(records, key=lambda record: order[record.name]))
        by_name = {record.name: record for record in records}

        tools = ToolInfo(
            python=python,
            helper=helper,
            bwrap_path=bwrap_path,
            bwrap_version=_fact(by_name, "bwrap.binary", "version"),
            libseccomp_available=by_name["libseccomp.library"].ok,
            libseccomp_version=_fact(by_name, "libseccomp.library", "version"),
        )
        landlock = LandlockSummary(
            abi=_fact(by_name, "landlock.abi", "abi"),
            tested=tuple(feature for name, feature in LANDLOCK_FEATURES.items() if by_name[name].ok),
        )
        cgroup = CgroupSummary(
            mount=cgroup_mount,
            parent=cgroup_parent,
            delegated=by_name["cgroup.delegation"].ok,
            controllers=tuple(_fact(by_name, "cgroup.delegation", "enabled") or ()),
            leaf_created=by_name["cgroup.leaf"].ok,
            limits_written=by_name["cgroup.limits"].ok,
            kill_verified=by_name["cgroup.kill"].ok,
        )
        tiers = evaluate_tiers(records)
        return cls(
            schema_version=SCHEMA_VERSION,
            created_at=created_at,
            host=host,
            tools=tools,
            capabilities=records,
            landlock=landlock,
            cgroup=cgroup,
            tiers=tiers,
            selected_tier=select_tier(tiers),
            probe_sha256=semantic_hash(host.architecture, records),
        )

    # Lookups

    def capability(self, name: str) -> Capability:
        for record in self.capabilities:
            if record.name == name:
                return record
        raise KeyError(name)

    def evaluation(self, tier: SandboxTier | str) -> TierEvaluation:
        tier = _tier(tier, "tier")
        return next(evaluation for evaluation in self.tiers if evaluation.tier is tier)

    def supports(self, tier: SandboxTier | str) -> bool:
        """Every requirement of ``tier`` holds. A higher selected tier does not imply a lower one."""
        return self.evaluation(tier).satisfied

    def require(self, tier: SandboxTier | str) -> TierEvaluation:
        """Confirm the caller's required tier, or raise. Never falls back to a lower tier."""
        evaluation = self.evaluation(tier)
        if not evaluation.satisfied:
            raise UnsupportedTierError(
                evaluation.tier,
                evaluation.missing,
                reason=f"{evaluation.tier.value} ({evaluation.tier.label}) is not supported on this host: {evaluation.summary()}",
            )
        return evaluation

    def require_risky_execution(self, tier: SandboxTier | str) -> TierEvaluation:
        """Gate for exec and network tools: T0 is refused outright, any other tier must be proven."""
        tier = _tier(tier, "tier")
        if not tier.is_sandbox:
            raise UnsupportedTierError(
                tier,
                (),
                reason="T0 is NOT a sandbox: risky exec and network tools need a proven T1 or T2",
            )
        return self.require(tier)

    # Serialization

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "host": self.host.as_dict(),
            "tools": self.tools.as_dict(),
            "capabilities": [record.as_dict() for record in self.capabilities],
            "landlock": self.landlock.as_dict(),
            "cgroup": self.cgroup.as_dict(),
            "tiers": {evaluation.tier.value: evaluation.as_dict() for evaluation in self.tiers},
            "selected_tier": self.selected_tier.value,
            "probe_sha256": self.probe_sha256,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> ProbeReport:
        return cls.from_dict(strict_json_loads(text))

    @classmethod
    def from_dict(cls, value: Any) -> ProbeReport:
        """Parse strictly, rebuild from the inputs, and refuse any field that disagrees with the rebuild."""
        _check_keys(value, REPORT_KEYS, "report")
        if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
            raise ProbeReportError(f"unsupported schema version {value['schema_version']!r}", field="schema_version")
        host = HostInfo.from_dict(value["host"])

        tools = _check_keys(value["tools"], ("python", "helper", "bwrap", "libseccomp"), "tools")
        _check_keys(tools["bwrap"], ("path", "version"), "tools.bwrap")
        _check_keys(tools["libseccomp"], ("available", "version"), "tools.libseccomp")

        raw_capabilities = value["capabilities"]
        if not isinstance(raw_capabilities, list):
            raise ProbeReportError("must be a list", field="capabilities")
        capabilities = [Capability.from_dict(item, f"capabilities[{index}]") for index, item in enumerate(raw_capabilities)]

        _check_keys(value["landlock"], ("abi", "tested"), "landlock")
        cgroup = _check_keys(
            value["cgroup"],
            ("mount", "parent", "delegated", "controllers", "leaf_created", "limits_written", "kill_verified"),
            "cgroup",
        )
        tiers = _check_keys(value["tiers"], [tier.value for tier in TIERS_DESCENDING], "tiers")
        for tier_name, evaluation in tiers.items():
            _check_keys(evaluation, ("label", "satisfied", "missing"), f"tiers.{tier_name}")
            if not isinstance(evaluation["missing"], list):
                raise ProbeReportError("must be a list", field=f"tiers.{tier_name}.missing")
            for index, item in enumerate(evaluation["missing"]):
                where = f"tiers.{tier_name}.missing[{index}]"
                _check_keys(item, ("capability", "status", "code", "detail"), where)
                _status(item["status"], f"{where}.status")
        _tier(value["selected_tier"], "selected_tier")
        if not isinstance(value["probe_sha256"], str) or not _SHA256.fullmatch(value["probe_sha256"]):
            raise ProbeReportError("must be 64 lowercase hex characters", field="probe_sha256")

        rebuilt = cls.build(
            created_at=value["created_at"],
            host=host,
            python=tools["python"],
            helper=tools["helper"],
            bwrap_path=tools["bwrap"]["path"],
            capabilities=capabilities,
            cgroup_mount=cgroup["mount"],
            cgroup_parent=cgroup["parent"],
        )
        expected = rebuilt.as_dict()
        if raw_capabilities != expected["capabilities"]:
            raise ProbeReportError("records are not in canonical order or form", field="capabilities")
        for key in ("tools", "landlock", "cgroup"):
            if value[key] != expected[key]:
                raise ProbeReportError("contradicts the capability evidence", field=key)
        if value["tiers"] != expected["tiers"] or value["selected_tier"] != expected["selected_tier"]:
            raise ProbeReportError(
                f"inconsistent tier evidence: the records select {expected['selected_tier']}, "
                f"the report claims {value['selected_tier']}",
                field="tiers",
            )
        if value["probe_sha256"] != expected["probe_sha256"]:
            raise ProbeReportError("does not match the semantic capabilities", field="probe_sha256")
        return rebuilt


__all__ = [
    "CAPABILITY_DESCRIPTIONS",
    "CAPABILITY_NAMES",
    "SCHEMA_VERSION",
    "TIER_LABELS",
    "TIER_REQUIREMENTS",
    "TIERS_DESCENDING",
    "Capability",
    "CapabilityStatus",
    "CgroupSummary",
    "HostInfo",
    "LandlockSummary",
    "MissingRequirement",
    "ProbeReport",
    "SandboxTier",
    "TierEvaluation",
    "ToolInfo",
    "canonical_hash",
    "canonical_json",
    "evaluate_tiers",
    "select_tier",
    "semantic_hash",
    "strict_json_loads",
]
