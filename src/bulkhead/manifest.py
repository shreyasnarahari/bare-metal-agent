"""Security metadata: what a tool's sandbox may read, write, and reach, and what it may consume.

Every type here is an immutable dataclass validated at construction, with one
canonical JSON form (``as_dict``), a strict loader (``from_dict``: unknown
fields, unknown enum values, and non-finite numbers are refused), and a stable
``sha256``. Invalid or contradictory metadata raises ``SecuritySpecError``
when it is built, so a registry can never hold a spec that later turns into
"allow everything".

Paths are written either absolute (``/usr``) or under the explicit workspace
token (``$WORKSPACE/out``). Separators are normalized, repeated or trailing
slashes collapse, and ``.``/``..`` components are refused rather than
resolved. This is a decision aid for the policy engine, not a filesystem
boundary: lexical normalization says nothing about symlinks or hardlinks, so
the OS sandbox (part 4) enforces the final access.

Network destinations are exact ``host:port`` authorities, never URL prefixes
or wildcards. An empty ``Egress.allow`` means no network at all.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

from .errors import SecuritySpecError
from .taint import Sink, Trust
from .types import canonical_hash

WORKSPACE = "$WORKSPACE"
MAX_PATHS = 64
MAX_PATH_CHARS = 512
MAX_AUTHORITIES = 64
# Hard ceilings no spec or task policy may exceed.
GLOBAL_MAXIMA: dict[str, float] = {
    "cpu_s": 600,
    "wall_s": 3600,
    "mem_mb": 16384,
    "pids": 4096,
    "disk_mb": 20480,
    "out_kb": 8192,
}
MAX_EGRESS_BYTES = 64 * 1024 * 1024
DEFAULT_PORTS = {"http": 80, "https": 443}

_LABEL = re.compile(r"(?!-)[a-z0-9-]{1,63}(?<!-)")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


# Trust and Sink live in bulkhead.taint and are re-exported here for spec authors.
FILESYSTEM_SINKS = (Sink.FILESYSTEM, Sink.FILESYSTEM_SENSITIVE)


class PathAccess(str, Enum):
    """How a path-valued tool argument is used."""

    READ = "read"
    WRITE = "write"


# Strict construction helpers


def check_keys(value: Any, expected: Iterable[str], where: str, *, optional: Iterable[str] = (), error=SecuritySpecError) -> Mapping[str, Any]:
    """``value`` is a mapping with every expected key, optionally some optional keys, and nothing else."""
    if not isinstance(value, Mapping):
        raise error(f"{where} must be a JSON object")
    expected, optional = tuple(expected), tuple(optional)
    unknown = sorted(set(value) - set(expected) - set(optional))
    missing = [key for key in expected if key not in value]
    if unknown:
        raise error(f"{where} has unknown field(s) {unknown}")
    if missing:
        raise error(f"{where} is missing field(s) {missing}")
    return value


def _sequence(value: Any, where: str, *, limit: int, error=SecuritySpecError) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise error(f"{where} must be a list")
    items = tuple(value)
    if len(items) > limit:
        raise error(f"{where} has more than {limit} entries")
    return items


def _enum(kind: type[Enum], value: Any, where: str, *, error=SecuritySpecError) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(value)
    except ValueError:
        raise error(f"{where}: unknown {kind.__name__} {value!r}") from None


def number(value: Any, where: str, *, integer: bool = False, error=SecuritySpecError) -> int | float:
    """A finite number (never a bool). Integer-valued floats become ints so hashes are stable."""
    if type(value) is bool or not isinstance(value, (int, float)):
        raise error(f"{where} must be a number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error(f"{where} must be finite")
        if value.is_integer():
            value = int(value)
        elif integer:
            raise error(f"{where} must be a whole number")
    return value


# Paths


def normalize_path(text: Any, where: str = "path", *, error=SecuritySpecError) -> str:
    """Canonical form of an absolute or ``$WORKSPACE`` path; ambiguity is refused, not resolved."""
    if not isinstance(text, str) or not text:
        raise error(f"{where} must be a non-empty string")
    if len(text) > MAX_PATH_CHARS or _CONTROL.search(text):
        raise error(f"{where} is too long or contains control characters")
    if "\\" in text:
        raise error(f"{where} must use forward slashes")
    if text == WORKSPACE:
        return WORKSPACE
    if text.startswith(WORKSPACE + "/"):
        root, rest = WORKSPACE, text[len(WORKSPACE) + 1 :]
    elif text.startswith("/"):
        root, rest = "", text[1:]
    else:
        raise error(f"{where} must start with / or {WORKSPACE}")
    if "$" in rest:
        raise error(f"{where}: only a leading {WORKSPACE} is expanded")
    parts = [part for part in rest.split("/") if part]
    if any(part in (".", "..") for part in parts):
        raise error(f"{where} must not contain . or .. components")
    if not parts:
        return root or "/"
    return f"{root}/{'/'.join(parts)}"


def normalize_argument_path(text: Any, where: str = "path", *, error=SecuritySpecError) -> str:
    """A path a tool receives as an argument. A relative path is taken relative to the workspace."""
    if isinstance(text, str) and text and not text.startswith(("/", "$")):
        text = f"{WORKSPACE}/{text}"
    return normalize_path(text, where, error=error)


def beneath(path: str, parent: str) -> bool:
    """``path`` is ``parent`` or inside it. ``$WORKSPACE`` and ``/`` are separate roots."""
    if path == parent:
        return True
    if parent == "/":
        return path.startswith("/")
    return path.startswith(parent + "/")


def beneath_any(path: str, parents: Iterable[str]) -> bool:
    return any(beneath(path, parent) for parent in parents)


def normalize_paths(values: Any, where: str, *, error=SecuritySpecError) -> tuple[str, ...]:
    items = _sequence(values, where, limit=MAX_PATHS, error=error)
    return tuple(sorted({normalize_path(item, f"{where}[{index}]", error=error) for index, item in enumerate(items)}))


# Network authorities


def normalize_authority(text: Any, where: str = "authority", *, error=SecuritySpecError) -> str:
    """Canonical ``host:port``: lowercase host, IP literals in canonical form, an explicit port."""
    if not isinstance(text, str) or not text or len(text) > 300:
        raise error(f"{where} must be a host:port string")
    if any(character in text for character in "/@*?# ") or _CONTROL.search(text):
        raise error(f"{where} must be a bare host:port (no scheme, path, user, or wildcard)")
    if text.startswith("["):
        host, separator, port = text[1:].partition("]:")
        try:
            host = f"[{ipaddress.IPv6Address(host).compressed}]"
        except ValueError:
            raise error(f"{where} has an invalid IPv6 literal") from None
    else:
        host, separator, port = text.rpartition(":")
        host = host.lower().rstrip(".")
        try:
            host = str(ipaddress.IPv4Address(host))
        except ValueError:
            labels = host.split(".")
            if not host or len(host) > 253 or not all(_LABEL.fullmatch(label) for label in labels):
                raise error(f"{where} has an invalid host name") from None
    if not separator or not port.isdigit() or port != str(int(port)) or not 1 <= int(port) <= 65535:
        raise error(f"{where} needs an explicit port from 1 to 65535")
    return f"{host}:{int(port)}"


def destination_authority(value: Any, where: str = "destination", *, error=SecuritySpecError) -> str:
    """The ``host:port`` an argument would reach: a ``host:port`` string or an http(s) URL."""
    if isinstance(value, str) and "://" in value:
        try:
            parts = urlsplit(value)
            port = parts.port
        except ValueError:
            raise error(f"{where} is not a parseable URL") from None
        if parts.scheme not in DEFAULT_PORTS or not parts.hostname or parts.username is not None or "@" in parts.netloc:
            raise error(f"{where} must be an http or https URL without credentials")
        host = parts.hostname if ":" not in parts.hostname else f"[{parts.hostname}]"
        return normalize_authority(f"{host}:{port or DEFAULT_PORTS[parts.scheme]}", where, error=error)
    return normalize_authority(value, where, error=error)


def normalize_authorities(values: Any, where: str, *, error=SecuritySpecError) -> tuple[str, ...]:
    items = _sequence(values, where, limit=MAX_AUTHORITIES, error=error)
    return tuple(sorted({normalize_authority(item, f"{where}[{index}]", error=error) for index, item in enumerate(items)}))


# Limits, egress, sandbox, spec


@dataclass(frozen=True)
class Limits:
    """Resource ceilings. ``out_kb`` caps stdout and stderr combined; each stream also has a subcap."""

    cpu_s: float
    mem_mb: int
    pids: int
    disk_mb: int
    out_kb: int
    wall_s: float = 30
    stdout_kb: int | None = None  # defaults to out_kb
    stderr_kb: int | None = None  # defaults to out_kb

    FIELDS = ("cpu_s", "mem_mb", "pids", "disk_mb", "out_kb", "wall_s", "stdout_kb", "stderr_kb")

    def __post_init__(self) -> None:
        for name in ("cpu_s", "wall_s", "mem_mb", "pids", "disk_mb", "out_kb"):
            value = number(getattr(self, name), f"limits.{name}", integer=name not in ("cpu_s", "wall_s"))
            if value <= 0 or value > GLOBAL_MAXIMA[name]:
                raise SecuritySpecError(f"limits.{name} must be positive and at most {GLOBAL_MAXIMA[name]:g}")
            object.__setattr__(self, name, value)
        for name in ("stdout_kb", "stderr_kb"):
            value = getattr(self, name)
            value = self.out_kb if value is None else number(value, f"limits.{name}", integer=True)
            if value <= 0 or value > self.out_kb:
                raise SecuritySpecError(f"limits.{name} must be positive and at most out_kb")
            object.__setattr__(self, name, value)

    def ceilings(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.FIELDS}

    def as_dict(self) -> dict[str, Any]:
        return self.ceilings()

    @classmethod
    def from_dict(cls, value: Any, where: str = "limits") -> Limits:
        check_keys(value, ("cpu_s", "mem_mb", "pids", "disk_mb", "out_kb"), where, optional=("wall_s", "stdout_kb", "stderr_kb"))
        return cls(**dict(value))


DEFAULT_LIMITS = Limits(cpu_s=10, mem_mb=512, pids=64, disk_mb=200, out_kb=64)


@dataclass(frozen=True)
class Egress:
    """Exact ``host:port`` destinations and byte ceilings. Empty ``allow`` means no network."""

    allow: tuple[str, ...] = ()
    max_bytes_out: int = 64 * 1024
    max_bytes_in: int = 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "allow", normalize_authorities(self.allow, "egress.allow"))
        for name in ("max_bytes_out", "max_bytes_in"):
            value = number(getattr(self, name), f"egress.{name}", integer=True)
            if value <= 0 or value > MAX_EGRESS_BYTES:
                raise SecuritySpecError(f"egress.{name} must be positive and at most {MAX_EGRESS_BYTES}")
            object.__setattr__(self, name, value)

    @property
    def enabled(self) -> bool:
        return bool(self.allow)

    def as_dict(self) -> dict[str, Any]:
        return {"allow": list(self.allow), "max_bytes_out": self.max_bytes_out, "max_bytes_in": self.max_bytes_in}

    @classmethod
    def from_dict(cls, value: Any, where: str = "egress") -> Egress:
        check_keys(value, ("allow",), where, optional=("max_bytes_out", "max_bytes_in"))
        return cls(**dict(value))


@dataclass(frozen=True)
class Sandbox:
    """The view a sandboxed tool gets. ``fs_deny`` wins over every read and write allowance."""

    fs_read: tuple[str, ...] = ()
    fs_write: tuple[str, ...] = ()
    fs_deny: tuple[str, ...] = ()
    net: Egress = field(default_factory=Egress)
    limits: Limits = DEFAULT_LIMITS
    secrets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("fs_read", "fs_write", "fs_deny"):
            object.__setattr__(self, name, normalize_paths(getattr(self, name), f"sandbox.{name}"))
        outside = [path for path in self.fs_write if not beneath(path, WORKSPACE)]
        if outside:
            raise SecuritySpecError(f"sandbox.fs_write must stay inside {WORKSPACE}; refused {outside}")
        if not isinstance(self.net, Egress) or not isinstance(self.limits, Limits):
            raise SecuritySpecError("sandbox.net must be an Egress and sandbox.limits a Limits")
        secrets = _sequence(self.secrets, "sandbox.secrets", limit=MAX_PATHS)
        if secrets:
            # Never echo the names or values: the refusal itself must not leak them.
            raise SecuritySpecError(
                f"sandbox.secrets must be empty: a sandboxed tool gets no credentials ({len(secrets)} given)"
            )
        object.__setattr__(self, "secrets", ())

    def denied(self, path: str) -> bool:
        return beneath_any(path, self.fs_deny)

    def readable(self, path: str) -> bool:
        return not self.denied(path) and beneath_any(path, (*self.fs_read, *self.fs_write))

    def writable(self, path: str) -> bool:
        return not self.denied(path) and beneath_any(path, self.fs_write)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fs_read": list(self.fs_read),
            "fs_write": list(self.fs_write),
            "fs_deny": list(self.fs_deny),
            "net": self.net.as_dict(),
            "limits": self.limits.as_dict(),
            "secrets": [],
        }

    @classmethod
    def from_dict(cls, value: Any, where: str = "sandbox") -> Sandbox:
        check_keys(value, ("fs_read", "fs_write", "fs_deny", "net", "limits", "secrets"), where)
        return cls(
            fs_read=value["fs_read"],
            fs_write=value["fs_write"],
            fs_deny=value["fs_deny"],
            net=Egress.from_dict(value["net"], f"{where}.net"),
            limits=Limits.from_dict(value["limits"], f"{where}.limits"),
            secrets=value["secrets"],
        )


@dataclass(frozen=True)
class SecuritySpec:
    """A tool's security metadata: its sandbox (``None`` for a trusted in-process tool), output trust, and sinks.

    Sinks must cover what the sandbox allows: network egress needs ``NETWORK``
    and filesystem writes need ``FILESYSTEM`` or ``FILESYSTEM_SENSITIVE``. An
    in-process tool cannot claim a network or filesystem sink; it must be given a
    sandbox first.
    """

    sandbox: Sandbox | None
    output_trust: Trust = Trust.UNTRUSTED
    sinks: tuple[Sink, ...] = ()

    def __post_init__(self) -> None:
        if self.sandbox is not None and not isinstance(self.sandbox, Sandbox):
            raise SecuritySpecError("sandbox must be a Sandbox or None")
        object.__setattr__(self, "output_trust", _enum(Trust, self.output_trust, "output_trust"))
        sinks = _sequence(self.sinks, "sinks", limit=len(Sink))
        normalized = tuple(sorted({_enum(Sink, sink, "sinks") for sink in sinks}, key=lambda sink: sink.value))
        object.__setattr__(self, "sinks", normalized)
        if self.sandbox is None:
            claimed = [sink.value for sink in normalized if sink is Sink.NETWORK or sink in FILESYSTEM_SINKS]
            if claimed:
                raise SecuritySpecError(f"an in-process tool cannot declare {claimed}; give it a sandbox")
            return
        if self.sandbox.net.enabled and Sink.NETWORK not in normalized:
            raise SecuritySpecError("a sandbox with egress must declare the network sink")
        if self.sandbox.fs_write and not set(normalized) & set(FILESYSTEM_SINKS):
            raise SecuritySpecError("a sandbox with writable paths must declare the filesystem sink")

    @property
    def sandboxed(self) -> bool:
        return self.sandbox is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sandbox": self.sandbox.as_dict() if self.sandbox is not None else None,
            "output_trust": self.output_trust.value,
            "sinks": [sink.value for sink in self.sinks],
        }

    def sha256(self) -> str:
        return canonical_hash(self.as_dict())

    @classmethod
    def from_dict(cls, value: Any, where: str = "security") -> SecuritySpec:
        check_keys(value, ("sandbox", "output_trust", "sinks"), where)
        sandbox = value["sandbox"]
        return cls(
            sandbox=Sandbox.from_dict(sandbox, f"{where}.sandbox") if sandbox is not None else None,
            output_trust=value["output_trust"],
            sinks=value["sinks"],
        )


@dataclass(frozen=True)
class Bounds:
    """Inclusive numeric bounds on one argument. At least one side is set."""

    minimum: int | float | None = None
    maximum: int | float | None = None

    def __post_init__(self) -> None:
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, number(value, f"bounds.{name}"))
        if self.minimum is None and self.maximum is None:
            raise SecuritySpecError("bounds need a minimum, a maximum, or both")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise SecuritySpecError("bounds.minimum must not exceed bounds.maximum")

    def contains(self, value: int | float) -> bool:
        return (self.minimum is None or value >= self.minimum) and (self.maximum is None or value <= self.maximum)

    def widened_to(self, value: int | float) -> Bounds:
        """The smallest bounds that keep these limits' sides and also admit ``value``."""
        minimum = None if self.minimum is None else min(self.minimum, value)
        maximum = None if self.maximum is None else max(self.maximum, value)
        return Bounds(minimum, maximum)

    def as_dict(self) -> dict[str, Any]:
        return {"minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_dict(cls, value: Any, where: str = "bounds", *, error=SecuritySpecError) -> Bounds:
        check_keys(value, ("minimum", "maximum"), where, error=error)
        return cls(value["minimum"], value["maximum"])


__all__ = [
    "DEFAULT_LIMITS",
    "DEFAULT_PORTS",
    "FILESYSTEM_SINKS",
    "GLOBAL_MAXIMA",
    "MAX_EGRESS_BYTES",
    "WORKSPACE",
    "Bounds",
    "Egress",
    "Limits",
    "PathAccess",
    "Sandbox",
    "SecuritySpec",
    "Sink",
    "Trust",
    "beneath",
    "beneath_any",
    "check_keys",
    "destination_authority",
    "normalize_argument_path",
    "normalize_authorities",
    "normalize_authority",
    "normalize_path",
    "normalize_paths",
    "number",
]
