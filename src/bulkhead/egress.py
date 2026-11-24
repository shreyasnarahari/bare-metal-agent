"""Egress decisions: the only network path out of a sandbox is an HTTP CONNECT to the outside proxy.

Everything here is pure: parsing a CONNECT request, normalizing its
authority, classifying addresses, and deciding. ``bulkhead.proxy`` does the
I/O and asks this module before every resolution and connection.

- Only ``CONNECT host:port HTTP/1.x`` is accepted. Absolute-form proxy
  requests (``GET http://...``), transparent TCP, UDP, and DNS are not
  offered at all. HTTPS stays end-to-end encrypted: the proxy never sees
  paths, headers, or redirects inside TLS, so a redirect to another authority
  is a new CONNECT and is decided on its own.
- Authorities are parsed strictly. Userinfo, ``@``, paths, controls,
  whitespace, missing or ambiguous ports, malformed or zone-scoped IPv6, and
  non-canonical numeric hosts (``127.1``, ``2130706433``, ``0x7f000001``,
  ``0177.0.0.1``) are refused rather than handed to a resolver that would
  accept them. DNS names are lowercased, IDNA-encoded where needed, and lose a
  trailing dot.
- Destinations match the policy's exact ``host:port`` entries. There are no
  wildcards and no suffix matching.
- A raw IP literal is refused unless that exact ``ip:port`` is allowed. Any
  address, literal or resolved, must also be in an allowed address class
  (``public`` by default), so an allowed name cannot be pointed at loopback,
  private networks, or cloud metadata. The proxy connects to the address it
  validated, never re-resolving, so a later DNS change cannot redirect it.

Decisions carry stable rule ids for the ``egress_decision`` journal event. A
request that cannot even be parsed is recorded under ``PLACEHOLDER_AUTHORITY``,
never by echoing what the client sent.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Any

from .errors import BulkheadError
from .manifest import Egress, SecuritySpecError, normalize_authority

MAX_HEADER_BYTES = 8192
MAX_HEADER_LINES = 64
PLACEHOLDER_AUTHORITY = "unparseable.invalid:1"
_TOKEN = re.compile(r"[A-Z]{3,10}")
_VERSION = re.compile(r"HTTP/1\.[01]")
_FORBIDDEN = re.compile(r"[\x00-\x20\x7f@/?#\\*]")

# Rule ids (stable; they appear in egress_decision records and proxy responses).
ALLOW = "bulkhead.egress.allow"
MALFORMED = "bulkhead.egress.malformed"
METHOD = "bulkhead.egress.method"
HEADERS = "bulkhead.egress.headers"
NOT_ALLOWED = "bulkhead.egress.not_allowed"
IP_LITERAL = "bulkhead.egress.ip_literal"
ADDRESS_CLASS = "bulkhead.egress.address_class"
UNRESOLVED = "bulkhead.egress.unresolved"
CONNECTIONS = "bulkhead.egress.connections"
UPSTREAM = "bulkhead.egress.upstream_failed"
BYTES_OUT = "bulkhead.egress.bytes_out"
BYTES_IN = "bulkhead.egress.bytes_in"
IDLE = "bulkhead.egress.idle"
TIME = "bulkhead.egress.time"
# Rules after which a connection was allowed (the decision is "allow"; the rule says how it ended).
ALLOWED_RULES = frozenset({ALLOW, UPSTREAM, BYTES_OUT, BYTES_IN, IDLE, TIME})


class EgressError(BulkheadError, ValueError):
    """A request the proxy refuses before any resolution. ``rule`` is its stable rule id."""

    def __init__(self, message: str, *, rule: str) -> None:
        super().__init__(message)
        self.rule = rule


@dataclass(frozen=True)
class Authority:
    """A canonical ``host:port``: a lowercase ASCII DNS name, a dotted IPv4 address, or a bracketed IPv6 address."""

    host: str
    port: int

    @property
    def text(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def address(self) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
        """The literal address, or None for a DNS name."""
        try:
            return ipaddress.ip_address(self.host.strip("[]"))
        except ValueError:
            return None


def parse_authority(target: Any) -> Authority:
    """Strictly parse a CONNECT target into a canonical authority, or raise ``EgressError``."""
    if not isinstance(target, str) or not target or len(target) > 300:
        raise EgressError("the CONNECT target must be a host:port", rule=MALFORMED)
    if not target.isascii():
        host, separator, port = target.rpartition(":")
        try:
            target = host.rstrip(".").encode("idna").decode("ascii") + separator + port
        except UnicodeError:
            raise EgressError("the host is not a valid internationalized name", rule=MALFORMED) from None
    if _FORBIDDEN.search(target) or "%" in target:
        raise EgressError("the CONNECT target must be a bare host:port", rule=MALFORMED)
    try:
        canonical = normalize_authority(target, "CONNECT target")
    except SecuritySpecError as exc:
        raise EgressError(str(exc), rule=MALFORMED) from None
    host, _, port = canonical.rpartition(":")
    return Authority(host, int(port))


def parse_connect(head: bytes) -> tuple[Authority, bytes]:
    """Parse a proxy request head (``... \\r\\n\\r\\n``); return the authority and any bytes sent after it.

    Header values are never returned, logged, or interpreted: a client cannot
    name its own policy, episode, or credentials here.
    """
    end = head.find(b"\r\n\r\n")
    if end < 0 or end + 4 > MAX_HEADER_BYTES:
        raise EgressError("the request head is incomplete or too large", rule=HEADERS)
    lines = head[:end].split(b"\r\n")
    if len(lines) > MAX_HEADER_LINES:
        raise EgressError("too many header lines", rule=HEADERS)
    try:
        request_line = lines[0].decode("ascii")
    except UnicodeDecodeError:
        raise EgressError("the request line is not ASCII", rule=MALFORMED) from None
    parts = request_line.split(" ")
    if len(parts) != 3 or not _TOKEN.fullmatch(parts[0]) or not _VERSION.fullmatch(parts[2]):
        raise EgressError("the request line is malformed", rule=MALFORMED)
    if parts[0] != "CONNECT":
        raise EgressError("only CONNECT is supported", rule=METHOD)
    for line in lines[1:]:
        if b":" not in line or any(byte < 0x20 and byte != 0x09 for byte in line) or b"\x7f" in line:
            raise EgressError("a header line is malformed", rule=HEADERS)
    return parse_authority(parts[1]), head[end + 4:]


def classify(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """The address class of an IP address (IPv4-mapped IPv6 is classified as its IPv4 address)."""
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address.is_loopback:
        return "loopback"
    if address.is_link_local:
        return "link_local"
    if address.is_multicast:
        return "multicast"
    if address.is_unspecified or address.is_reserved:
        return "reserved"
    if address.is_private:
        return "private"
    if address.is_global:
        return "public"
    return "reserved"


@dataclass(frozen=True)
class Decision:
    """One decision. ``address`` is the pinned IP to connect to when allowed."""

    allowed: bool
    rule_id: str
    authority: str
    address_class: str
    address: str | None = None


@dataclass(frozen=True)
class EgressPolicy:
    """One call's egress policy, bound by the supervisor from the sandbox spec (never from the client)."""

    allow: frozenset[str]
    address_classes: frozenset[str]
    max_bytes_out: int
    max_bytes_in: int
    max_connections: int = 8
    max_requests: int = 64
    idle_s: float = 10.0
    connect_timeout_s: float = 5.0

    @classmethod
    def from_egress(cls, egress: Egress, **options: Any) -> EgressPolicy:
        return cls(frozenset(egress.allow), frozenset(egress.address_classes), egress.max_bytes_out, egress.max_bytes_in, **options)

    def decide(self, authority: Authority) -> Decision:
        """Before resolution: exact allowlist membership, and the class of an IP literal."""
        literal = authority.address
        if authority.text not in self.allow:
            return Decision(False, IP_LITERAL if literal is not None else NOT_ALLOWED, authority.text, "ip_literal" if literal is not None else "unresolved")
        if literal is not None:
            address_class = classify(literal)
            if address_class not in self.address_classes:
                return Decision(False, ADDRESS_CLASS, authority.text, address_class)
            return Decision(True, ALLOW, authority.text, address_class, str(literal))
        return Decision(True, ALLOW, authority.text, "unresolved")

    def choose(self, authority: Authority, addresses: list[str]) -> Decision:
        """After resolution: pin the first address in an allowed class, or refuse."""
        if not addresses:
            return Decision(False, UNRESOLVED, authority.text, "unresolved")
        first_class = None
        for text in addresses:
            try:
                address = ipaddress.ip_address(text)
            except ValueError:
                continue
            address_class = classify(address)
            first_class = first_class or address_class
            if address_class in self.address_classes:
                return Decision(True, ALLOW, authority.text, address_class, str(address))
        return Decision(False, ADDRESS_CLASS, authority.text, first_class or "unresolved")


@dataclass
class EgressRecord:
    """One proxied request, as the ``egress_decision`` journal event records it."""

    request_id: str
    authority: str
    decision: str
    rule_id: str
    resolved_address_class: str
    bytes_in: int = 0
    bytes_out: int = 0
    truncated: bool = False
    elapsed_s: float = 0.0

    def as_audit(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "authority": self.authority, "decision": self.decision, "rule_id": self.rule_id,
            "resolved_address_class": self.resolved_address_class, "bytes_in": self.bytes_in, "bytes_out": self.bytes_out,
            "truncated": self.truncated, "elapsed_s": round(self.elapsed_s, 6),
        }


__all__ = [
    "ALLOW",
    "ALLOWED_RULES",
    "Authority",
    "Decision",
    "EgressError",
    "EgressPolicy",
    "EgressRecord",
    "MAX_HEADER_BYTES",
    "PLACEHOLDER_AUTHORITY",
    "classify",
    "parse_authority",
    "parse_connect",
]
