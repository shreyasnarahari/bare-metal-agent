"""Bulkhead part 6: pure egress decisions (strict CONNECT parsing, address classes, exact allowlisting)."""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path

import pytest

from bulkhead import audit
from bulkhead.egress import (
    ADDRESS_CLASS,
    ALLOW,
    HEADERS,
    IP_LITERAL,
    MALFORMED,
    METHOD,
    NOT_ALLOWED,
    UNRESOLVED,
    EgressError,
    EgressPolicy,
    EgressRecord,
    classify,
    parse_authority,
    parse_connect,
)
from bulkhead.errors import SecuritySpecError
from bulkhead.manifest import Egress, normalize_authority
from bulkhead.narrowing import egress_widening

TABLE = json.loads((Path(__file__).parent / "fixtures" / "bulkhead_network" / "authorities.json").read_text())


@pytest.mark.parametrize(("target", "canonical"), TABLE["accepted"])
def test_accepted_targets_normalize_canonically(target, canonical):
    assert parse_authority(target).text == canonical


@pytest.mark.parametrize(("target", "rule", "why"), TABLE["refused"], ids=[row[2] for row in TABLE["refused"]])
def test_refused_targets_never_reach_a_resolver(target, rule, why):
    with pytest.raises(EgressError) as raised:
        parse_authority(target)
    assert raised.value.rule == rule


def test_noncanonical_numeric_hosts_are_refused_in_specs_too():
    for text in ("127.1:80", "2130706433:80", "0x7f000001:80", "[fe80::1%eth0]:80"):
        with pytest.raises(SecuritySpecError):
            normalize_authority(text)
    with pytest.raises(SecuritySpecError):
        Egress(allow=("2130706433:80",))


# Request heads


def test_a_connect_head_yields_only_the_authority_and_early_bytes():
    authority, early = parse_connect(b"CONNECT Example.com:443 HTTP/1.1\r\nHost: x\r\nProxy-Authorization: Basic c2VjcmV0\r\nX-Bulkhead-Policy: allow-all\r\n\r\n\x16\x03\x01")
    assert authority.text == "example.com:443" and early == b"\x16\x03\x01"


@pytest.mark.parametrize(
    ("head", "rule"),
    [
        (b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n", METHOD),
        (b"POST example.com:443 HTTP/1.1\r\n\r\n", METHOD),
        (b"CONNECT example.com:443 HTTP/2.0\r\n\r\n", MALFORMED),
        (b"CONNECT  example.com:443 HTTP/1.1\r\n\r\n", MALFORMED),
        (b"connect example.com:443 HTTP/1.1\r\n\r\n", MALFORMED),
        (b"CONNECT example.com:443 HTTP/1.1\r\n", HEADERS),  # never finished
        (b"CONNECT example.com:443 HTTP/1.1\r\nX: " + b"a" * 9000 + b"\r\n\r\n", HEADERS),
        (b"CONNECT example.com:443 HTTP/1.1\r\n" + b"X: y\r\n" * 70 + b"\r\n", HEADERS),
        (b"CONNECT example.com:443 HTTP/1.1\r\nno-colon\r\n\r\n", HEADERS),
        (b"CONNECT example.com:443 HTTP/1.1\r\nX: a\x00b\r\n\r\n", HEADERS),
        ("CONNECT bücher.de:443 HTTP/1.1\r\n\r\n".encode(), MALFORMED),  # raw UTF-8 on the request line
        (b"CONNECT 2130706433:80 HTTP/1.1\r\n\r\n", MALFORMED),
    ],
)
def test_malformed_heads_are_refused_with_a_rule(head, rule):
    with pytest.raises(EgressError) as raised:
        parse_connect(head)
    assert raised.value.rule == rule


# Address classes


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("93.184.216.34", "public"), ("2606:2800:220:1::1", "public"), ("10.1.2.3", "private"), ("192.168.1.1", "private"),
        ("172.16.0.1", "private"), ("fd00::1", "private"), ("127.0.0.1", "loopback"), ("::1", "loopback"),
        ("169.254.169.254", "link_local"), ("fe80::1", "link_local"), ("224.0.0.251", "multicast"), ("ff02::1", "multicast"),
        ("0.0.0.0", "reserved"), ("240.0.0.1", "reserved"), ("100.64.0.1", "reserved"),
        ("::ffff:127.0.0.1", "loopback"), ("::ffff:10.0.0.1", "private"), ("::ffff:169.254.169.254", "link_local"),
    ],
)
def test_addresses_are_classified(address, expected):
    assert classify(ipaddress.ip_address(address)) == expected


# Policy decisions


POLICY = EgressPolicy(frozenset({"api.example.com:443", "10.0.0.5:8080", "127.0.0.1:9000"}), frozenset({"public"}), 1000, 1000)


@pytest.mark.parametrize(
    ("target", "allowed", "rule", "address_class"),
    [
        ("api.example.com:443", True, ALLOW, "unresolved"),
        ("api.example.com:80", False, NOT_ALLOWED, "unresolved"),
        ("evil.example.com:443", False, NOT_ALLOWED, "unresolved"),
        ("example.com:443", False, NOT_ALLOWED, "unresolved"),  # no suffix matching
        ("8.8.8.8:443", False, IP_LITERAL, "ip_literal"),
        ("[2001:4860::8888]:443", False, IP_LITERAL, "ip_literal"),
        ("10.0.0.5:8080", False, ADDRESS_CLASS, "private"),  # listed, but its class is not allowed
        ("127.0.0.1:9000", False, ADDRESS_CLASS, "loopback"),
    ],
)
def test_authority_decisions_are_exact(target, allowed, rule, address_class):
    decision = POLICY.decide(parse_authority(target))
    assert (decision.allowed, decision.rule_id, decision.address_class) == (allowed, rule, address_class)


def test_a_listed_literal_in_an_allowed_class_is_pinned():
    local = EgressPolicy(frozenset({"127.0.0.1:9000"}), frozenset({"loopback"}), 1000, 1000)
    decision = local.decide(parse_authority("127.0.0.1:9000"))
    assert decision.allowed and decision.address == "127.0.0.1" and decision.address_class == "loopback"


def test_resolution_is_checked_and_pinned():
    authority = parse_authority("api.example.com:443")
    public = POLICY.choose(authority, ["93.184.216.34"])
    assert public.allowed and public.address == "93.184.216.34"
    # An allowed name that resolves to metadata, loopback, or a private range is refused.
    for addresses, address_class in ((["169.254.169.254"], "link_local"), (["127.0.0.1"], "loopback"), (["10.9.9.9", "fd00::2"], "private")):
        decision = POLICY.choose(authority, addresses)
        assert (decision.allowed, decision.rule_id, decision.address_class) == (False, ADDRESS_CLASS, address_class)
    mixed = POLICY.choose(authority, ["10.0.0.1", "93.184.216.34"])  # the first allowed address is pinned
    assert mixed.allowed and mixed.address == "93.184.216.34"
    assert POLICY.choose(authority, []).rule_id == UNRESOLVED


def test_a_rebinding_answer_is_judged_on_its_own():
    authority = parse_authority("api.example.com:443")
    answers = iter([["93.184.216.34"], ["127.0.0.1"]])
    first, second = POLICY.choose(authority, next(answers)), POLICY.choose(authority, next(answers))
    assert first.allowed and not second.allowed and second.address_class == "loopback"


# The spec and narrowing


def test_address_classes_default_to_public_and_keep_old_hashes():
    assert Egress().address_classes == ("public",)
    assert set(Egress(allow=("a.example:443",)).as_dict()) == {"allow", "max_bytes_out", "max_bytes_in"}
    local = Egress(allow=("a.example:443",), address_classes=("loopback", "public"))
    assert local.as_dict()["address_classes"] == ["loopback", "public"]
    assert Egress.from_dict(json.loads(json.dumps(local.as_dict()))) == local
    for bad in ((), ("everywhere",), ("ip_literal",)):
        with pytest.raises(SecuritySpecError):
            Egress(allow=("a.example:443",), address_classes=bad)


def test_adding_an_address_class_is_a_widening():
    task = Egress(allow=("a.example:443",))
    assert egress_widening(Egress(allow=("a.example:443",), address_classes=("public", "private")), task) == {"address_classes": ["private"]}
    assert egress_widening(task, task) == {}


def test_records_are_valid_egress_decision_payloads():
    record = EgressRecord("req_0", "api.example.com:443", "allow", ALLOW, "public", 10, 20, False, 0.5)
    payload = audit.build("egress_decision", step=0, tool_call_id="call_0", **record.as_audit())
    assert payload["authority"] == "api.example.com:443" and set(payload) == set(audit.SECURITY_EVENT_FIELDS["egress_decision"])
