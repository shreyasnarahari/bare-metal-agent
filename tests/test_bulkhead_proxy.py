"""Bulkhead part 6: the outside CONNECT proxy, against real loopback fixtures.

All destinations are local fixtures on 127.0.0.1. Names like ``fixture.test``
resolve through an injected resolver, so the policy must explicitly allow the
loopback address class. The HTTPS fixture uses a throwaway certificate made
with ``openssl`` at test time (the tests skip HTTPS if it is missing); nothing
targets a public service.
"""

from __future__ import annotations

import http.server
import os
import socket
import ssl
import stat
import subprocess
import sys
import threading
import time

import pytest

from bulkhead.egress import (
    ADDRESS_CLASS,
    ALLOW,
    BYTES_IN,
    BYTES_OUT,
    CONNECTIONS,
    HEADERS,
    IDLE,
    IP_LITERAL,
    MALFORMED,
    METHOD,
    NOT_ALLOWED,
    PLACEHOLDER_AUTHORITY,
    TIME,
    UPSTREAM,
    EgressPolicy,
)
from bulkhead.manifest import Egress
from bulkhead.proxy import EgressProxy, call_proxy, system_connector

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="unsupported: Unix sockets on Linux")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body=b"", headers=()):
        self.send_response(code)
        for key, value in headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        port = self.server.server_address[1]
        if self.path == "/big":
            self._send(200, b"b" * 50_000)
        elif self.path == "/redirect-other":
            self._send(302, headers=[("Location", f"http://other.test:{port}/")])
        elif self.path == "/redirect-same":
            self._send(302, headers=[("Location", "/")])
        elif self.path == "/abort":
            self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\npartial")
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
        else:
            self._send(200, b"hello from the fixture\n")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self._send(200, b"stored\n")

    def log_message(self, *args):
        pass


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def fixture_server():
    server = serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler))
    yield server.server_address[1]
    server.shutdown()


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tls")
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(directory / "key.pem"), "-out", str(directory / "cert.pem"),
             "-days", "1", "-subj", "/CN=fixture.test", "-addext", "subjectAltName=DNS:fixture.test"],
            check=True, capture_output=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("unsupported: openssl is needed for the throwaway HTTPS fixture")
    return directory / "cert.pem", directory / "key.pem"


@pytest.fixture
def tls_server(certificate):
    cert, key = certificate
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    serve(server)
    yield server.server_address[1]
    server.shutdown()


NAMES = {"fixture.test": ["127.0.0.1"], "other.test": ["127.0.0.1"], "private.test": ["10.0.0.7"], "meta.test": ["169.254.169.254"]}


def resolver(host, port):
    return list(NAMES.get(host, []))


@pytest.fixture
def make_proxy(tmp_path):
    started = []

    def make(allow, *, classes=("loopback",), resolve=resolver, connector=system_connector, **options):
        options = {"max_bytes_out": 100_000, "max_bytes_in": 100_000, **options}
        policy = EgressPolicy(frozenset(allow), frozenset(classes), options.pop("max_bytes_out"), options.pop("max_bytes_in"), **options)
        path = tmp_path / f"p{len(started)}.sock"
        proxy = EgressProxy(policy, path, resolver=resolve, connector=connector).start()
        started.append(proxy)
        return proxy

    yield make
    for proxy in started:
        proxy.stop()


def open_tunnel(proxy, target, extra=b"", timeout=5):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(str(proxy.socket_path))
    sock.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n".encode() + extra + b"\r\n")
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = sock.recv(4096)
        if not chunk:
            break
        head += chunk
    status_line, _, rest = head.partition(b"\r\n")
    return sock, status_line.decode(), rest


def read_all(sock):
    data = b""
    while True:
        try:
            chunk = sock.recv(65536)
        except (TimeoutError, ConnectionResetError):
            break
        if not chunk:
            break
        data += chunk
    return data


def get(sock, path, host="fixture.test"):
    sock.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}\r\nCookie: session=s3cret\r\nAuthorization: Bearer t0ken\r\nConnection: close\r\n\r\n".encode())
    return read_all(sock)


def only(proxy):
    [record] = proxy.records
    return record


# Allowed traffic


def test_an_allowed_http_tunnel_works_and_its_record_holds_no_request_data(make_proxy, fixture_server):
    proxy = make_proxy({f"fixture.test:{fixture_server}"})
    sock, status, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}", b"Proxy-Authorization: Basic c2VjcmV0\r\n")
    response = get(sock, "/private/path?token=abc123")
    sock.close()
    proxy.stop()

    assert status == "HTTP/1.1 200 Connection established" and response.endswith(b"hello from the fixture\n")
    record = only(proxy)
    assert (record.decision, record.rule_id, record.resolved_address_class, record.truncated) == ("allow", ALLOW, "loopback", False)
    assert record.bytes_out > 0 and record.bytes_in == len(response)
    text = repr(record.as_audit())
    for leaked in ("token", "abc123", "private/path", "s3cret", "t0ken", "c2VjcmV0", "Cookie"):
        assert leaked not in text


def test_https_passes_through_encrypted_with_explicit_certificate_handling(make_proxy, tls_server, certificate):
    proxy = make_proxy({f"fixture.test:{tls_server}"})
    sock, status, _ = open_tunnel(proxy, f"fixture.test:{tls_server}")
    assert status.endswith("200 Connection established")
    pinned = ssl.create_default_context(cafile=str(certificate[0]))
    with pinned.wrap_socket(sock, server_hostname="fixture.test") as tls:
        assert get(tls, "/").endswith(b"hello from the fixture\n")
    # Without the fixture's CA the client must refuse the certificate; the proxy never touches TLS.
    sock, status, _ = open_tunnel(proxy, f"fixture.test:{tls_server}")
    with pytest.raises(ssl.SSLCertVerificationError):
        ssl.create_default_context().wrap_socket(sock, server_hostname="fixture.test")
    sock.close()


# Denials


@pytest.mark.parametrize(
    ("target", "status", "rule"),
    [
        ("evil.test:443", "403", NOT_ALLOWED),
        ("fixture.test:1", "403", NOT_ALLOWED),  # allowed name, wrong port
        ("127.0.0.1:{port}", "403", IP_LITERAL),
        ("[::1]:{port}", "403", IP_LITERAL),
        ("2130706433:{port}", "400", MALFORMED),
        ("0x7f000001:{port}", "400", MALFORMED),
        ("127.1:{port}", "400", MALFORMED),
        ("user@fixture.test:{port}", "400", MALFORMED),
        ("fixture.test:{port}@evil.test:443", "400", MALFORMED),
    ],
)
def test_disallowed_destinations_are_refused_and_recorded(make_proxy, fixture_server, target, status, rule):
    proxy = make_proxy({f"fixture.test:{fixture_server}"})
    sock, status_line, rest = open_tunnel(proxy, target.format(port=fixture_server))
    body = rest + read_all(sock)
    sock.close()
    proxy.stop()

    assert status_line.split()[1] == status and rule.encode() in body
    assert b"fixture.test" not in body or "fixture.test" in target  # the allowlist is never listed
    record = only(proxy)
    assert (record.decision, record.rule_id) == ("deny", rule)
    if rule == MALFORMED:
        assert record.authority == PLACEHOLDER_AUTHORITY  # never an echo of what the client sent


def test_only_connect_is_offered(make_proxy, fixture_server):
    proxy = make_proxy({f"fixture.test:{fixture_server}"})
    sock = socket.socket(socket.AF_UNIX)
    sock.connect(str(proxy.socket_path))
    sock.sendall(f"GET http://fixture.test:{fixture_server}/ HTTP/1.1\r\nHost: fixture.test\r\n\r\n".encode())
    assert read_all(sock).startswith(b"HTTP/1.1 405")
    proxy.stop()
    assert only(proxy).rule_id == METHOD


def test_an_allowed_name_resolving_to_a_forbidden_class_is_refused(make_proxy):
    proxy = make_proxy({"private.test:80", "meta.test:80"}, classes=("public",))
    for target, address_class in (("private.test:80", "private"), ("meta.test:80", "link_local")):
        sock, status, _ = open_tunnel(proxy, target)
        sock.close()
        assert status.split()[1] == "403"
    proxy.stop()
    assert [(r.rule_id, r.resolved_address_class) for r in proxy.records] == [(ADDRESS_CLASS, "private"), (ADDRESS_CLASS, "link_local")]


def test_a_dns_change_cannot_redirect_a_connection(make_proxy, fixture_server):
    answers = iter([["127.0.0.1"], ["10.0.0.9"]])
    connected = []

    def connector(address, port, timeout):
        connected.append(address)
        return system_connector(address, port, timeout)

    proxy = make_proxy({f"fixture.test:{fixture_server}"}, resolve=lambda host, port: next(answers), connector=connector)
    first, status1, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    get(first, "/")
    second, status2, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    second.close()
    proxy.stop()

    assert status1.split()[1] == "200" and status2.split()[1] == "403"
    assert connected == ["127.0.0.1"]  # connected to the validated address, and only that
    assert [r.rule_id for r in proxy.records] == [ALLOW, ADDRESS_CLASS]


def test_a_redirect_to_another_authority_needs_its_own_connect(make_proxy, fixture_server):
    proxy = make_proxy({f"fixture.test:{fixture_server}"})
    sock, _, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    redirect = get(sock, "/redirect-other")
    assert b"302" in redirect.split(b"\r\n")[0] and f"other.test:{fixture_server}".encode() in redirect
    hop, status, _ = open_tunnel(proxy, f"other.test:{fixture_server}")  # what a client following it does
    hop.close()
    same, _, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    assert b"302" in get(same, "/redirect-same").split(b"\r\n")[0]
    again, status_same, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    assert get(again, "/").endswith(b"hello from the fixture\n")
    proxy.stop()

    assert status.split()[1] == "403" and status_same.split()[1] == "200"
    assert [r.rule_id for r in proxy.records] == [ALLOW, NOT_ALLOWED, ALLOW, ALLOW]


def test_the_client_cannot_override_the_policy(make_proxy, fixture_server):
    proxy = make_proxy({f"fixture.test:{fixture_server}"})
    extra = b"X-Bulkhead-Policy: allow-all\r\nX-Bulkhead-Episode: ep_other\r\nProxy-Authorization: Basic YWRtaW46YWRtaW4=\r\n"
    sock, status, _ = open_tunnel(proxy, "evil.test:443", extra)
    sock.close()
    assert status.split()[1] == "403"


# Ceilings


def test_bytes_in_are_capped_across_the_calls_connections(make_proxy, fixture_server):
    proxy = make_proxy({f"fixture.test:{fixture_server}"}, max_bytes_in=30_000)
    first, _, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    got_first = get(first, "/big")
    second, _, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    got_second = get(second, "/big")
    proxy.stop()

    assert len(got_first) == 30_000 and got_second == b""
    records = proxy.records
    assert [(r.rule_id, r.truncated) for r in records] == [(BYTES_IN, True), (BYTES_IN, True)]
    assert sum(r.bytes_in for r in records) == 30_000


def test_bytes_out_are_capped(make_proxy, fixture_server):
    proxy = make_proxy({f"fixture.test:{fixture_server}"}, max_bytes_out=2_000)
    sock, _, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    body = b"x" * 10_000
    try:
        sock.sendall(f"POST /upload HTTP/1.1\r\nHost: fixture.test\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
    except OSError:
        pass
    read_all(sock)
    proxy.stop()
    record = only(proxy)
    assert (record.rule_id, record.truncated, record.bytes_out) == (BYTES_OUT, True, 2_000)


def test_too_many_connections_are_refused(make_proxy, fixture_server):
    proxy = make_proxy({f"fixture.test:{fixture_server}"}, max_connections=2)
    statuses = []
    for _ in range(3):
        sock, status, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
        statuses.append(status.split()[1])
        sock.close()
    proxy.stop()
    assert statuses == ["200", "200", "403"] and proxy.records[-1].rule_id == CONNECTIONS


def test_request_records_are_bounded(make_proxy):
    proxy = make_proxy(set(), max_requests=3)
    for _ in range(6):
        sock = socket.socket(socket.AF_UNIX)
        sock.connect(str(proxy.socket_path))
        try:
            sock.sendall(b"CONNECT evil.test:443 HTTP/1.1\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # past the record ceiling the proxy may close before reading anything
        read_all(sock)
        sock.close()
    proxy.stop()
    assert len(proxy.records) == 4 and proxy.records[-1].rule_id == CONNECTIONS and proxy.dropped == 2


def test_a_slow_request_head_is_cut_off(make_proxy):
    proxy = make_proxy(set(), idle_s=0.5)
    sock = socket.socket(socket.AF_UNIX)
    sock.connect(str(proxy.socket_path))
    sock.settimeout(5)
    started = time.monotonic()
    sock.sendall(b"CONNECT evil")
    for piece in (b".te", b"st:4", b"43 HT"):
        time.sleep(0.2)
        try:
            sock.sendall(piece)
        except OSError:
            break
    data = read_all(sock)
    proxy.stop()
    assert data == b"" and time.monotonic() - started < 3
    assert only(proxy).rule_id == IDLE


def test_an_oversized_head_is_refused(make_proxy):
    proxy = make_proxy(set())
    sock = socket.socket(socket.AF_UNIX)
    sock.connect(str(proxy.socket_path))
    try:
        sock.sendall(b"CONNECT evil.test:443 HTTP/1.1\r\nX-Pad: " + b"a" * 20_000 + b"\r\n\r\n")
    except OSError:
        pass
    data = read_all(sock)
    proxy.stop()
    assert data.startswith(b"HTTP/1.1 400") and only(proxy).rule_id == HEADERS


def test_an_idle_tunnel_is_closed(make_proxy, fixture_server):
    proxy = make_proxy({f"fixture.test:{fixture_server}"}, idle_s=0.5)
    sock, status, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    assert status.split()[1] == "200"
    assert read_all(sock) == b""  # nothing sent: the proxy closes after idle_s
    proxy.stop()
    assert only(proxy).rule_id == IDLE


def test_abrupt_disconnects_and_upstream_failures_are_recorded(make_proxy, fixture_server):
    proxy = make_proxy({f"fixture.test:{fixture_server}", "other.test:1"})
    sock, _, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    assert b"partial" in get(sock, "/abort")
    refused, status, _ = open_tunnel(proxy, "other.test:1")  # nothing listens there
    refused.close()
    proxy.stop()
    assert status.split()[1] == "502"
    assert [(r.decision, r.rule_id) for r in proxy.records] == [("allow", ALLOW), ("allow", UPSTREAM)]


def test_stopping_closes_live_tunnels_promptly(make_proxy, fixture_server):
    proxy = make_proxy({f"fixture.test:{fixture_server}"}, idle_s=30)
    sock, _, _ = open_tunnel(proxy, f"fixture.test:{fixture_server}")
    started = time.monotonic()
    proxy.stop()
    assert time.monotonic() - started < 3 and only(proxy).rule_id == TIME
    sock.close()


# Socket hygiene


def test_each_calls_socket_is_private_and_gone_afterwards(make_proxy, fixture_server, tmp_path):
    first = make_proxy({f"fixture.test:{fixture_server}"})
    assert stat.S_IMODE(os.stat(first.socket_path).st_mode) == 0o600
    first.stop()
    assert not first.socket_path.exists()
    stale = socket.socket(socket.AF_UNIX)
    with pytest.raises(FileNotFoundError):
        stale.connect(str(first.socket_path))
    second = make_proxy(set())  # another call: its own socket and its own policy
    sock, status, _ = open_tunnel(second, f"fixture.test:{fixture_server}")
    sock.close()
    assert status.split()[1] == "403"


def test_call_proxy_uses_a_short_private_directory_and_removes_it():
    with call_proxy(Egress(allow=("a.example:443",))) as proxy:
        directory = proxy.socket_path.parent
        assert len(str(proxy.socket_path)) < 100 and stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    assert not directory.exists()
