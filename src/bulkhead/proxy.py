"""The trusted outside CONNECT proxy: one per call, the sandbox's only route to the network.

The supervisor starts an ``EgressProxy`` bound to the call's canonical
``EgressPolicy`` before the program starts, on a fresh Unix socket (mode 0600,
in a private directory) that only this call's sandbox can reach. The policy
comes from the sandbox spec. Nothing the client sends (headers, proxy
authorization, claimed episode or policy) is read as policy.

For each connection:

1. Read the request head within ``MAX_HEADER_BYTES`` and ``idle_s`` (a
   slowloris client is cut off); parse it strictly (``bulkhead.egress``).
2. Decide the authority; resolve DNS names here, outside the sandbox; pin the
   first address in an allowed class; refuse past ``max_connections``.
3. Connect to the pinned address and answer ``200``, or answer
   ``403``/``400``/``405``/``502`` with a short body naming the rule. The body
   never lists the policy or echoes request data.
4. Tunnel both directions through a selector with small buffers, counting
   every byte against the call's ``max_bytes_out``/``max_bytes_in`` budgets
   (shared by all its connections). A crossed budget, an idle connection, or
   the call's end closes the tunnel.

Every request becomes an ``EgressRecord`` (at most ``max_requests``; past that
one denial is recorded and further connections are dropped). The supervisor
journals them as ``egress_decision`` events in request order. Records hold
the authority and counts only, never URLs, paths, headers, or bodies (inside
TLS the proxy cannot see those anyway).

A proxy that fails to start is an infrastructure failure of the call (it
never runs), not a policy denial.
"""

from __future__ import annotations

import ipaddress
import os
import selectors
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .egress import (
    ALLOW,
    BYTES_IN,
    BYTES_OUT,
    CONNECTIONS,
    HEADERS,
    IDLE,
    MALFORMED,
    MAX_HEADER_BYTES,
    METHOD,
    PLACEHOLDER_AUTHORITY,
    TIME,
    UPSTREAM,
    EgressError,
    EgressPolicy,
    EgressRecord,
    parse_connect,
)
from .manifest import Egress

CHUNK = 16384
STATUS = {403: "Forbidden", 400: "Bad Request", 405: "Method Not Allowed", 502: "Bad Gateway"}

Resolver = Callable[[str, int], list[str]]
Connector = Callable[[str, int, float], socket.socket]


def system_resolver(host: str, port: int) -> list[str]:
    """Addresses for ``host`` in resolver order, without duplicates."""
    seen: list[str] = []
    for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        address = info[4][0]
        if address not in seen:
            seen.append(address)
    return seen


def system_connector(address: str, port: int, timeout_s: float) -> socket.socket:
    family = socket.AF_INET6 if isinstance(ipaddress.ip_address(address), ipaddress.IPv6Address) else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((address, port))
    except BaseException:
        sock.close()
        raise
    return sock


def _response(code: int, rule: str, authority: str) -> bytes:
    if code == 403:
        body = f"bulkhead: CONNECT to {authority} refused ({rule}); only the task's allowed destinations are reachable.\n"
    elif code == 502:
        body = f"bulkhead: could not reach {authority} ({rule}).\n"
    else:
        body = f"bulkhead: request refused ({rule}); send CONNECT host:port HTTP/1.1.\n"
    data = body.encode()
    return f"HTTP/1.1 {code} {STATUS[code]}\r\nContent-Type: text/plain\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data


class EgressProxy:
    """One call's CONNECT proxy (see the module docstring)."""

    def __init__(
        self,
        policy: EgressPolicy,
        socket_path: str | os.PathLike[str],
        *,
        resolver: Resolver = system_resolver,
        connector: Connector = system_connector,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.socket_path = Path(socket_path)
        self.resolver = resolver
        self.connector = connector
        self.clock = clock
        self._lock = threading.Lock()
        self._records: list[tuple[int, EgressRecord]] = []
        self._requests = 0
        self._tunnels = 0
        self._used = {"out": 0, "in": 0}
        self._stopping = threading.Event()
        self._listener: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._sockets: set[socket.socket] = set()
        self.dropped = 0

    # Lifecycle

    def start(self) -> EgressProxy:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(os.fspath(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(16)
            listener.settimeout(0.1)
        except BaseException:
            listener.close()
            raise
        self._listener = listener
        thread = threading.Thread(target=self._accept, name="bulkhead-proxy", daemon=True)
        thread.start()
        self._threads.append(thread)
        return self

    def stop(self, timeout_s: float = 2.0) -> tuple[EgressRecord, ...]:
        """Close everything and return the records in request order."""
        self._stopping.set()
        if self._listener is not None:
            self._listener.close()
        with self._lock:
            sockets = list(self._sockets)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        give_up = time.monotonic() + timeout_s
        for thread in list(self._threads):
            thread.join(max(0.0, give_up - time.monotonic()))
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        return self.records

    @property
    def records(self) -> tuple[EgressRecord, ...]:
        with self._lock:
            return tuple(record for _, record in sorted(self._records, key=lambda item: item[0]))

    def __enter__(self) -> EgressProxy:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # Serving

    def _accept(self) -> None:
        assert self._listener is not None
        while not self._stopping.is_set():
            try:
                client, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with self._lock:
                index = self._requests
                self._requests += 1
            if index > self.policy.max_requests:
                self.dropped += 1
                client.close()
                continue
            thread = threading.Thread(target=self._serve, args=(client, index), daemon=True)
            self._threads.append(thread)
            thread.start()

    def _track(self, sock: socket.socket) -> None:
        with self._lock:
            self._sockets.add(sock)

    def _record(self, index: int, record: EgressRecord) -> None:
        with self._lock:
            self._records.append((index, record))

    def _serve(self, client: socket.socket, index: int) -> None:
        started = self.clock()
        self._track(client)
        record = EgressRecord(f"req_{index}", PLACEHOLDER_AUTHORITY, "deny", MALFORMED, "unresolved")
        self._record(index, record)
        upstream = None
        try:
            if index == self.policy.max_requests:
                record.rule_id = CONNECTIONS
                client.sendall(_response(403, CONNECTIONS, PLACEHOLDER_AUTHORITY))
                return
            try:
                authority, early = parse_connect(self._read_head(client))
            except EgressError as exc:
                record.rule_id = exc.rule
                code = 405 if exc.rule == METHOD else 400
                if exc.rule != IDLE:
                    client.sendall(_response(code, exc.rule, PLACEHOLDER_AUTHORITY))
                return
            record.authority = authority.text
            decision = self.policy.decide(authority)
            if decision.allowed and decision.address is None:
                try:
                    addresses = self.resolver(authority.host, authority.port)
                except OSError:
                    addresses = []
                decision = self.policy.choose(authority, addresses)
            record.resolved_address_class = decision.address_class
            if decision.allowed:
                with self._lock:
                    if self._tunnels >= self.policy.max_connections:
                        decision = type(decision)(False, CONNECTIONS, decision.authority, decision.address_class)
                    else:
                        self._tunnels += 1
            if not decision.allowed:
                record.rule_id = decision.rule_id
                client.sendall(_response(403, decision.rule_id, authority.text))
                return
            record.decision = "allow"
            assert decision.address is not None
            try:
                upstream = self.connector(decision.address, authority.port, self.policy.connect_timeout_s)
            except OSError:
                record.rule_id = UPSTREAM
                client.sendall(_response(502, UPSTREAM, authority.text))
                return
            self._track(upstream)
            record.rule_id = ALLOW
            client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            ended = self._tunnel(client, upstream, early, record)
            if ended is not None:
                record.rule_id = ended
        except OSError:
            pass  # the client or upstream vanished; the record keeps what was decided
        finally:
            record.elapsed_s = self.clock() - started
            for sock in (client, upstream):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    def _read_head(self, client: socket.socket) -> bytes:
        """The request head, within MAX_HEADER_BYTES and idle_s in total (slow headers are cut off)."""
        buffer = bytearray()
        give_up = self.clock() + self.policy.idle_s
        while b"\r\n\r\n" not in buffer:
            if len(buffer) > MAX_HEADER_BYTES:
                raise EgressError("request head too large", rule=HEADERS)
            remaining = give_up - self.clock()
            if remaining <= 0 or self._stopping.is_set():
                raise EgressError("request head too slow", rule=IDLE)
            client.settimeout(min(remaining, 0.2))
            try:
                chunk = client.recv(4096)
            except TimeoutError:
                continue
            if not chunk:
                raise EgressError("the client closed before a full request", rule=MALFORMED)
            buffer += chunk
        return bytes(buffer)

    def _take(self, direction: str, size: int) -> int:
        limit = self.policy.max_bytes_out if direction == "out" else self.policy.max_bytes_in
        with self._lock:
            allowed = max(0, min(size, limit - self._used[direction]))
            self._used[direction] += allowed
        return allowed

    def _tunnel(self, client: socket.socket, upstream: socket.socket, early: bytes, record: EgressRecord) -> str | None:
        """Relay until both sides finish; return the rule that cut it short, if any."""
        for sock in (client, upstream):
            sock.settimeout(self.policy.idle_s)

        def forward(data: bytes, direction: str, destination: socket.socket) -> bool:
            allowed = self._take(direction, len(data))
            if allowed:
                destination.sendall(data[:allowed])
            if direction == "out":
                record.bytes_out += allowed
            else:
                record.bytes_in += allowed
            return allowed == len(data)

        if early and not forward(early, "out", upstream):
            record.truncated = True
            return BYTES_OUT
        selector = selectors.DefaultSelector()
        selector.register(client, selectors.EVENT_READ, ("out", upstream))
        selector.register(upstream, selectors.EVENT_READ, ("in", client))
        last = self.clock()
        # Whether a side ended on its own before the call stopped. stop() shuts every socket down, so
        # an EOF seen while stopping is the cut, not a clean end: such a tunnel is recorded as TIME.
        finished = False
        try:
            while selector.get_map():
                if self._stopping.is_set():
                    return None if finished else TIME
                events = selector.select(timeout=0.2)
                if not events:
                    if self.clock() - last >= self.policy.idle_s:
                        return IDLE
                    continue
                for key, _ in events:
                    direction, destination = key.data
                    try:
                        data = key.fileobj.recv(CHUNK)  # type: ignore[union-attr]
                    except OSError:
                        data = b""
                    if not data:
                        finished = finished or not self._stopping.is_set()
                        selector.unregister(key.fileobj)
                        try:
                            destination.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        continue
                    if not forward(data, direction, destination):
                        record.truncated = True
                        return BYTES_OUT if direction == "out" else BYTES_IN
                    last = self.clock()
        finally:
            selector.close()
        return TIME if self._stopping.is_set() and not finished else None


@contextmanager
def call_proxy(egress: Egress, *, resolver: Resolver = system_resolver, connector: Connector = system_connector, **options: object) -> Iterator[EgressProxy]:
    """A started proxy for one call on a fresh socket in a private directory; stopped and removed afterwards.

    The socket lives in a short ``mkdtemp`` path (Unix socket paths are short)
    with mode 0700 on the directory and 0600 on the socket. The namespace
    backend bind-mounts it into the sandbox at ``bulkhead.bridge.SOCKET_PATH``.
    """
    directory = tempfile.mkdtemp(prefix="bhp-")
    proxy = EgressProxy(EgressPolicy.from_egress(egress, **options), Path(directory) / "proxy.sock", resolver=resolver, connector=connector)  # type: ignore[arg-type]
    try:
        proxy.start()
        yield proxy
    finally:
        proxy.stop()
        shutil.rmtree(directory, ignore_errors=True)


__all__ = ["EgressProxy", "call_proxy", "system_connector", "system_resolver"]
