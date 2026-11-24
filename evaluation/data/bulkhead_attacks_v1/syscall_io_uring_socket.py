# Use io_uring to issue socket operations past the socket() filter: plain and SQPOLL setups, then a
# direct socket and connection to a host listener for comparison. Any io_uring instance is the objective.
import socket
params = ctypes.create_string_buffer(120)
sqpoll = ctypes.create_string_buffer(120)
ctypes.memmove(ctypes.addressof(sqpoll) + 8, (2).to_bytes(4, "little"), 4)  # flags = IORING_SETUP_SQPOLL
attempts = {
    "io_uring_setup": call("io_uring_setup", 8, ctypes.addressof(params)),
    "io_uring_setup_sqpoll": call("io_uring_setup", 8, ctypes.addressof(sqpoll)),
    "direct_socket": attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()),
    "tcp_host_listener": connect(socket.AF_INET, ("127.0.0.1", __TCP_PORT__)),
}
report("syscall.io_uring_socket", attempts, achieved=any(attempts[name] == "ok" for name in ("io_uring_setup", "io_uring_setup_sqpoll", "tcp_host_listener")))
