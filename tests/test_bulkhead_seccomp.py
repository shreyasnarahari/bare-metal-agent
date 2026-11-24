"""Bulkhead part 4: the seccomp filter, checked by simulation everywhere and on the real kernel where possible.

The simulator tests read the compiled BPF program's decision for any syscall,
argument, or architecture without installing it. The direct tests install the
filter in a fresh child (``bulkhead._seccomp_exec``) with no namespaces, so
they test the filter by itself, comparing each primitive with and without it.
"""

from __future__ import annotations

import errno
import json
import os
import platform
import sys
from pathlib import Path

import pytest

from bulkhead import seccomp
from bulkhead.backends.base import Program
from bulkhead.seccomp import (
    ALLOWED_SYSCALLS,
    AUDIT_ARCH_ARM,
    AUDIT_ARCH_I386,
    BLOCKED_CLASSES,
    CLONE_NAMESPACE_FLAGS,
    DEFAULT_POLICY,
    X32_SYSCALL_BIT,
    SeccompUnavailable,
    SyscallPolicy,
    compile_filter,
    describe,
    simulate,
    syscall_number,
)

from bulkhead.eval.attacks import MARKER, environment_canary, judge, run_script

from exec_support import DATASET, HostSeccompBackend, request_for

LINUX = sys.platform.startswith("linux")
SUPPORTED = platform.machine().lower() in seccomp.ARCHITECTURES


def _library_present() -> bool:
    try:
        seccomp._library()
    except SeccompUnavailable:
        return False
    return True


needs_libseccomp = pytest.mark.skipif(not (LINUX and SUPPORTED and _library_present()), reason="unsupported: needs Linux x86_64/aarch64 with libseccomp")


@pytest.fixture(scope="module")
def compiled():
    if not (LINUX and SUPPORTED and _library_present()):
        pytest.skip("unsupported: needs Linux x86_64/aarch64 with libseccomp")
    return compile_filter()


def decide(compiled, name, *args, arch=None):
    number = syscall_number(name)
    assert number is not None, name
    return describe(simulate(compiled.bpf, number, arch=compiled.audit_arch if arch is None else arch, args=args))


# The policy itself


def test_the_allowlist_and_the_blocked_classes_never_overlap():
    blocked = {name for names in BLOCKED_CLASSES.values() for name in names}
    assert not blocked & set(ALLOWED_SYSCALLS)
    assert not {"clone", "clone3", "socket", "socketpair"} & (blocked | set(ALLOWED_SYSCALLS))
    assert set(DEFAULT_POLICY.deny) == blocked and set(DEFAULT_POLICY.deny.values()) == {errno.EPERM}


def test_every_class_the_plan_names_is_blocked():
    blocked = {name for names in BLOCKED_CLASSES.values() for name in names}
    for name in (
        "ptrace", "process_vm_readv", "process_vm_writev", "io_uring_setup", "io_uring_enter", "io_uring_register",
        "bpf", "perf_event_open", "keyctl", "init_module", "finit_module", "delete_module", "mount", "umount2",
        "pivot_root", "setns", "unshare", "open_by_handle_at", "reboot", "kexec_load", "kexec_file_load", "userfaultfd",
        "setuid", "capset",
    ):
        assert name in blocked, name


def test_a_contradictory_policy_is_refused():
    with pytest.raises(ValueError, match="both allowed and denied"):
        SyscallPolicy(allow=("read", "ptrace"))
    with pytest.raises(ValueError, match="dedicated rules"):
        SyscallPolicy(allow=("read", "socket"))
    with pytest.raises(ValueError, match="AF_UNIX, nothing, or"):
        SyscallPolicy(socket_families=(1, 2))
    with pytest.raises(ValueError, match="AF_UNIX, nothing, or"):
        SyscallPolicy(socket_families=(2,))


def test_an_unsupported_architecture_is_refused_not_guessed(monkeypatch):
    monkeypatch.setattr(seccomp.platform, "machine", lambda: "riscv64")
    with pytest.raises(SeccompUnavailable, match="riscv64"):
        seccomp.native_architecture()
    with pytest.raises(SeccompUnavailable):
        seccomp._compile(DEFAULT_POLICY)


def test_describe_names_kernel_actions():
    assert describe(seccomp.RET_ALLOW) == "allow"
    assert describe(seccomp.RET_ERRNO | errno.EPERM) == "errno:EPERM"
    assert describe(seccomp.RET_KILL_PROCESS) == "kill_process"


# What the compiled filter decides (simulated)


def test_compilation_targets_this_machine_and_is_cached(compiled):
    assert compiled.arch == platform.machine().lower()
    assert compiled.audit_arch == seccomp.ARCHITECTURES[compiled.arch]
    assert compiled.instructions > 50 and len(compiled.bpf) % 8 == 0
    assert compile_filter() is compiled
    assert compiled.policy_sha256 == DEFAULT_POLICY.sha256()
    # Names missing on this architecture are reported, never silently treated as allowed.
    for name in compiled.skipped:
        assert syscall_number(name) is None


def test_the_interpreter_and_shell_syscalls_are_allowed(compiled):
    for name in ("read", "write", "openat", "close", "mmap", "brk", "execve", "wait4", "futex", "getrandom", "rt_sigaction", "exit_group"):
        assert decide(compiled, name) == "allow", name


def test_every_blocked_syscall_returns_eperm(compiled):
    for names in BLOCKED_CLASSES.values():
        for name in names:
            if syscall_number(name) is not None:
                assert decide(compiled, name) == "errno:EPERM", name


def test_unlisted_syscalls_return_enosys_so_libraries_fall_back(compiled):
    assert decide(compiled, "clone3") == "errno:ENOSYS"
    for name in ("seccomp", "landlock_create_ruleset", "io_setup"):
        if syscall_number(name) is not None:
            assert decide(compiled, name) == "errno:ENOSYS", name
    assert describe(simulate(compiled.bpf, 999, arch=compiled.audit_arch)) == "errno:ENOSYS"


def test_clone_is_allowed_for_threads_and_processes_but_never_new_namespaces(compiled):
    thread_flags = 0x003D0F00  # CLONE_VM|FS|FILES|SIGHAND|THREAD|SYSVSEM|SETTLS|PARENT_SETTID|CHILD_CLEARTID
    assert decide(compiled, "clone", thread_flags) == "allow"
    assert decide(compiled, "clone", 0x01200011) == "allow"  # fork: CHILD_SETTID|CHILD_CLEARTID|SIGCHLD
    for name, flag in CLONE_NAMESPACE_FLAGS.items():
        assert decide(compiled, "clone", flag | 17) == "errno:EPERM", name


def test_only_unix_sockets_may_be_created(compiled):
    assert decide(compiled, "socket", 1, 1, 0) == "allow"
    assert decide(compiled, "socketpair", 1, 1, 0) == "allow"
    for family in (2, 10, 16, 17, 38, 40):  # INET, INET6, NETLINK, PACKET, ALG, VSOCK
        assert decide(compiled, "socket", family, 1, 0) == "errno:EACCES", family
        assert decide(compiled, "socketpair", family, 1, 0) == "errno:EACCES", family


def test_a_foreign_architecture_or_abi_kills_the_process(compiled):
    read = syscall_number("read")
    foreign = AUDIT_ARCH_I386 if compiled.arch == "x86_64" else AUDIT_ARCH_ARM
    assert describe(simulate(compiled.bpf, read, arch=foreign)) == "kill_process"
    if compiled.arch == "x86_64":
        assert describe(simulate(compiled.bpf, read | X32_SYSCALL_BIT, arch=compiled.audit_arch)) == "kill_process"


def test_the_memfd_holds_exactly_the_program_at_offset_zero(compiled):
    fd = compiled.memfd()
    try:
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
        assert os.read(fd, len(compiled.bpf) + 1) == compiled.bpf
        assert not os.get_inheritable(fd)  # passed to a child only through pass_fds
    finally:
        os.close(fd)


def test_the_attack_prelude_uses_this_machines_syscall_numbers(compiled):
    text = (DATASET.scripts_dir / "_prelude.py").read_text()
    table = text[text.index("NR = {"): text.index("}\n", text.index("NR = {")) + 1]
    numbers = eval(table[len("NR = "):])  # a literal dict in a fixture file we control
    column = 0 if compiled.arch == "x86_64" else 1
    for name, pair in numbers.items():
        assert syscall_number(name) == pair[column], name


# On the real kernel: the filter installed in a fresh child, with no namespaces


BASELINE = r"""
import ctypes, errno, json, os, socket
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
def raw(number, *args):
    ctypes.set_errno(0)
    rc = libc.syscall(ctypes.c_long(number), *[ctypes.c_long(a) for a in args])
    return "ok" if rc >= 0 else errno.errorcode[ctypes.get_errno()]
def attempt(function):
    try:
        function()
        return "ok"
    except OSError as exc:
        return errno.errorcode[exc.errno]
buffer = ctypes.create_string_buffer(8)
iov = (ctypes.c_void_p * 2)(ctypes.cast(buffer, ctypes.c_void_p), 8)
print(json.dumps({
    "ptrace_traceme": raw(NUMBERS["ptrace"], 0, 0, 0, 0),
    "process_vm_readv_self": raw(NUMBERS["process_vm_readv"], os.getpid(), ctypes.addressof(iov), 1, ctypes.addressof(iov), 1, 0),
    "unshare_user": attempt(lambda: os.unshare(0x10000000)),
    "inet_socket": attempt(lambda: socket.socket(socket.AF_INET).close()),
    "unix_socket": attempt(lambda: socket.socket(socket.AF_UNIX).close()),
}))
"""


def run(tmp_path, text, *, filtered=True, kind="python"):
    backend = HostSeccompBackend(tmp_path / ("filtered" if filtered else "plain"), filtered=filtered)
    outcome = backend.execute(request_for(Program.from_text(kind, text), timeout_s=20), {})
    return outcome


@needs_libseccomp
def test_python_and_the_shell_run_normally_under_the_filter(tmp_path):
    program = (
        "import json, subprocess, threading\n"
        "box = []\n"
        "thread = threading.Thread(target=box.append, args=(6 * 7,)); thread.start(); thread.join()\n"
        "shell = subprocess.run(['/bin/sh', '-c', 'echo shell; ls / > /dev/null && echo listed'], capture_output=True, text=True)\n"
        "status = dict(line.split(':', 1) for line in open('/proc/self/status') if ':' in line)\n"
        "print(json.dumps({'thread': box, 'shell': shell.stdout.split(), 'seccomp': status['Seccomp'].strip(), 'nnp': status['NoNewPrivs'].strip()}))\n"
    )
    outcome = run(tmp_path, program)

    assert outcome.audit["started"] and outcome.audit["exit_code"] == 0, outcome.process
    assert json.loads(outcome.process.stdout) == {"thread": [42], "shell": ["shell", "listed"], "seccomp": "2", "nnp": "1"}
    shell = run(tmp_path, "for word in a b; do echo $word; done; exit 4", kind="shell")
    assert (shell.process.stdout, shell.audit["exit_code"]) == ("a\nb\n", 4)


@needs_libseccomp
def test_each_primitive_works_without_the_filter_and_is_refused_with_it(tmp_path):
    numbers = {name: syscall_number(name) for name in ("ptrace", "process_vm_readv")}
    program = f"NUMBERS = {numbers!r}\n" + BASELINE
    plain = json.loads(run(tmp_path, program, filtered=False).process.stdout)
    filtered = json.loads(run(tmp_path, program).process.stdout)

    assert filtered == {
        "ptrace_traceme": "EPERM",
        "process_vm_readv_self": "EPERM",
        "unshare_user": "EPERM",
        "inet_socket": "EACCES",
        "unix_socket": "ok",
    }
    # Where this kernel allows the primitive outright, the difference is the filter's doing.
    assert plain["ptrace_traceme"] == "ok" and plain["process_vm_readv_self"] == "ok" and plain["inet_socket"] == "ok"
    assert plain["unshare_user"] in ("ok", "EPERM", "ENOSPC")  # userns may be disabled on a host


@needs_libseccomp
def test_a_foreign_abi_syscall_is_killed_and_recorded_as_a_seccomp_violation(tmp_path):
    if platform.machine().lower() != "x86_64":
        pytest.skip("unsupported: the x32 ABI exists only on x86_64")
    program = "import ctypes; print('before', flush=True); ctypes.CDLL(None).syscall(0x40000000 | 39); print('after')"
    outcome = run(tmp_path, program)

    assert outcome.process.stdout == "before\n"
    assert outcome.audit["signal"] == 31 and outcome.audit["exit_code"] is None
    assert outcome.audit["violations"] == [{"kind": "seccomp", "rule_id": "bulkhead.seccomp.kill", "detail": "the program was killed by its seccomp filter (SIGSYS)"}]


DIRECT = [attack for attack in DATASET.attacks if attack.seccomp_direct]


def test_the_seccomp_only_attacks_are_the_syscall_classes():
    assert {attack.attack_id for attack in DIRECT} >= {
        "proc.ptrace", "proc.process_vm", "syscall.io_uring_setup", "syscall.io_uring_socket", "syscall.bpf", "syscall.kernel_primitives",
        "syscall.foreign_abi", "ns.mounts", "ns.nested_namespaces", "priv.escalation", "out.env_secret", "out.inherited_descriptors",
    }


@needs_libseccomp
@pytest.mark.parametrize("attack", DIRECT, ids=[attack.attack_id for attack in DIRECT])
def test_seccomp_class_attacks_fail_under_the_filter_alone(tmp_path, attack):
    """The syscall half of the attack gate, on this kernel, with the filter alone (no namespaces)."""
    if attack.architectures is not None and platform.machine().lower() not in attack.architectures:
        pytest.skip(f"unsupported: {attack.attack_id} needs {attack.architectures}")
    token = MARKER + "0" * 32
    with environment_canary(token):  # the harness's environment holds a secret the program must never see
        evidence = run_script(DATASET, attack, HostSeccompBackend(tmp_path / "work"), tmp_path, profile="host-seccomp-test", extra_secrets=(token,))
    verdict = judge(attack, attack.expectation("bwrap-t2"), evidence)

    assert verdict.verdict == "blocked", verdict.reasons


def test_the_direct_harness_refuses_a_malformed_filter(tmp_path):
    import subprocess

    from exec_support import HARNESS

    read, write = os.pipe()
    os.write(write, b"\x00" * 7)
    os.close(write)
    done = subprocess.run([sys.executable, "-I", "-S", str(HARNESS), str(read), "--", "/bin/true"], pass_fds=(read,), capture_output=True, timeout=20)
    os.close(read)
    assert done.returncode == 126 and b"malformed" in done.stderr
    assert Path(HARNESS).read_text().count("NOT a sandbox") == 1
