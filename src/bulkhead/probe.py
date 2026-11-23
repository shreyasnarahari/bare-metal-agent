"""``python -m bulkhead.probe``: measure which sandbox tier this host can actually enforce.

The probe runs real operations in a temporary directory (see
``bulkhead.capabilities``), selects the highest tier whose every requirement
was proven, and prints a human summary. ``--json PATH`` also writes the strict,
versioned report atomically. ``--require-tier T`` exits with status 3 unless
every requirement of ``T`` holds. A requested tier is confirmed or refused,
never replaced by a lower one: running at another tier needs a new
configuration and manifest.

Selection is evidence, not permission. No Bulkhead backend executes a tool in
part 1, and a T0 result is always reported as ``NOT a sandbox``.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import uuid
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .capabilities import (
    CGROUP_PARENT_ENV,
    DEFAULT_TIMEOUT_S,
    HELPER,
    ChildRun,
    ProbeContext,
    collect_capabilities,
    display_path,
    host_info,
    outside_listeners,
    run_bounded,
)
from .errors import UnsupportedTierError
from .types import CAPABILITY_NAMES, Capability, CapabilityStatus, ProbeReport, SandboxTier

EXIT_TIER_UNSUPPORTED = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_bwrap(explicit: str | os.PathLike[str] | None) -> str | None:
    """An explicitly given binary, else ``bwrap`` on PATH. Nothing is installed or searched for elsewhere."""
    if explicit:
        return str(Path(explicit).expanduser())
    return shutil.which("bwrap")


def run_probe(
    *,
    bwrap: str | os.PathLike[str] | None = None,
    cgroup_parent: str | os.PathLike[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    runner: Callable[..., ChildRun] = run_bounded,
) -> ProbeReport:
    """Probe this host and return the report. Changes no system setting and leaves nothing behind."""
    created_at = utc_now()
    host = host_info()
    bwrap_path = resolve_bwrap(bwrap)
    parent = Path(cgroup_parent).expanduser() if cgroup_parent else None
    cgroup_mount: Path | None = None
    if not sys.platform.startswith("linux"):
        capabilities = [
            Capability(name, CapabilityStatus.UNSUPPORTED, "not_linux", f"{sys.platform} lacks the Linux features Bulkhead uses")
            for name in CAPABILITY_NAMES
        ]
    else:
        workdir = Path(tempfile.mkdtemp(prefix="bulkhead-probe-"))
        try:
            with outside_listeners(workdir) as (port, unix_socket):
                context = ProbeContext(
                    workdir=workdir,
                    timeout_s=timeout_s,
                    bwrap=bwrap_path,
                    cgroup_parent=parent,
                    tcp_port=port,
                    unix_socket=unix_socket,
                    runner=runner,
                )
                capabilities = collect_capabilities(context)
                cgroup_mount = context.cgroup_mount
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            if workdir.exists():
                warnings.warn(f"could not fully remove the probe directory {workdir}", RuntimeWarning, stacklevel=2)
    return ProbeReport.build(
        created_at=created_at,
        host=host,
        python=display_path(sys.executable),
        helper=display_path(HELPER),
        bwrap_path=display_path(bwrap_path),
        capabilities=capabilities,
        cgroup_mount=str(cgroup_mount) if cgroup_mount is not None else None,
        cgroup_parent=display_path(parent),
    )


# Output


def _selected_line(report: ProbeReport) -> str:
    tier = report.selected_tier
    if tier is SandboxTier.T0:
        return (
            "Selected tier: T0 ... NOT a sandbox. Exec and network tools are denied; "
            "only trusted in-memory tools and diagnostics may run."
        )
    if tier is SandboxTier.T1:
        missing = ", ".join(item.capability for item in report.evaluation(SandboxTier.T2).missing)
        return f"Selected tier: T1 ... Landlock sandbox. Missing T2 namespace/cgroup properties: {missing}."
    namespaces = ", ".join(report.capability("bwrap.namespaces").facts.get("namespaces") or ())
    controllers = ", ".join(report.capability("cgroup.limits").facts.get("controllers") or ())
    return (
        f"Selected tier: T2 ... Namespace sandbox: fresh {namespaces} namespaces; seccomp; no_new_privs; "
        f"cgroup v2 ({controllers}) with cgroup.kill."
    )


def render_report(report: ProbeReport) -> str:
    """The human summary. It states the selected tier with its required wording, T0 included."""
    host, tools, landlock, cgroup = report.host, report.tools, report.landlock, report.cgroup
    traits = [trait for trait, present in (("WSL", host.wsl), ("container", host.container), ("root", host.root)) if present]
    if tools.bwrap_version:
        bwrap = f"bubblewrap {tools.bwrap_version} ({tools.bwrap_path})"
    elif tools.bwrap_path:
        bwrap = f"bubblewrap at {tools.bwrap_path} is not usable"
    else:
        bwrap = "bubblewrap not found on PATH"
    if cgroup.parent is None:
        cgroup_line = f"cgroup v2: {'mounted at ' + cgroup.mount if cgroup.mount else 'not mounted'}; no delegated parent configured"
    else:
        proven = [item for item, done in (("leaf", cgroup.leaf_created), ("limits", cgroup.limits_written), ("cgroup.kill", cgroup.kill_verified)) if done]
        cgroup_line = (
            f"cgroup v2: parent {cgroup.parent} {'delegated' if cgroup.delegated else 'NOT usable'}; "
            f"controllers {' '.join(cgroup.controllers) or 'none'}; proven {', '.join(proven) or 'nothing'}"
        )
    lines = [
        f"Bulkhead capability probe (schema {report.schema_version}, {report.created_at})",
        f"Host: {host.os} {host.kernel_release} {host.architecture}" + (f" ({', '.join(traits)})" if traits else ""),
        f"Tools: {bwrap}; libseccomp {tools.libseccomp_version or ('available' if tools.libseccomp_available else 'not available')}",
        f"Landlock: ABI {landlock.abi}; tested {', '.join(landlock.tested) or 'nothing'}"
        if landlock.abi is not None
        else "Landlock: not available",
        cgroup_line,
        "",
        "Capabilities:",
    ]
    width = max(len(name) for name in CAPABILITY_NAMES)
    for record in report.capabilities:
        lines.append(f"  {record.status.value:<11} {record.name:<{width}}  {record.detail}")
    lines += ["", "Tiers:"]
    for evaluation in report.tiers:
        tier = evaluation.tier
        if evaluation.satisfied:
            state = "no requirements" if tier is SandboxTier.T0 else "every requirement proven"
        else:
            state = f"rejected: {evaluation.summary()}"
        lines.append(f"  {tier.value}  {tier.label:<17}  {state}")
    lines += [
        "",
        _selected_line(report),
        "Selection is evidence, not permission to execute: a caller must require a tier (--require-tier).",
        f"probe_sha256: {report.probe_sha256}",
    ]
    return "\n".join(lines)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Write to a temporary sibling, then rename it into place: readers never see half a file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_report(report: ProbeReport, path: str | os.PathLike[str], *, secrets: Sequence[str] = ()) -> Path:
    """Write the strict JSON report atomically; refuse if any given secret value appears in it."""
    text = report.to_json()
    if any(secret and secret in text for secret in secrets):
        raise ValueError("refusing to write a probe report that contains a secret value")
    return atomic_write_text(path, text)


def load_report(path: str | os.PathLike[str]) -> ProbeReport:
    """Read and strictly validate a report written by ``write_report``."""
    return ProbeReport.from_json(Path(path).read_text(encoding="utf-8"))


def _positive_seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bulkhead.probe",
        description="Probe which Bulkhead sandbox tier this host can enforce. Changes no system setting.",
    )
    parser.add_argument("--json", type=Path, metavar="PATH", help="also write the strict JSON report here (atomically)")
    parser.add_argument(
        "--require-tier",
        choices=[tier.value for tier in SandboxTier],
        help="exit 3 unless every requirement of this tier holds; there is never a fallback to a lower tier",
    )
    parser.add_argument("--bwrap", metavar="PATH", help="bubblewrap binary to probe (default: bwrap on PATH)")
    parser.add_argument(
        "--cgroup-parent",
        metavar="PATH",
        default=os.environ.get(CGROUP_PARENT_ENV) or None,
        help=f"delegated cgroup v2 parent for the leaf test (default: ${CGROUP_PARENT_ENV}; never chosen automatically)",
    )
    parser.add_argument("--timeout", type=_positive_seconds, default=DEFAULT_TIMEOUT_S, help="seconds allowed per child operation")
    args = parser.parse_args(argv)

    report = run_probe(bwrap=args.bwrap, cgroup_parent=args.cgroup_parent, timeout_s=args.timeout)
    print(render_report(report))
    if args.json is not None:
        print(f"Report written to {write_report(report, args.json)}")
    if args.require_tier:
        required = SandboxTier(args.require_tier)
        try:
            report.require(required)
        except UnsupportedTierError as exc:
            print(f"Required tier not met: {exc}", file=sys.stderr)
            return EXIT_TIER_UNSUPPORTED
        print(f"Required tier {required.value} ({required.label}) is supported.")
    return 0


__all__ = [
    "EXIT_TIER_UNSUPPORTED",
    "atomic_write_text",
    "load_report",
    "main",
    "render_report",
    "resolve_bwrap",
    "run_probe",
    "utc_now",
    "write_report",
]


if __name__ == "__main__":
    sys.exit(main())
