"""Start, health-check, and stop a ``llama-server`` process.

``LlamaCppServer`` owns one server process: it builds the command line (model,
GPU layers, context size, parallel slots, API key, extra flags), sends output
to a log file, waits for ``/health`` to report the model loaded, and stops the
whole process group on exit. It refuses to start on a port that is already in
use rather than killing whatever holds it.
"""

from __future__ import annotations

import os
import signal
import socket
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, BinaryIO, Sequence

from .llama_cpp import LlamaCppClient, LlamaCppError, LlamaCppTimeoutError


class LlamaCppServerError(RuntimeError):
    """The server could not start, or exited before becoming healthy."""


class LlamaCppServer:
    """One managed llama-server process. ``context_size`` is split across ``parallel`` slots."""

    def __init__(
        self,
        executable: str | Path,
        model: str | Path,
        api_key: str,
        log_path: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        context_size: int = 16384,
        gpu_layers: int = 99,
        parallel: int = 1,
        extra_args: Sequence[str] = (),
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if context_size < 1:
            raise ValueError("context_size must be positive")
        if parallel < 1:
            raise ValueError("parallel must be positive")

        self.executable = Path(executable)
        self.model = Path(model)
        self.api_key = api_key
        self.log_path = Path(log_path)
        self.host = host
        self.port = port
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.parallel = parallel
        self.extra_args = tuple(extra_args)
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file: BinaryIO | None = None

    @property
    def process(self) -> subprocess.Popen[bytes] | None:
        return self._process

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def settings(self) -> dict[str, Any]:
        """The launch settings, for records (the API key is deliberately left out)."""
        return {
            "executable": str(self.executable),
            "model": str(self.model),
            "host": self.host,
            "port": self.port,
            "context_size": self.context_size,
            "gpu_layers": self.gpu_layers,
            "parallel": self.parallel,
            "jinja": True,
            "extra_args": list(self.extra_args),
        }

    def start(self) -> None:
        if self.is_running:
            raise LlamaCppServerError("llama-server is already running")
        if not self.executable.is_file():
            raise FileNotFoundError(f"llama-server executable not found: {self.executable}")
        if not self.model.is_file():
            raise FileNotFoundError(f"GGUF model not found: {self.model}")

        # Check the port first: llama-server's own bind error is buried in its log.
        # SO_REUSEADDR matches llama-server, so a port left in TIME_WAIT after stop()
        # is usable, while a live listener still makes bind() fail.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((self.host, self.port))
            except OSError as exc:
                raise LlamaCppServerError(
                    f"Cannot start llama-server: {self.host}:{self.port} is already in use"
                ) from exc

        self.executable.chmod(self.executable.stat().st_mode | stat.S_IXUSR)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_file is not None:
            self._log_file.close()
        self._log_file = self.log_path.open("ab")

        # -ngl 99 offloads every layer to the GPU; --jinja uses the GGUF's chat template for /props.
        command = [
            str(self.executable),
            "-m",
            str(self.model),
            "--jinja",
            "-ngl",
            str(self.gpu_layers),
            "-c",
            str(self.context_size),
            "--parallel",
            str(self.parallel),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--api-key",
            self.api_key,
            *self.extra_args,
        ]
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self.executable.parent),
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # its own process group, so stop() can signal every child
            )
        except OSError as exc:
            self._log_file.close()
            self._log_file = None
            raise LlamaCppServerError(f"Could not execute llama-server: {exc}") from exc

    def wait_until_healthy(
        self,
        client: LlamaCppClient,
        *,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:
        """Poll ``/health`` until the model is loaded; raise early if the process dies or CUDA is wrong."""
        if self._process is None:
            raise LlamaCppServerError("llama-server has not been started")
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")

        deadline = time.monotonic() + timeout
        last_error: LlamaCppError | None = None
        while True:
            if self._process.poll() is not None:
                raise LlamaCppServerError(
                    f"llama-server exited before becoming healthy. {self.log_tail()}"
                ) from last_error
            try:
                return client.health()
            except LlamaCppError as exc:
                # 503 means "still loading" and None means "not listening yet"; anything else is real.
                if getattr(exc, "status_code", None) not in (None, 503):
                    raise
                last_error = exc
                fatal = self._fatal_cuda_log()
                if fatal:
                    raise LlamaCppServerError(fatal) from exc

            if time.monotonic() >= deadline:
                raise LlamaCppTimeoutError(
                    f"llama-server did not become healthy within {timeout} seconds. "
                    f"{self.log_tail()}"
                ) from last_error
            time.sleep(poll_interval)

    # A binary built for another GPU architecture keeps running but can never serve; fail fast.
    def _fatal_cuda_log(self) -> str | None:
        if not self.log_path.exists():
            return None
        log = self.log_path.read_text(errors="replace")
        if "no kernel image is available for execution on the device" not in log:
            return None
        return (
            "llama-server was compiled for a different GPU and cannot execute "
            "kernels on this device. Re-run the llama.cpp build cell so it "
            f"compiles for this GPU. {self.log_tail()}"
        )

    def log_tail(self, lines: int = 30) -> str:
        """The last lines of the server log, for error messages."""
        if self._log_file is not None:
            self._log_file.flush()
        if not self.log_path.exists():
            return "No server log was created."
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])

    def stop(self, timeout: float = 20.0) -> None:
        """SIGTERM the process group, then SIGKILL if it outlives ``timeout``. Always closes the log."""
        try:
            process = self._process
            if process is not None and process.poll() is None:
                # llama-server can fork after start_new_session; signal the group.
                for action in (signal.SIGTERM, signal.SIGKILL):
                    if process.poll() is not None:
                        break
                    try:
                        os.killpg(process.pid, action)
                    except ProcessLookupError:
                        process.terminate() if action == signal.SIGTERM else process.kill()
                    try:
                        process.wait(timeout=timeout if action == signal.SIGTERM else 5)
                    except subprocess.TimeoutExpired:
                        continue
        finally:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def __enter__(self) -> LlamaCppServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
