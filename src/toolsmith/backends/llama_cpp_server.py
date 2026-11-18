from __future__ import annotations

import socket
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, BinaryIO, Sequence

from .llama_cpp import LlamaCppClient, LlamaCppError, LlamaCppTimeoutError


class LlamaCppServerError(RuntimeError):
    pass


class LlamaCppServer:
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
        extra_args: Sequence[str] = (),
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if context_size < 1:
            raise ValueError("context_size must be positive")

        self.executable = Path(executable)
        self.model = Path(model)
        self.api_key = api_key
        self.log_path = Path(log_path)
        self.host = host
        self.port = port
        self.context_size = context_size
        self.gpu_layers = gpu_layers
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
        return {
            "executable": str(self.executable),
            "model": str(self.model),
            "host": self.host,
            "port": self.port,
            "context_size": self.context_size,
            "gpu_layers": self.gpu_layers,
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

        with socket.socket() as probe:
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

        command = [
            str(self.executable),
            "-m",
            str(self.model),
            "--jinja",
            "-ngl",
            str(self.gpu_layers),
            "-c",
            str(self.context_size),
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
                start_new_session=True,
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
                if getattr(exc, "status_code", None) not in (None, 503):
                    raise
                last_error = exc

            if time.monotonic() >= deadline:
                raise LlamaCppTimeoutError(
                    f"llama-server did not become healthy within {timeout} seconds. "
                    f"{self.log_tail()}"
                ) from last_error
            time.sleep(poll_interval)

    def log_tail(self, lines: int = 30) -> str:
        if self._log_file is not None:
            self._log_file.flush()
        if not self.log_path.exists():
            return "No server log was created."
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])

    def stop(self, timeout: float = 20.0) -> None:
        try:
            if self.is_running and self._process is not None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
        finally:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def __enter__(self) -> LlamaCppServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
