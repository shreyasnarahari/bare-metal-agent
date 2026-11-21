import json
import socket
import sys
import textwrap

import pytest

from toolsmith.backends import (
    LlamaCppClient,
    LlamaCppResponseError,
    LlamaCppServer,
    LlamaCppServerError,
)


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_model(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"test model")
    return model


def write_fake_server(path):
    path.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json
            import sys
            from http.server import BaseHTTPRequestHandler, HTTPServer

            args = sys.argv[1:]
            port = int(args[args.index('--port') + 1])
            api_key = args[args.index('--api-key') + 1]

            class Handler(BaseHTTPRequestHandler):
                def reply(self, status, payload):
                    body = json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def authorized(self):
                    return self.headers.get('Authorization') == f'Bearer {{api_key}}'

                def do_GET(self):
                    if not self.authorized():
                        self.reply(401, {{'error': 'unauthorized'}})
                    elif self.path == '/health':
                        self.reply(200, {{'status': 'ok'}})
                    else:
                        self.reply(404, {{'error': 'not found'}})

                def do_POST(self):
                    if not self.authorized():
                        self.reply(401, {{'error': 'unauthorized'}})
                        return
                    length = int(self.headers.get('Content-Length', '0'))
                    payload = json.loads(self.rfile.read(length))
                    if payload.get('stream'):
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/event-stream')
                        self.end_headers()
                        events = [
                            {{'content': 'Echo: ', 'stop': False}},
                            {{'content': payload['prompt'], 'stop': False}},
                            {{'content': '', 'stop': True, 'truncated': False, 'timings': {{'predicted_n': 2}}}},
                        ]
                        for event in events:
                            self.wfile.write(f"data: {{json.dumps(event)}}\\n\\n".encode())
                            self.wfile.flush()
                        return
                    self.reply(200, {{
                        'content': f"Echo: {{payload['prompt']}}",
                        'seed': payload.get('seed'),
                        'stopped_limit': False,
                    }})

                def log_message(self, *args):
                    pass

            HTTPServer(('127.0.0.1', port), Handler).serve_forever()
            """
        )
    )
    path.chmod(0o644)


def test_server_client_completion_and_shutdown_end_to_end(tmp_path):
    executable = tmp_path / "fake-llama-server"
    write_fake_server(executable)
    server = LlamaCppServer(
        executable,
        make_model(tmp_path),
        "test-key",
        tmp_path / "logs" / "server.log",
        port=unused_port(),
    )

    try:
        server.start()
        assert server.is_running
        assert server.settings()["jinja"] is True
        assert "test-key" not in json.dumps(server.settings())

        with LlamaCppClient(server.base_url, "test-key", timeout=2.0) as client:
            health = server.wait_until_healthy(client, timeout=5.0, poll_interval=0.01)
            response = client.complete("Blue", seed=7)

        assert health == {"status": "ok"}
        assert response == {"content": "Echo: Blue", "seed": 7, "stopped_limit": False}
    finally:
        server.stop()

    assert not server.is_running
    assert server.log_path.is_file()
    server.stop()


def test_server_streaming_completion_end_to_end(tmp_path):
    executable = tmp_path / "fake-llama-server"
    write_fake_server(executable)
    server = LlamaCppServer(
        executable,
        make_model(tmp_path),
        "test-key",
        tmp_path / "server.log",
        port=unused_port(),
    )

    try:
        server.start()
        with LlamaCppClient(server.base_url, "test-key", timeout=2.0) as client:
            server.wait_until_healthy(client, timeout=5.0, poll_interval=0.01)
            events = list(client.stream_complete("Blue", seed=7))

        assert "".join(event["content"] for event in events) == "Echo: Blue"
        assert events[-1]["stop"] is True
        assert events[-1]["timings"]["predicted_n"] == 2
    finally:
        server.stop()


def test_health_does_not_retry_authentication_errors(tmp_path):
    executable = tmp_path / "fake-llama-server"
    write_fake_server(executable)
    server = LlamaCppServer(
        executable,
        make_model(tmp_path),
        "correct-key",
        tmp_path / "server.log",
        port=unused_port(),
    )

    try:
        server.start()
        with LlamaCppClient(server.base_url, "wrong-key", timeout=2.0) as client:
            with pytest.raises(LlamaCppResponseError) as caught:
                server.wait_until_healthy(client, timeout=5.0, poll_interval=0.01)
        assert caught.value.status_code == 401
    finally:
        server.stop()


def test_server_rejects_a_port_that_is_already_in_use(tmp_path):
    executable = tmp_path / "fake-llama-server"
    write_fake_server(executable)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = LlamaCppServer(
            executable,
            make_model(tmp_path),
            "test-key",
            tmp_path / "server.log",
            port=port,
        )

        with pytest.raises(LlamaCppServerError, match="already in use"):
            server.start()

    assert server.process is None


def test_server_reports_early_process_exit_with_log_output(tmp_path):
    executable = tmp_path / "broken-llama-server"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "print('synthetic startup failure', flush=True)\n"
        "sys.exit(2)\n"
    )
    server = LlamaCppServer(
        executable,
        make_model(tmp_path),
        "test-key",
        tmp_path / "server.log",
        port=unused_port(),
    )

    try:
        server.start()
        with LlamaCppClient(server.base_url, "test-key", timeout=0.1) as client:
            with pytest.raises(LlamaCppServerError, match="synthetic startup failure"):
                server.wait_until_healthy(client, timeout=2.0, poll_interval=0.01)
    finally:
        server.stop()
