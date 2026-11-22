from pathlib import Path

import pytest


class SyncJobs:
    """``BackgroundJobs`` stand-in for notebook exec tests: each job runs at submit time."""

    log_dir = Path("/tmp/sync-jobs")

    def __init__(self):
        self.submitted = []
        self.results = {}
        self.errors = {}

    def submit(self, name, fn, *args, after=(), **kwargs):
        missing = [dependency for dependency in after if dependency not in self.submitted]
        assert not missing, f"{name} waits for jobs never submitted: {missing}"
        self.submitted.append(name)
        try:
            self.results[name] = fn(*args, **kwargs)
        except Exception as error:  # surfaced by wait(), as the real class does
            self.errors[name] = error

    def wait(self, name, **options):
        if name in self.errors:
            raise RuntimeError(f"Background job {name!r} failed: {self.errors[name]}") from self.errors[name]
        return self.results[name]

    def active(self):
        return []

    def status(self):
        return {name: "failed" if name in self.errors else "done" for name in self.submitted}

    def drain(self, timeout, match=None):
        return {}

    def cancel_all(self):
        return []


@pytest.fixture
def sync_jobs():
    return SyncJobs()


@pytest.fixture
def background_logs(tmp_path, monkeypatch):
    path = tmp_path / "background-logs"
    monkeypatch.setenv("TOOLSMITH_BACKGROUND_LOGS", str(path))
    return path
