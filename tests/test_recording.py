import json


from toolsmith.recording import JsonlRecorder


def test_recorder_appends_redacted_jsonl_and_creates_parent(tmp_path):
    path = tmp_path / "records" / "smoke.jsonl"
    recorder = JsonlRecorder(path, secrets=["hf-secret", "local-secret"])

    saved = recorder.append(
        {
            "request": {
                "headers": {"Authorization": "Bearer local-secret"},
                "prompt": "Use hf-secret only as test data",
            },
            "response": {"content": "café"},
            "api-key": "local-secret",
        }
    )

    assert saved["request"]["headers"]["Authorization"] == "[redacted]"
    assert saved["request"]["prompt"] == "Use [redacted] only as test data"
    assert saved["api-key"] == "[redacted]"
    assert "recorded_at" in saved

    raw = path.read_text()
    assert "hf-secret" not in raw
    assert "local-secret" not in raw
    assert len(raw.splitlines()) == 1
    assert json.loads(raw)["response"]["content"] == "café"


def test_recorder_appends_independent_records_and_preserves_timestamp(tmp_path):
    path = tmp_path / "smoke.jsonl"
    recorder = JsonlRecorder(path)

    recorder.append({"event": "first", "recorded_at": "2026-01-01T00:00:00+00:00"})
    recorder.append({"event": "second"})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["first", "second"]
    assert records[0]["recorded_at"] == "2026-01-01T00:00:00+00:00"
    assert "recorded_at" in records[1]


def test_snapshot_redacts_like_append_without_writing(tmp_path):
    path = tmp_path / "unused.jsonl"
    recorder = JsonlRecorder(path, secrets=["local-secret"])

    snapshot = recorder.snapshot({"note": "uses local-secret", "Authorization": "Bearer x"})

    assert snapshot["note"] == "uses [redacted]"
    assert snapshot["Authorization"] == "[redacted]"
    assert "recorded_at" in snapshot
    assert not path.exists()


def test_context_is_added_to_every_record_without_overriding_its_fields(tmp_path):
    from toolsmith.recording import JsonlRecorder

    calls = []

    def stamp():
        calls.append(1)
        return {"runtime": {"gpu_family": "T4", "note": "secret-value"}, "event": "from-context"}

    recorder = JsonlRecorder(tmp_path / "records.jsonl", secrets=("secret-value",), context=stamp)
    first = recorder.append({"event": "timing", "seconds": 1.5})
    second = recorder.append({"event": "timing", "runtime": "own"})

    assert first["runtime"]["gpu_family"] == "T4" and first["event"] == "timing"
    assert first["runtime"]["note"] == "[redacted]"  # context is redacted like the record
    assert second["runtime"] == "own"  # the record's own field wins
    assert len(calls) == 2  # a callable context is evaluated per record

    fixed = JsonlRecorder(tmp_path / "fixed.jsonl", context={"gpu": "L4"})
    assert fixed.append({"event": "x"})["gpu"] == "L4"
