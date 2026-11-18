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
