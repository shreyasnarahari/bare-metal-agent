import pytest

from toolsmith.artifacts import copy_file_atomic, is_gguf


def test_gguf_validation_requires_magic_and_exact_size(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUFpayload")

    assert is_gguf(model)
    assert is_gguf(model, expected_size=11)
    assert not is_gguf(model, expected_size=12)
    assert not is_gguf(tmp_path / "missing.gguf")

    model.write_bytes(b"NOPEpayload")
    assert not is_gguf(model)


def test_atomic_copy_replaces_destination_and_reports_progress(tmp_path):
    source = tmp_path / "source.gguf"
    destination = tmp_path / "nested" / "destination.gguf"
    source.write_bytes(b"GGUFpayload")
    destination.parent.mkdir()
    destination.write_bytes(b"old")
    updates = []

    result = copy_file_atomic(
        source,
        destination,
        expected_size=11,
        chunk_size=3,
        progress=lambda copied, total: updates.append((copied, total)),
    )

    assert result == destination
    assert destination.read_bytes() == b"GGUFpayload"
    assert updates == [(3, 11), (6, 11), (9, 11), (11, 11)]
    assert list(destination.parent.glob("*.tmp")) == []


def test_atomic_copy_cleans_up_when_source_size_changes(tmp_path):
    source = tmp_path / "source.gguf"
    destination = tmp_path / "destination.gguf"
    source.write_bytes(b"GGUFpayload")

    def interrupt_copy(copied, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        copy_file_atomic(source, destination, progress=interrupt_copy)

    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_copy_rejects_wrong_source_size(tmp_path):
    source = tmp_path / "source.gguf"
    source.write_bytes(b"GGUFpayload")

    with pytest.raises(ValueError, match="expected 100"):
        copy_file_atomic(source, tmp_path / "destination.gguf", expected_size=100)
