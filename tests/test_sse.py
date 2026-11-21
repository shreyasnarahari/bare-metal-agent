import json

import pytest

from toolsmith.sse import SSEDecodeError, SSEJSONDecoder


def decode_in_chunks(data, chunks):
    decoder = SSEJSONDecoder()
    events = []
    for chunk in chunks:
        events.extend(decoder.feed(chunk))
    events.extend(decoder.finish())
    return events, decoder.done_received


def test_sse_decodes_crlf_comments_multiline_json_and_done_at_every_byte_split():
    payload = (
        b": keepalive\r\n"
        b"data: {\"content\":\"caf\xc3\xa9\",\r\n"
        b"data: \"stop\":false}\r\n\r\n"
        b"data: [DONE]\r\n\r\n"
    )
    expected = [{"content": "café", "stop": False}]

    for position in range(len(payload) + 1):
        events, done = decode_in_chunks(
            payload, [payload[:position], payload[position:]]
        )
        assert events == expected
        assert done

    events, done = decode_in_chunks(payload, [bytes([byte]) for byte in payload])
    assert events == expected
    assert done


def test_sse_dispatches_final_event_without_trailing_blank_line():
    payload = b'data: {"content":"done","stop":true}'

    events, done = decode_in_chunks(payload, [payload])

    assert events == [{"content": "done", "stop": True}]
    assert not done


@pytest.mark.parametrize("payload", [b"\xff", b"\xc3"])
def test_sse_rejects_invalid_or_truncated_utf8(payload):
    with pytest.raises(SSEDecodeError):
        decode_in_chunks(payload, [payload])


@pytest.mark.parametrize(
    "payload",
    [
        b"data: not-json\n\n",
        b"data: [1,2]\n\n",
    ],
)
def test_sse_rejects_invalid_json_events(payload):
    with pytest.raises(SSEDecodeError):
        decode_in_chunks(payload, [payload])
