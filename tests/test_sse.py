from llmserve.client.sse import SseEvent, SseParser

STREAM = (
    b": keep-alive\n"
    b"\n"
    b'data: {"choices":[{"text":" a"}]}\n'
    b"\n"
    b'data: {"choices":[{"text":" b"}]}\n'
    b"\n"
    b"data: [DONE]\n"
    b"\n"
)


def parse(*chunks: bytes) -> list[SseEvent]:
    p = SseParser()
    events: list[SseEvent] = []
    for c in chunks:
        events.extend(p.feed(c))
    events.extend(p.close())
    return events


def test_whole_stream_gives_expected_events() -> None:
    events = parse(STREAM)
    assert [e.data for e in events] == [
        '{"choices":[{"text":" a"}]}',
        '{"choices":[{"text":" b"}]}',
        "[DONE]",
    ]


def test_split_at_every_byte_offset_yields_identical_events() -> None:
    expected = parse(STREAM)
    for split in range(len(STREAM) + 1):
        assert parse(STREAM[:split], STREAM[split:]) == expected, f"split at {split}"


def test_byte_by_byte_feed() -> None:
    assert parse(*[bytes([b]) for b in STREAM]) == parse(STREAM)


def test_multiline_data_is_joined_with_newlines() -> None:
    events = parse(b"data: first\ndata: second\n\n")
    assert events[0].data == "first\nsecond"


def test_crlf_line_endings_and_event_fields() -> None:
    events = parse(b"event: chunk\r\nid: 7\r\nretry: 100\r\ndata: x\r\n\r\n")
    assert events == [SseEvent(data="x", event="chunk", id="7", retry_ms=100)]


def test_comment_only_lines_produce_no_events() -> None:
    assert parse(b": ping\n\n: pong\n") == []


def test_unknown_fields_are_ignored() -> None:
    assert parse(b"foo: bar\ndata: x\n\n")[0].data == "x"


def test_data_without_trailing_blank_line_is_buffered_then_flushed_on_close() -> None:
    p = SseParser()
    assert p.feed(b'data: {"a":1}') == []
    assert [e.data for e in p.close()] == ['{"a":1}']


def test_value_without_space_after_colon_is_kept_verbatim() -> None:
    assert parse(b'data:{"a":1}\n\n')[0].data == '{"a":1}'
