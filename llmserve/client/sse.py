"""Incremental Server-Sent Events parser (Phase 1 design item 2).

The client reads the HTTP body as a byte stream, so an SSE event can be split at any byte
boundary: the parser keeps a buffer and only emits events on a blank line. It handles `data:`,
`event:`, `id:`, `retry:`, multi-line data, `: keep-alive` comments, LF and CRLF line endings,
and a payload split across TCP reads.

Deviation from the WHATWG spec: `close()` dispatches a buffered event instead of discarding it.
A truncated stream then surfaces as a parse error on the last chunk rather than silently losing
data, which is what we want when a request's tail went missing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SseEvent:
    data: str
    event: str = "message"
    id: str | None = None
    retry_ms: int | None = None


class SseParser:
    def __init__(self) -> None:
        self._buffer = b""
        self._data: list[str] = []
        self._event: str | None = None
        self._id: str | None = None
        self._retry_ms: int | None = None

    def feed(self, chunk: bytes) -> list[SseEvent]:
        """Consume bytes and return every event completed by them."""
        self._buffer += chunk
        events: list[SseEvent] = []
        while True:
            nl = self._buffer.find(b"\n")
            if nl < 0:
                break
            line, self._buffer = self._buffer[:nl], self._buffer[nl + 1 :]
            if line.endswith(b"\r"):
                line = line[:-1]
            event = self._line(line.decode("utf-8", errors="replace"))
            if event is not None:
                events.append(event)
        return events

    def close(self) -> list[SseEvent]:
        """Flush a trailing event that never saw its blank line (truncated stream)."""
        if self._buffer:
            line = self._buffer[:-1] if self._buffer.endswith(b"\r") else self._buffer
            self._buffer = b""
            self._line(line.decode("utf-8", errors="replace"))
        event = self._dispatch()
        return [event] if event is not None else []

    def _line(self, line: str) -> SseEvent | None:
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None  # comment: keep-alive or a retry hint
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value
        elif field == "id":
            self._id = value
        elif field == "retry":
            if value.isdigit():
                self._retry_ms = int(value)
        # Unknown fields are ignored, per the SSE spec.
        return None

    def _dispatch(self) -> SseEvent | None:
        if not self._data:
            self._event = None
            return None
        event = SseEvent(
            data="\n".join(self._data),
            event=self._event or "message",
            id=self._id,
            retry_ms=self._retry_ms,
        )
        self._data = []
        self._event = None
        return event
