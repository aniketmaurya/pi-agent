from __future__ import annotations

import pytest

from pi_py.agent_core.event_stream import AssistantMessageEventStream, EventStream
from pi_py.agent_core.types import AssistantMessage, StopReason, TextContent, Usage


def make_assistant(text: str, stop_reason: StopReason = "stop") -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="mock-api",
        provider="mock-provider",
        model="mock-model",
        stop_reason=stop_reason,
        usage=Usage(),
    )


@pytest.mark.asyncio
async def test_assistant_stream_result_from_done_event() -> None:
    stream = AssistantMessageEventStream()
    message = make_assistant("hello")

    stream.push({"type": "done", "reason": "stop", "message": message})

    seen = []
    async for event in stream:
        seen.append(event["type"])

    assert seen == ["done"]
    assert await stream.result() == message


@pytest.mark.asyncio
async def test_assistant_stream_result_from_error_event() -> None:
    stream = AssistantMessageEventStream()
    message = make_assistant("", stop_reason="error")

    stream.push({"type": "error", "reason": "error", "error": message})

    async for _ in stream:
        pass

    assert await stream.result() == message


@pytest.mark.asyncio
async def test_generic_event_stream_end_without_result_raises() -> None:
    stream: EventStream[dict[str, str], str] = EventStream(
        is_complete=lambda event: event["type"] == "done",
        extract_result=lambda event: event["value"],
    )

    stream.push({"type": "chunk", "value": "a"})
    stream.end()

    async for _ in stream:
        pass

    with pytest.raises(RuntimeError, match="terminal result"):
        await stream.result()
