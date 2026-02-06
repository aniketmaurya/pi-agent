from __future__ import annotations

import asyncio

import pytest

from pi_py.agent_core import (
    Agent,
    AgentLoopConfig,
    AssistantMessageEventStream,
    LlmContext,
    Model,
    TextContent,
)
from pi_py.agent_core.types import AssistantMessage, StopReason, Usage


def make_assistant(text: str, *, stop_reason: StopReason = "stop") -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="mock-api",
        provider="mock-provider",
        model="mock-model",
        stop_reason=stop_reason,
        usage=Usage(),
    )


def blocking_stream(
    _model: Model,
    _context: LlmContext,
    _config: AgentLoopConfig,
    _abort_event: asyncio.Event | None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()

    async def _emit() -> None:
        await asyncio.sleep(0)
        stream.push({"type": "start", "partial": make_assistant("")})

    asyncio.create_task(_emit())
    return stream


@pytest.mark.asyncio
async def test_agent_has_sane_default_state() -> None:
    agent = Agent(stream_fn=lambda *_: AssistantMessageEventStream())

    assert agent.state.system_prompt == ""
    assert agent.state.model is not None
    assert agent.state.thinking_level == "off"
    assert agent.state.tools == []
    assert agent.state.messages == []
    assert agent.state.is_streaming is False


@pytest.mark.asyncio
async def test_prompt_while_streaming_raises() -> None:
    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()

        async def _emit() -> None:
            await asyncio.sleep(0)
            stream.push({"type": "start", "partial": make_assistant("")})
            while abort_event is not None and not abort_event.is_set():
                await asyncio.sleep(0.01)
            stream.push(
                {
                    "type": "error",
                    "reason": "aborted",
                    "error": make_assistant("", stop_reason="aborted"),
                }
            )

        asyncio.create_task(_emit())
        return stream

    agent = Agent(stream_fn=stream_fn)

    first = asyncio.create_task(agent.prompt("first"))
    await asyncio.sleep(0.02)

    with pytest.raises(RuntimeError, match="already processing a prompt"):
        await agent.prompt("second")

    agent.abort()
    await first


@pytest.mark.asyncio
async def test_continue_while_streaming_raises() -> None:
    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()

        async def _emit() -> None:
            await asyncio.sleep(0)
            stream.push({"type": "start", "partial": make_assistant("")})
            while abort_event is not None and not abort_event.is_set():
                await asyncio.sleep(0.01)
            stream.push(
                {
                    "type": "error",
                    "reason": "aborted",
                    "error": make_assistant("", stop_reason="aborted"),
                }
            )

        asyncio.create_task(_emit())
        return stream

    agent = Agent(stream_fn=stream_fn)

    first = asyncio.create_task(agent.prompt("first"))
    await asyncio.sleep(0.02)

    with pytest.raises(RuntimeError, match="already processing"):
        await agent.continue_()

    agent.abort()
    await first


@pytest.mark.asyncio
async def test_continue_from_assistant_message_raises() -> None:
    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        _abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        return AssistantMessageEventStream()

    agent = Agent(stream_fn=stream_fn)
    agent.append_message(make_assistant("done"))

    with pytest.raises(RuntimeError, match="Cannot continue from message role"):
        await agent.continue_()


@pytest.mark.asyncio
async def test_agent_forwards_session_id_to_stream_fn() -> None:
    received_session_ids: list[str | None] = []

    def stream_fn(
        _model: Model,
        _context: LlmContext,
        config: AgentLoopConfig,
        _abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        received_session_ids.append(config.session_id)
        stream = AssistantMessageEventStream()

        async def _emit() -> None:
            await asyncio.sleep(0)
            stream.push({"type": "done", "reason": "stop", "message": make_assistant("ok")})

        asyncio.create_task(_emit())
        return stream

    agent = Agent(stream_fn=stream_fn, session_id="session-1")
    agent.set_model(Model(id="mock-model", provider="mock-provider", api="mock-api"))

    await agent.prompt("hello")

    agent.session_id = "session-2"
    await agent.prompt("again")

    assert received_session_ids == ["session-1", "session-2"]
