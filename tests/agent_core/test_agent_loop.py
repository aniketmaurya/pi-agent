from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from pi_py.agent_core import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    AssistantMessageEventStream,
    LlmContext,
    Model,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
    agent_loop,
    default_convert_to_llm,
)
from pi_py.agent_core.types import (
    AgentMessage,
    AssistantContentBlock,
    AssistantMessageEvent,
    StopReason,
)


def make_model() -> Model:
    return Model(id="mock-model", provider="mock-provider", api="mock-api")


def make_assistant(
    content: list[AssistantContentBlock],
    *,
    stop_reason: StopReason = "stop",
) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="mock-api",
        provider="mock-provider",
        model="mock-model",
        stop_reason=stop_reason,
        usage=Usage(),
    )


def make_stream(*events: AssistantMessageEvent) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()

    async def _emit() -> None:
        await asyncio.sleep(0)
        for event in events:
            stream.push(event)

    asyncio.create_task(_emit())
    return stream


@pytest.mark.asyncio
async def test_agent_loop_emits_basic_event_flow() -> None:
    prompt = UserMessage(content="hello")
    context = AgentContext(system_prompt="", messages=[], tools=[])

    config = AgentLoopConfig(
        model=make_model(),
        convert_to_llm=default_convert_to_llm,
    )

    stream = agent_loop(
        [prompt],
        context,
        config,
        stream_fn=lambda *_: make_stream(
            {
                "type": "done",
                "reason": "stop",
                "message": make_assistant([TextContent(text="hi")]),
            }
        ),
    )

    event_types: list[str] = []
    async for event in stream:
        event_types.append(event["type"])

    result_messages = await stream.result()

    assert [getattr(message, "role", "") for message in result_messages] == ["user", "assistant"]
    assert event_types == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_calls() -> None:
    executed: list[str] = []

    async def run_echo(
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[Any]], None] | None = None,
    ) -> AgentToolResult[Any]:
        del tool_call_id, abort_event, on_update
        value = str(params["value"])
        executed.append(value)
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {value}")],
            details={"value": value},
        )

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters={"type": "object"},
        execute=run_echo,
    )

    prompt = UserMessage(content="do the thing")
    context = AgentContext(system_prompt="", messages=[], tools=[tool])

    config = AgentLoopConfig(
        model=make_model(),
        convert_to_llm=default_convert_to_llm,
    )

    call_count = 0

    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        _abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        nonlocal call_count
        if call_count == 0:
            message = make_assistant(
                [ToolCall(id="call-1", name="echo", arguments={"value": "hello"})],
                stop_reason="toolUse",
            )
            call_count += 1
            return make_stream({"type": "done", "reason": "toolUse", "message": message})

        call_count += 1
        message = make_assistant([TextContent(text="done")])
        return make_stream({"type": "done", "reason": "stop", "message": message})

    stream = agent_loop([prompt], context, config, stream_fn=stream_fn)

    event_types: list[str] = []
    async for event in stream:
        event_types.append(event["type"])

    assert executed == ["hello"]
    assert "tool_execution_start" in event_types
    assert "tool_execution_end" in event_types


@pytest.mark.asyncio
async def test_agent_loop_skips_remaining_tools_when_steering_arrives() -> None:
    executed: list[str] = []

    async def run_tool(
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[Any]], None] | None = None,
    ) -> AgentToolResult[Any]:
        del tool_call_id, abort_event, on_update
        value = str(params["value"])
        executed.append(value)
        return AgentToolResult(content=[TextContent(text=value)], details={})

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo tool",
        execute=run_tool,
    )

    prompt = UserMessage(content="run two tools")
    context = AgentContext(system_prompt="", messages=[], tools=[tool])

    steering_state = {"count": 0}

    async def get_steering_messages() -> list[AgentMessage]:
        steering_state["count"] += 1
        if steering_state["count"] == 2:
            return [UserMessage(content="stop now")]
        return []

    config = AgentLoopConfig(
        model=make_model(),
        convert_to_llm=default_convert_to_llm,
        get_steering_messages=get_steering_messages,
    )

    call_count = 0

    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        _abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return make_stream(
                {
                    "type": "done",
                    "reason": "toolUse",
                    "message": make_assistant(
                        [
                            ToolCall(id="call-1", name="echo", arguments={"value": "first"}),
                            ToolCall(id="call-2", name="echo", arguments={"value": "second"}),
                        ],
                        stop_reason="toolUse",
                    ),
                }
            )

        call_count += 1
        return make_stream(
            {
                "type": "done",
                "reason": "stop",
                "message": make_assistant([TextContent(text="after steering")]),
            }
        )

    stream = agent_loop([prompt], context, config, stream_fn=stream_fn)
    events: list[dict[str, Any]] = []
    async for event in stream:
        events.append(event)

    tool_end_events = [event for event in events if event["type"] == "tool_execution_end"]

    assert executed == ["first"]
    assert len(tool_end_events) == 2
    assert tool_end_events[0]["is_error"] is False
    assert tool_end_events[1]["is_error"] is True
