from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, TypeVar, cast

from .event_stream import EventStream
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    GetMessagesFn,
    LlmContext,
    StreamFn,
    TextContent,
    ToolCall,
    ToolResultMessage,
    assistant_tool_calls,
    message_role,
)


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    abort_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    stream = _create_agent_stream()

    async def _run() -> None:
        new_messages: list[AgentMessage] = list(prompts)
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=[*context.messages, *prompts],
            tools=context.tools,
        )

        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})

        for prompt in prompts:
            stream.push({"type": "message_start", "message": prompt})
            stream.push({"type": "message_end", "message": prompt})

        await _run_loop(current_context, new_messages, config, abort_event, stream, stream_fn)

    asyncio.create_task(_run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    abort_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    if message_role(context.messages[-1]) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = _create_agent_stream()

    async def _run() -> None:
        new_messages: list[AgentMessage] = []
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages),
            tools=context.tools,
        )

        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})

        await _run_loop(current_context, new_messages, config, abort_event, stream, stream_fn)

    asyncio.create_task(_run())
    return stream


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    return EventStream[AgentEvent, list[AgentMessage]](
        is_complete=lambda event: event["type"] == "agent_end",
        extract_result=lambda event: event["messages"] if event["type"] == "agent_end" else [],
    )


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    abort_event: asyncio.Event | None,
    stream: EventStream[AgentEvent, list[AgentMessage]],
    stream_fn: StreamFn | None,
) -> None:
    first_turn = True
    pending_messages = await _maybe_get_messages(config.get_steering_messages)

    while True:
        has_more_tool_calls = True
        steering_after_tools: list[AgentMessage] | None = None

        while has_more_tool_calls or pending_messages:
            if not first_turn:
                stream.push({"type": "turn_start"})
            else:
                first_turn = False

            if pending_messages:
                for message in pending_messages:
                    stream.push({"type": "message_start", "message": message})
                    stream.push({"type": "message_end", "message": message})
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            message = await _stream_assistant_response(
                current_context,
                config,
                abort_event,
                stream,
                stream_fn,
            )
            new_messages.append(message)

            if message.stop_reason in {"error", "aborted"}:
                stream.push({"type": "turn_end", "message": message, "tool_results": []})
                stream.push({"type": "agent_end", "messages": new_messages})
                stream.end(new_messages)
                return

            tool_calls = assistant_tool_calls(message)
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                tool_execution = await _execute_tool_calls(
                    tools=current_context.tools,
                    assistant_message=message,
                    abort_event=abort_event,
                    stream=stream,
                    get_steering_messages=config.get_steering_messages,
                )
                tool_results.extend(tool_execution["tool_results"])
                steering_after_tools = tool_execution.get("steering_messages")

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            stream.push({"type": "turn_end", "message": message, "tool_results": tool_results})

            if steering_after_tools:
                pending_messages = steering_after_tools
                steering_after_tools = None
            else:
                pending_messages = await _maybe_get_messages(config.get_steering_messages)

        follow_up_messages = await _maybe_get_messages(config.get_follow_up_messages)
        if follow_up_messages:
            pending_messages = follow_up_messages
            continue

        break

    stream.push({"type": "agent_end", "messages": new_messages})
    stream.end(new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    abort_event: asyncio.Event | None,
    stream: EventStream[AgentEvent, list[AgentMessage]],
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    messages = context.messages
    if config.transform_context:
        messages = await config.transform_context(messages, abort_event)

    llm_messages = await _maybe_await(config.convert_to_llm(messages))
    llm_context = LlmContext(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,
    )

    stream_function = stream_fn
    if stream_function is None:
        raise RuntimeError("stream_fn is required")

    if config.get_api_key:
        resolved_key = await _maybe_await(config.get_api_key(config.model.provider))
        if resolved_key is not None:
            config = replace(config, api_key=resolved_key)

    response = await _maybe_await(stream_function(config.model, llm_context, config, abort_event))

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        event_type = event["type"]

        if event_type == "start":
            partial = event.get("partial")
            if isinstance(partial, AssistantMessage):
                partial_message = partial
                context.messages.append(partial_message)
                added_partial = True
                stream.push({"type": "message_start", "message": partial_message})
            continue

        if event_type in {
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        }:
            if partial_message is not None:
                partial = event.get("partial")
                if isinstance(partial, AssistantMessage):
                    partial_message = partial
                    context.messages[-1] = partial_message
                    stream.push(
                        {
                            "type": "message_update",
                            "message": partial_message,
                            "assistant_message_event": event,
                        }
                    )
            continue

        if event_type in {"done", "error"}:
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                stream.push({"type": "message_start", "message": final_message})

            stream.push({"type": "message_end", "message": final_message})
            return final_message

    return await response.result()


async def _execute_tool_calls(
    tools: list[AgentTool] | None,
    assistant_message: AssistantMessage,
    abort_event: asyncio.Event | None,
    stream: EventStream[AgentEvent, list[AgentMessage]],
    get_steering_messages: GetMessagesFn | None,
) -> dict[str, Any]:
    tool_calls = assistant_tool_calls(assistant_message)
    results: list[ToolResultMessage] = []
    steering_messages: list[AgentMessage] | None = None

    for index, tool_call in enumerate(tool_calls):
        tool = next(
            (candidate for candidate in tools or [] if candidate.name == tool_call.name),
            None,
        )
        tool_call_id = tool_call.id
        tool_name = tool_call.name
        tool_args = tool_call.arguments

        stream.push(
            {
                "type": "tool_execution_start",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "args": tool_args,
            }
        )

        result: AgentToolResult[Any]
        is_error = False

        try:
            if tool is None:
                raise RuntimeError(f"Tool {tool_name} not found")

            def on_update(
                partial: AgentToolResult[Any],
                *,
                call_id: str = tool_call_id,
                name: str = tool_name,
                args: dict[str, Any] = tool_args,
            ) -> None:
                stream.push(
                    {
                        "type": "tool_execution_update",
                        "tool_call_id": call_id,
                        "tool_name": name,
                        "args": args,
                        "partial_result": partial,
                    }
                )

            result = await tool.execute(
                tool_call_id,
                tool_args,
                abort_event,
                on_update,
            )
        except Exception as exc:  # noqa: BLE001
            result = AgentToolResult(content=[TextContent(text=str(exc))], details={})
            is_error = True

        stream.push(
            {
                "type": "tool_execution_end",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result": result,
                "is_error": is_error,
            }
        )

        tool_result_message = ToolResultMessage(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=result.content,
            details=result.details,
            is_error=is_error,
        )

        results.append(tool_result_message)
        stream.push({"type": "message_start", "message": tool_result_message})
        stream.push({"type": "message_end", "message": tool_result_message})

        if get_steering_messages is not None:
            steering = await _maybe_get_messages(get_steering_messages)
            if steering:
                steering_messages = steering
                for skipped_call in tool_calls[index + 1 :]:
                    results.append(_skip_tool_call(skipped_call, stream))
                break

    return {"tool_results": results, "steering_messages": steering_messages}


def _skip_tool_call(
    tool_call: ToolCall,
    stream: EventStream[AgentEvent, list[AgentMessage]],
) -> ToolResultMessage:
    result: AgentToolResult[dict[str, Any]] = AgentToolResult(
        content=[TextContent(text="Skipped due to queued user message.")],
        details={},
    )

    stream.push(
        {
            "type": "tool_execution_start",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "args": tool_call.arguments,
        }
    )
    stream.push(
        {
            "type": "tool_execution_end",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "result": result,
            "is_error": True,
        }
    )

    tool_result_message = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=True,
    )

    stream.push({"type": "message_start", "message": tool_result_message})
    stream.push({"type": "message_end", "message": tool_result_message})
    return tool_result_message


T = TypeVar("T")


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value


async def _maybe_get_messages(
    getter: Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]] | None,
) -> list[AgentMessage]:
    if getter is None:
        return []

    messages = await _maybe_await(getter())
    return list(messages or [])
