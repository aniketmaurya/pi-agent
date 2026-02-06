from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, TypeVar, cast

from ...agent_core.event_stream import AssistantMessageEventStream
from ...agent_core.types import (
    AgentTool,
    AssistantContentBlock,
    AssistantMessage,
    AssistantStream,
    ImageContent,
    LlmContext,
    Model,
    StopReason,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)
from ..types import PiAIRequest
from .openai import _load_async_openai_class

DoneReason = Literal["stop", "length", "toolUse"]
OpenAICompletionsRequestFn: TypeAlias = Callable[
    [dict[str, Any], str, str | None],
    Awaitable[Mapping[str, Any]],
]
OpenAICompletionsStreamRequestFn: TypeAlias = Callable[
    [dict[str, Any], str, str | None],
    AsyncIterator[Mapping[str, Any]] | Awaitable[AsyncIterator[Mapping[str, Any]]],
]
T = TypeVar("T")


class _RequestAbortedError(RuntimeError):
    pass


@dataclass(slots=True)
class OpenAICompletionsProvider:
    request_fn: OpenAICompletionsRequestFn | None = None
    stream_request_fn: OpenAICompletionsStreamRequestFn | None = None
    api_key_env: str = "OPENAI_API_KEY"

    async def stream(
        self,
        request: PiAIRequest,
        abort_event: asyncio.Event | None = None,
    ) -> AssistantStream:
        stream = AssistantMessageEventStream()
        asyncio.create_task(self._emit(stream, request, abort_event))
        return stream

    async def _emit(
        self,
        stream: AssistantMessageEventStream,
        request: PiAIRequest,
        abort_event: asyncio.Event | None,
    ) -> None:
        await asyncio.sleep(0)

        if abort_event is not None and abort_event.is_set():
            stream.push(
                {
                    "type": "error",
                    "reason": "aborted",
                    "error": _assistant_error_message(
                        model=request.model,
                        stop_reason="aborted",
                        error_message="Request aborted",
                    ),
                }
            )
            return

        try:
            api_key = _resolve_api_key(request, self.api_key_env)
            base_url = request.model.base_url or None

            if self.stream_request_fn is not None:
                payload = _build_openai_completions_payload(request, stream=True)
                events = await _maybe_await(
                    self.stream_request_fn(payload, api_key, base_url)
                )
                await _consume_openai_completions_stream(
                    stream=stream,
                    model=request.model,
                    events=events,
                    abort_event=abort_event,
                )
                return

            if self.request_fn is not None:
                payload = _build_openai_completions_payload(request, stream=False)
                response = await self.request_fn(payload, api_key, base_url)
                assistant_message = _assistant_from_openai_completions_response(
                    model=request.model,
                    response=response,
                )
                _push_terminal_message(stream, assistant_message)
                return

            payload = _build_openai_completions_payload(request, stream=True)
            events = _stream_openai_completions_chunks(payload, api_key, base_url)
            await _consume_openai_completions_stream(
                stream=stream,
                model=request.model,
                events=events,
                abort_event=abort_event,
            )
        except _RequestAbortedError:
            stream.push(
                {
                    "type": "error",
                    "reason": "aborted",
                    "error": _assistant_error_message(
                        model=request.model,
                        stop_reason="aborted",
                        error_message="Request aborted",
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            stream.push(
                {
                    "type": "error",
                    "reason": "error",
                    "error": _assistant_error_message(
                        model=request.model,
                        stop_reason="error",
                        error_message=str(exc),
                    ),
                }
            )


@dataclass(slots=True)
class _OpenAICompletionsStreamingState:
    partial: AssistantMessage
    current_kind: Literal["text", "thinking", "tool"] | None = None
    current_content_index: int | None = None
    tool_arg_buffers: dict[int, str] = field(default_factory=dict)


def _push_terminal_message(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
) -> None:
    if message.stop_reason in {"error", "aborted"}:
        stream.push(
            {
                "type": "error",
                "reason": "aborted" if message.stop_reason == "aborted" else "error",
                "error": message,
            }
        )
        return

    stream.push(
        {
            "type": "done",
            "reason": _done_reason_for_message(message),
            "message": message,
        }
    )


async def _consume_openai_completions_stream(
    *,
    stream: AssistantMessageEventStream,
    model: Model,
    events: AsyncIterator[Mapping[str, Any]],
    abort_event: asyncio.Event | None,
) -> None:
    state = _OpenAICompletionsStreamingState(partial=_new_partial_message(model))
    stream.push({"type": "start", "partial": state.partial})

    async for raw_chunk in events:
        if abort_event is not None and abort_event.is_set():
            raise _RequestAbortedError("Request aborted")
        chunk = _chunk_to_mapping(raw_chunk)
        _apply_openai_completions_chunk(stream=stream, state=state, chunk=chunk)

    _finish_current_block(state=state, stream=stream)

    if state.partial.stop_reason in {"error", "aborted"}:
        raise RuntimeError("OpenAI chat completion failed.")

    _push_terminal_message(stream, _clone_assistant_message(state.partial))


def _apply_openai_completions_chunk(
    *,
    stream: AssistantMessageEventStream,
    state: _OpenAICompletionsStreamingState,
    chunk: Mapping[str, Any],
) -> None:
    usage = chunk.get("usage")
    if isinstance(usage, Mapping):
        state.partial.usage = _extract_usage(usage)

    choices = _as_mapping_list(chunk.get("choices"))
    if not choices:
        return

    choice = choices[0]
    finish_reason = _as_str(choice.get("finish_reason"))
    if finish_reason is not None:
        state.partial.stop_reason = _map_stop_reason(finish_reason)

    delta = choice.get("delta")
    if not isinstance(delta, Mapping):
        return

    text_delta = _as_str(delta.get("content"))
    if text_delta:
        content_index = _ensure_text_block(state=state, stream=stream)
        text_block = cast(TextContent, state.partial.content[content_index])
        text_block.text += text_delta
        stream.push(
            {
                "type": "text_delta",
                "content_index": content_index,
                "delta": text_delta,
                "partial": state.partial,
            }
        )

    reasoning_field, reasoning_delta = _first_reasoning_delta(delta)
    if reasoning_field is not None and reasoning_delta is not None:
        content_index = _ensure_thinking_block(
            state=state,
            stream=stream,
            signature=reasoning_field,
        )
        thinking_block = cast(ThinkingContent, state.partial.content[content_index])
        thinking_block.thinking += reasoning_delta
        stream.push(
            {
                "type": "thinking_delta",
                "content_index": content_index,
                "delta": reasoning_delta,
                "partial": state.partial,
            }
        )

    tool_calls = _as_mapping_list(delta.get("tool_calls"))
    for tool_call in tool_calls:
        _apply_tool_call_delta(stream=stream, state=state, tool_call=tool_call)

    reasoning_details = _as_mapping_list(delta.get("reasoning_details"))
    for detail in reasoning_details:
        detail_type = _as_str(detail.get("type"))
        detail_id = _as_str(detail.get("id"))
        detail_data = detail.get("data")
        if detail_type != "reasoning.encrypted" or not detail_id or detail_data is None:
            continue

        serialized = _safe_json_dumps(detail)
        if serialized is None:
            continue
        for block in state.partial.content:
            if isinstance(block, ToolCall) and block.id == detail_id:
                block.thought_signature = serialized
                break


def _ensure_text_block(
    *,
    state: _OpenAICompletionsStreamingState,
    stream: AssistantMessageEventStream,
) -> int:
    if state.current_kind == "text" and state.current_content_index is not None:
        return state.current_content_index

    _finish_current_block(state=state, stream=stream)

    content_index = len(state.partial.content)
    state.partial.content.append(TextContent(text=""))
    state.current_kind = "text"
    state.current_content_index = content_index
    stream.push(
        {
            "type": "text_start",
            "content_index": content_index,
            "partial": state.partial,
        }
    )
    return content_index


def _ensure_thinking_block(
    *,
    state: _OpenAICompletionsStreamingState,
    stream: AssistantMessageEventStream,
    signature: str,
) -> int:
    if state.current_kind == "thinking" and state.current_content_index is not None:
        current = state.partial.content[state.current_content_index]
        if isinstance(current, ThinkingContent):
            if current.thinking_signature is None:
                current.thinking_signature = signature
            if current.thinking_signature == signature:
                return state.current_content_index

    _finish_current_block(state=state, stream=stream)

    content_index = len(state.partial.content)
    state.partial.content.append(
        ThinkingContent(thinking="", thinking_signature=signature)
    )
    state.current_kind = "thinking"
    state.current_content_index = content_index
    stream.push(
        {
            "type": "thinking_start",
            "content_index": content_index,
            "partial": state.partial,
        }
    )
    return content_index


def _apply_tool_call_delta(
    *,
    stream: AssistantMessageEventStream,
    state: _OpenAICompletionsStreamingState,
    tool_call: Mapping[str, Any],
) -> None:
    tool_id = _as_str(tool_call.get("id"))
    function = tool_call.get("function")
    function_mapping = function if isinstance(function, Mapping) else {}
    tool_name = _as_str(function_mapping.get("name"))

    existing_tool_call: ToolCall | None = None
    if state.current_kind == "tool" and state.current_content_index is not None:
        current = state.partial.content[state.current_content_index]
        if isinstance(current, ToolCall):
            existing_tool_call = current

    if (
        existing_tool_call is None
        or (tool_id and existing_tool_call.id and existing_tool_call.id != tool_id)
    ):
        _finish_current_block(state=state, stream=stream)
        content_index = len(state.partial.content)
        created_tool_call = ToolCall(
            id=tool_id or "",
            name=tool_name or "tool",
            arguments={},
        )
        state.partial.content.append(created_tool_call)
        state.current_kind = "tool"
        state.current_content_index = content_index
        state.tool_arg_buffers[content_index] = ""
        stream.push(
            {
                "type": "toolcall_start",
                "content_index": content_index,
                "partial": state.partial,
            }
        )
        existing_tool_call = created_tool_call
    else:
        content_index_opt = state.current_content_index
        assert content_index_opt is not None
        content_index = content_index_opt

    if tool_id:
        existing_tool_call.id = tool_id
    if tool_name:
        existing_tool_call.name = tool_name

    delta = _as_str(function_mapping.get("arguments")) or ""
    if delta:
        buffer = state.tool_arg_buffers.get(content_index, "")
        buffer += delta
        state.tool_arg_buffers[content_index] = buffer
        existing_tool_call.arguments = _parse_streaming_json(buffer)

    stream.push(
        {
            "type": "toolcall_delta",
            "content_index": content_index,
            "delta": delta,
            "partial": state.partial,
        }
    )


def _finish_current_block(
    *,
    state: _OpenAICompletionsStreamingState,
    stream: AssistantMessageEventStream,
) -> None:
    if state.current_kind is None or state.current_content_index is None:
        return

    content_index = state.current_content_index
    block = state.partial.content[content_index]

    if isinstance(block, TextContent):
        stream.push(
            {
                "type": "text_end",
                "content_index": content_index,
                "content": block.text,
                "partial": state.partial,
            }
        )
    elif isinstance(block, ThinkingContent):
        stream.push(
            {
                "type": "thinking_end",
                "content_index": content_index,
                "content": block.thinking,
                "partial": state.partial,
            }
        )
    elif isinstance(block, ToolCall):
        buffer = state.tool_arg_buffers.get(content_index, "")
        block.arguments = _extract_tool_call_arguments(buffer)
        stream.push(
            {
                "type": "toolcall_end",
                "content_index": content_index,
                "tool_call": block,
                "partial": state.partial,
            }
        )

    state.current_kind = None
    state.current_content_index = None


def _first_reasoning_delta(delta: Mapping[str, Any]) -> tuple[str | None, str | None]:
    reasoning_fields = ("reasoning_content", "reasoning", "reasoning_text")
    for field_name in reasoning_fields:
        value = _as_str(delta.get(field_name))
        if value:
            return field_name, value
    return None, None


def _build_openai_completions_payload(
    request: PiAIRequest,
    *,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model.id,
        "messages": _to_openai_completions_messages(request.context),
        "stream": stream,
    }

    if stream:
        payload["stream_options"] = {"include_usage": True}

    tools = _to_openai_completions_tools(request.context.tools)
    if tools:
        payload["tools"] = tools

    reasoning_effort = _map_reasoning_effort(request.reasoning)
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    return payload


def _to_openai_completions_messages(context: LlmContext) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if context.system_prompt:
        messages.append({"role": "system", "content": context.system_prompt})

    context_messages = context.messages
    index = 0
    while index < len(context_messages):
        message = context_messages[index]

        if isinstance(message, UserMessage):
            messages.append(_to_openai_completions_user_message(message))
            index += 1
            continue

        if isinstance(message, AssistantMessage):
            assistant_message = _to_openai_completions_assistant_message(message)
            if assistant_message is not None:
                messages.append(assistant_message)
            index += 1
            continue

        if isinstance(message, ToolResultMessage):
            image_blocks: list[dict[str, Any]] = []
            while index < len(context_messages):
                current = context_messages[index]
                if not isinstance(current, ToolResultMessage):
                    break

                text_result = _tool_result_text(current)
                has_images = any(
                    isinstance(block, ImageContent) for block in current.content
                )
                if not text_result and has_images:
                    text_result = "(see attached image)"

                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "content": text_result,
                    "tool_call_id": current.tool_call_id,
                }
                if current.tool_name:
                    tool_message["name"] = current.tool_name
                messages.append(tool_message)

                for block in current.content:
                    if isinstance(block, ImageContent):
                        image_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block.mime_type};base64,{block.data}",
                                },
                            }
                        )

                index += 1

            if image_blocks:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Attached image(s) from tool result:",
                            },
                            *image_blocks,
                        ],
                    }
                )
            continue

    return messages


def _to_openai_completions_user_message(message: UserMessage) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": "user", "content": message.content}

    content: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            content.append({"type": "text", "text": block.text})
            continue

        if isinstance(block, ImageContent):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.mime_type};base64,{block.data}",
                    },
                }
            )

    return {"role": "user", "content": content}


def _to_openai_completions_assistant_message(
    message: AssistantMessage,
) -> dict[str, Any] | None:
    assistant_message: dict[str, Any] = {"role": "assistant", "content": None}

    text_blocks = [
        block
        for block in message.content
        if isinstance(block, TextContent) and block.text.strip()
    ]
    thinking_blocks = [
        block
        for block in message.content
        if isinstance(block, ThinkingContent) and block.thinking.strip()
    ]

    if text_blocks:
        assistant_message["content"] = [
            {"type": "text", "text": block.text} for block in text_blocks
        ]

    if thinking_blocks:
        thinking_text = "\n\n".join(block.thinking for block in thinking_blocks)
        content = assistant_message.get("content")
        if isinstance(content, list):
            content.insert(0, {"type": "text", "text": thinking_text})
        else:
            assistant_message["content"] = [{"type": "text", "text": thinking_text}]

    tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
    if tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(
                        tool_call.arguments,
                        separators=(",", ":"),
                    ),
                },
            }
            for tool_call in tool_calls
        ]

    content_value = assistant_message.get("content")
    has_content = False
    if isinstance(content_value, str):
        has_content = bool(content_value)
    elif isinstance(content_value, list):
        has_content = len(content_value) > 0

    if not has_content and "tool_calls" not in assistant_message:
        return None

    return assistant_message


def _to_openai_completions_tools(
    tools: Sequence[AgentTool] | None,
) -> list[dict[str, Any]]:
    if not tools:
        return []

    payload_tools: list[dict[str, Any]] = []
    for tool in tools:
        payload_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": _coerce_json_schema(tool.parameters),
                },
            }
        )
    return payload_tools


def _assistant_from_openai_completions_response(
    *,
    model: Model,
    response: Mapping[str, Any],
) -> AssistantMessage:
    choices = _as_mapping_list(response.get("choices"))
    first_choice = choices[0] if choices else {}
    raw_message = first_choice.get("message")
    message: Mapping[str, Any]
    if isinstance(raw_message, Mapping):
        message = raw_message
    else:
        message = {}

    content: list[AssistantContentBlock] = []

    content_value = message.get("content")
    if isinstance(content_value, str):
        if content_value:
            content.append(TextContent(text=content_value))
    else:
        for part in _as_mapping_list(content_value):
            part_type = _as_str(part.get("type"))
            if part_type == "text":
                text = _as_str(part.get("text"))
                if text:
                    content.append(TextContent(text=text))

    for field_name in ("reasoning_content", "reasoning", "reasoning_text"):
        reasoning = _as_str(message.get(field_name))
        if reasoning:
            content.append(
                ThinkingContent(
                    thinking=reasoning,
                    thinking_signature=field_name,
                )
            )

    tool_calls = _as_mapping_list(message.get("tool_calls"))
    for tool_call in tool_calls:
        function = tool_call.get("function")
        function_mapping = function if isinstance(function, Mapping) else {}
        tool_id = _as_str(tool_call.get("id"))
        tool_name = _as_str(function_mapping.get("name"))
        if not tool_id or not tool_name:
            continue
        arguments = _extract_tool_call_arguments(function_mapping.get("arguments"))
        content.append(ToolCall(id=tool_id, name=tool_name, arguments=arguments))

    error_message = _extract_error_message(response)
    stop_reason = _derive_stop_reason(
        content=content,
        finish_reason=_as_str(first_choice.get("finish_reason")),
        error_message=error_message,
    )
    usage = _extract_usage(response.get("usage"))

    if not content:
        content = [TextContent(text="")]

    return AssistantMessage(
        content=content,
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=usage,
        stop_reason=stop_reason,
        error_message=error_message,
    )


def _derive_stop_reason(
    *,
    content: Sequence[AssistantContentBlock],
    finish_reason: str | None,
    error_message: str | None,
) -> StopReason:
    if error_message:
        return "error"

    mapped_reason = _map_stop_reason(finish_reason)
    if mapped_reason != "stop":
        return mapped_reason

    if any(isinstance(block, ToolCall) for block in content):
        return "toolUse"
    return "stop"


def _map_stop_reason(reason: str | None) -> StopReason:
    if reason in {None, "stop"}:
        return "stop"
    if reason == "length":
        return "length"
    if reason in {"function_call", "tool_calls"}:
        return "toolUse"
    if reason == "content_filter":
        return "error"
    return "stop"


def _extract_error_message(response: Mapping[str, Any]) -> str | None:
    error = response.get("error")
    if isinstance(error, Mapping):
        message = _as_str(error.get("message"))
        if message:
            return message
    return None


def _extract_usage(usage_data: Any) -> Usage:
    if not isinstance(usage_data, Mapping):
        return Usage(cost=UsageCost())

    prompt_tokens = _as_int(usage_data.get("prompt_tokens"))
    completion_tokens = _as_int(usage_data.get("completion_tokens"))
    total_tokens = _as_int(usage_data.get("total_tokens"))

    prompt_details = usage_data.get("prompt_tokens_details")
    cached_tokens = 0
    if isinstance(prompt_details, Mapping):
        cached_tokens = _as_int(prompt_details.get("cached_tokens"))

    completion_details = usage_data.get("completion_tokens_details")
    reasoning_tokens = 0
    if isinstance(completion_details, Mapping):
        reasoning_tokens = _as_int(completion_details.get("reasoning_tokens"))

    input_tokens = max(0, prompt_tokens - cached_tokens)
    output_tokens = completion_tokens + reasoning_tokens
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens + cached_tokens

    return Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cached_tokens,
        cache_write=0,
        total_tokens=total_tokens,
        cost=UsageCost(),
    )


def _extract_tool_call_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}

        if isinstance(parsed, Mapping):
            return {str(key): value for key, value in parsed.items()}
    return {}


def _parse_streaming_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    return _extract_tool_call_arguments(raw)


def _tool_result_text(message: ToolResultMessage) -> str:
    text_chunks = [
        block.text
        for block in message.content
        if isinstance(block, TextContent)
    ]
    return "\n".join(text_chunks).strip()


def _coerce_json_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {"type": "object", "properties": {}}

    schema_dict = dict(schema)
    if "type" not in schema_dict:
        schema_dict["type"] = "object"
    return schema_dict


def _resolve_api_key(request: PiAIRequest, api_key_env: str) -> str:
    if request.api_key:
        return request.api_key

    api_key = os.getenv(api_key_env)
    if api_key:
        return api_key

    raise RuntimeError(
        "Missing OpenAI API key. Set request.api_key or OPENAI_API_KEY."
    )


def _map_reasoning_effort(reasoning: ThinkingLevel | None) -> str | None:
    if reasoning in {None, "off"}:
        return None
    if reasoning in {"minimal", "low", "medium"}:
        return reasoning
    return "high"


def _new_partial_message(model: Model) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(cost=UsageCost()),
        stop_reason="stop",
    )


def _clone_assistant_message(message: AssistantMessage) -> AssistantMessage:
    cloned_content: list[AssistantContentBlock] = []
    for block in message.content:
        if isinstance(block, TextContent):
            cloned_content.append(
                TextContent(
                    text=block.text,
                    text_signature=block.text_signature,
                )
            )
            continue

        if isinstance(block, ThinkingContent):
            cloned_content.append(
                ThinkingContent(
                    thinking=block.thinking,
                    thinking_signature=block.thinking_signature,
                )
            )
            continue

        if isinstance(block, ToolCall):
            cloned_content.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.arguments),
                    thought_signature=block.thought_signature,
                )
            )

    return AssistantMessage(
        content=cloned_content,
        api=message.api,
        provider=message.provider,
        model=message.model,
        usage=message.usage,
        stop_reason=message.stop_reason,
        error_message=message.error_message,
    )


def _done_reason_for_message(message: AssistantMessage) -> DoneReason:
    if message.stop_reason == "toolUse":
        return "toolUse"
    if message.stop_reason == "length":
        return "length"
    return "stop"


def _assistant_error_message(
    *,
    model: Model,
    stop_reason: StopReason,
    error_message: str,
) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(cost=UsageCost()),
        stop_reason=stop_reason,
        error_message=error_message,
    )


def _as_mapping_list(raw: Any) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list):
        return []

    mappings: list[Mapping[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            mappings.append(item)
    return mappings


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _safe_json_dumps(value: Any) -> str | None:
    try:
        return json.dumps(value, separators=(",", ":"))
    except TypeError:
        return None


def _chunk_to_mapping(chunk: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(chunk, Mapping):
        return dict(chunk)

    to_dict = getattr(chunk, "to_dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict(warnings=False)
        except TypeError:
            dumped = to_dict()
        if isinstance(dumped, Mapping):
            return dict(dumped)

    model_dump = getattr(chunk, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(warnings=False)
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)

    fields = ("id", "created", "model", "choices", "usage")
    mapped: dict[str, Any] = {}
    for field_name in fields:
        if hasattr(chunk, field_name):
            mapped[field_name] = getattr(chunk, field_name)
    return mapped


def _response_to_mapping(response: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return dict(response)

    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict(warnings=False)
        except TypeError:
            dumped = to_dict()
        if isinstance(dumped, Mapping):
            return dict(dumped)

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(warnings=False)
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)

    raise RuntimeError(
        "Unsupported OpenAI chat completion response type. "
        f"Expected mapping-like object, got: {type(response).__name__}"
    )


async def _stream_openai_completions_chunks(
    payload: dict[str, Any],
    api_key: str,
    base_url: str | None,
) -> AsyncIterator[Mapping[str, Any]]:
    async_openai = _load_async_openai_class()
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    async with async_openai(**client_kwargs) as client:
        stream = await client.chat.completions.create(**payload)
        async for chunk in stream:
            yield _chunk_to_mapping(chunk)


async def _request_openai_completions(
    payload: dict[str, Any],
    api_key: str,
    base_url: str | None,
) -> dict[str, Any]:
    async_openai = _load_async_openai_class()
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    async with async_openai(**client_kwargs) as client:
        response = await client.chat.completions.create(**payload)
    return _response_to_mapping(response)


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value
