from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

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
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)
from ..types import PiAIRequest

DoneReason = Literal["stop", "length", "toolUse"]
OpenAIRequestFn: TypeAlias = Callable[
    [dict[str, Any], str, str | None],
    Awaitable[Mapping[str, Any]],
]


@dataclass(slots=True)
class OpenAIResponsesProvider:
    request_fn: OpenAIRequestFn | None = None
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
            payload = _build_openai_payload(request)
            base_url = request.model.base_url or None
            response = await self._request(payload, api_key, base_url)
            assistant_message = _assistant_from_openai_response(
                request.model,
                response,
            )

            if assistant_message.stop_reason in {"error", "aborted"}:
                stream.push(
                    {
                        "type": "error",
                        "reason": (
                            "aborted"
                            if assistant_message.stop_reason == "aborted"
                            else "error"
                        ),
                        "error": assistant_message,
                    }
                )
                return

            done_reason = _done_reason_for_message(assistant_message)
            stream.push(
                {
                    "type": "done",
                    "reason": done_reason,
                    "message": assistant_message,
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

    async def _request(
        self,
        payload: dict[str, Any],
        api_key: str,
        base_url: str | None,
    ) -> dict[str, Any]:
        if self.request_fn is not None:
            return dict(await self.request_fn(payload, api_key, base_url))
        return await _request_openai_response(payload, api_key, base_url)


def _resolve_api_key(request: PiAIRequest, api_key_env: str) -> str:
    if request.api_key:
        return request.api_key

    api_key = os.getenv(api_key_env)
    if api_key:
        return api_key

    raise RuntimeError(
        "Missing OpenAI API key. Set request.api_key or OPENAI_API_KEY."
    )


def _build_openai_payload(request: PiAIRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model.id,
        "input": _to_openai_input(request.context),
    }

    if request.context.system_prompt:
        payload["instructions"] = request.context.system_prompt

    tools = _to_openai_tools(request.context.tools)
    if tools:
        payload["tools"] = tools

    effort = _map_reasoning_effort(request.reasoning)
    if effort is not None:
        payload["reasoning"] = {"effort": effort}

    if request.session_id:
        payload["metadata"] = {"session_id": request.session_id}

    return payload


def _to_openai_input(context: LlmContext) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for message in context.messages:
        if isinstance(message, UserMessage):
            items.append(_to_openai_user_item(message))
            continue

        if isinstance(message, AssistantMessage):
            items.extend(_to_openai_assistant_items(message))
            continue

        if isinstance(message, ToolResultMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _tool_result_text(message),
                }
            )

    return items


def _to_openai_user_item(message: UserMessage) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": "user", "content": message.content}

    content_items: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            content_items.append({"type": "input_text", "text": block.text})
        elif isinstance(block, ImageContent):
            content_items.append(
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:{block.mime_type};base64,{block.data}"
                    ),
                }
            )

    if not content_items:
        content_items.append({"type": "input_text", "text": ""})

    return {"role": "user", "content": content_items}


def _to_openai_assistant_items(message: AssistantMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    text_chunks = [
        block.text
        for block in message.content
        if isinstance(block, TextContent) and block.text
    ]
    if text_chunks:
        items.append(
            {
                "role": "assistant",
                "content": "\n".join(text_chunks),
            }
        )

    for block in message.content:
        if isinstance(block, ToolCall):
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": json.dumps(
                        block.arguments,
                        separators=(",", ":"),
                    ),
                }
            )

    return items


def _to_openai_tools(tools: Sequence[AgentTool] | None) -> list[dict[str, Any]]:
    if not tools:
        return []

    payload_tools: list[dict[str, Any]] = []
    for tool in tools:
        parameters = _coerce_json_schema(tool.parameters)
        payload_tools.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            }
        )
    return payload_tools


def _coerce_json_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {"type": "object", "properties": {}}

    schema_dict = dict(schema)
    if "type" not in schema_dict:
        schema_dict["type"] = "object"
    return schema_dict


def _assistant_from_openai_response(
    model: Model,
    response: Mapping[str, Any],
) -> AssistantMessage:
    content: list[AssistantContentBlock] = []
    for item in _as_mapping_list(response.get("output")):
        item_type = _as_str(item.get("type"))
        if item_type == "message":
            content.extend(_extract_text_content(item))
            continue

        if item_type == "function_call":
            tool_call = _extract_tool_call(item)
            if tool_call is not None:
                content.append(tool_call)

    error_message = _extract_error_message(response)
    stop_reason = _derive_stop_reason(content, response, error_message)
    if not content:
        content = [TextContent(text="")]

    usage = _extract_usage(response.get("usage"))
    return AssistantMessage(
        content=content,
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=usage,
        stop_reason=stop_reason,
        error_message=error_message,
    )


def _extract_text_content(item: Mapping[str, Any]) -> list[TextContent]:
    blocks: list[TextContent] = []
    for part in _as_mapping_list(item.get("content")):
        part_type = _as_str(part.get("type"))
        if part_type not in {"output_text", "text"}:
            continue

        text = _as_str(part.get("text"))
        if text:
            blocks.append(TextContent(text=text))
    return blocks


def _extract_tool_call(item: Mapping[str, Any]) -> ToolCall | None:
    call_id = _as_str(item.get("call_id")) or _as_str(item.get("id"))
    name = _as_str(item.get("name"))
    if not call_id or not name:
        return None

    arguments = _extract_tool_call_arguments(item.get("arguments"))
    return ToolCall(id=call_id, name=name, arguments=arguments)


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


def _extract_error_message(response: Mapping[str, Any]) -> str | None:
    error = response.get("error")
    if isinstance(error, Mapping):
        message = _as_str(error.get("message"))
        if message:
            return message

    status = _as_str(response.get("status"))
    if status == "failed":
        return "OpenAI response failed."
    if status == "cancelled":
        return "OpenAI response cancelled."
    return None


def _derive_stop_reason(
    content: Sequence[AssistantContentBlock],
    response: Mapping[str, Any],
    error_message: str | None,
) -> StopReason:
    if error_message:
        status = _as_str(response.get("status"))
        if status == "cancelled":
            return "aborted"
        return "error"

    if any(isinstance(block, ToolCall) for block in content):
        return "toolUse"

    status = _as_str(response.get("status"))
    if status == "incomplete":
        return "length"
    return "stop"


def _extract_usage(usage_data: Any) -> Usage:
    if not isinstance(usage_data, Mapping):
        return Usage(cost=UsageCost())

    input_tokens = _as_int(usage_data.get("input_tokens"))
    output_tokens = _as_int(usage_data.get("output_tokens"))
    total_tokens = _as_int(usage_data.get("total_tokens"))

    input_details = usage_data.get("input_tokens_details")
    cache_read = 0
    if isinstance(input_details, Mapping):
        cache_read = _as_int(input_details.get("cached_tokens"))

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=0,
        total_tokens=total_tokens,
        cost=UsageCost(),
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


def _map_reasoning_effort(reasoning: ThinkingLevel | None) -> str | None:
    if reasoning in {None, "off"}:
        return None
    if reasoning in {"minimal", "low"}:
        return "low"
    if reasoning == "medium":
        return "medium"
    return "high"


def _tool_result_text(message: ToolResultMessage) -> str:
    text_chunks = [
        block.text
        for block in message.content
        if isinstance(block, TextContent)
    ]
    return "\n".join(text_chunks).strip()


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


async def _request_openai_response(
    payload: dict[str, Any],
    api_key: str,
    base_url: str | None,
) -> dict[str, Any]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenAI provider requires the `openai` package. "
            "Install it with: uv add openai"
        ) from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    async with AsyncOpenAI(**client_kwargs) as client:
        response = await client.responses.create(**payload)

    if isinstance(response, Mapping):
        return dict(response)

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return cast(dict[str, Any], dict(dumped))

    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, Mapping):
            return cast(dict[str, Any], dict(dumped))

    raise RuntimeError(
        "Unsupported OpenAI response type. "
        f"Expected mapping-like object, got: {type(response).__name__}"
    )
