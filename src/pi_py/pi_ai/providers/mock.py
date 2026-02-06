from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from ...agent_core.event_stream import AssistantMessageEventStream
from ...agent_core.types import (
    AssistantContentBlock,
    AssistantMessage,
    AssistantStream,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from ..types import PiAIRequest

ActionableMessage = UserMessage | ToolResultMessage


@dataclass(slots=True)
class MockProvider:
    weather_tool_name: str = "get_weather"
    _tool_call_counter: int = 0

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
                    "error": self._assistant_message(
                        request=request,
                        content=[TextContent(text="")],
                        stop_reason="aborted",
                        error_message="Request aborted",
                    ),
                }
            )
            return

        assistant_message = self._build_assistant_message(request)
        done_reason: Literal["stop", "toolUse"] = (
            "toolUse" if assistant_message.stop_reason == "toolUse" else "stop"
        )
        stream.push(
            {
                "type": "done",
                "reason": done_reason,
                "message": assistant_message,
            }
        )

    def _build_assistant_message(self, request: PiAIRequest) -> AssistantMessage:
        latest_actionable_message = _find_latest_actionable_message(request)

        if isinstance(latest_actionable_message, ToolResultMessage):
            if latest_actionable_message.tool_name == self.weather_tool_name:
                result_text = _extract_tool_result_text(latest_actionable_message)
                return self._assistant_message(
                    request=request,
                    content=[TextContent(text=f"Weather update: {result_text}")],
                    stop_reason="stop",
                )

            result_text = _extract_tool_result_text(latest_actionable_message)
            return self._assistant_message(
                request=request,
                content=[TextContent(text=f"Tool result: {result_text}")],
                stop_reason="stop",
            )

        latest_user_text = _extract_user_text(latest_actionable_message)
        if "weather" in latest_user_text.lower():
            city = _extract_city_from_prompt(latest_user_text)
            tool_call = ToolCall(
                id=self._next_tool_call_id(),
                name=self.weather_tool_name,
                arguments={"city": city},
            )
            return self._assistant_message(
                request=request,
                content=[tool_call],
                stop_reason="toolUse",
            )

        return self._assistant_message(
            request=request,
            content=[TextContent(text=f"Echo: {latest_user_text or '...'}")],
            stop_reason="stop",
        )

    def _assistant_message(
        self,
        *,
        request: PiAIRequest,
        content: list[AssistantContentBlock],
        stop_reason: StopReason,
        error_message: str | None = None,
    ) -> AssistantMessage:
        return AssistantMessage(
            content=content,
            api=request.model.api,
            provider=request.model.provider,
            model=request.model.id,
            usage=Usage(),
            stop_reason=stop_reason,
            error_message=error_message,
        )

    def _next_tool_call_id(self) -> str:
        self._tool_call_counter += 1
        return f"tool-call-{self._tool_call_counter}"


def _find_latest_actionable_message(request: PiAIRequest) -> ActionableMessage | None:
    for message in reversed(request.context.messages):
        if isinstance(message, ToolResultMessage):
            return message

        if isinstance(message, UserMessage):
            return message
    return None


def _extract_user_text(message: ActionableMessage | None) -> str:
    if not isinstance(message, UserMessage):
        return ""

    if isinstance(message.content, str):
        return message.content

    text_parts = [
        block.text
        for block in message.content
        if isinstance(block, TextContent)
    ]
    return " ".join(text_parts).strip()


def _extract_tool_result_text(message: ToolResultMessage) -> str:
    text_parts = [block.text for block in message.content if isinstance(block, TextContent)]
    return " ".join(text_parts).strip() or "(no text)"


def _extract_city_from_prompt(prompt: str) -> str:
    marker = " in "
    lower_prompt = prompt.lower()
    marker_index = lower_prompt.rfind(marker)
    if marker_index == -1:
        return "San Francisco"

    city = prompt[marker_index + len(marker) :].strip(" ?.!")
    return city.title() if city else "San Francisco"
