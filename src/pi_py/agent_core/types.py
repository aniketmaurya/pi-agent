from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(slots=True)
class UsageCost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: UsageCost = field(default_factory=UsageCost)


@dataclass(slots=True)
class TextContent:
    text: str
    type: Literal["text"] = "text"
    text_signature: str | None = None


@dataclass(slots=True)
class ThinkingContent:
    thinking: str
    type: Literal["thinking"] = "thinking"
    thinking_signature: str | None = None


@dataclass(slots=True)
class ImageContent:
    data: str
    mime_type: str
    type: Literal["image"] = "image"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["toolCall"] = "toolCall"
    thought_signature: str | None = None


UserContentBlock: TypeAlias = TextContent | ImageContent
AssistantContentBlock: TypeAlias = TextContent | ThinkingContent | ToolCall
ToolResultContentBlock: TypeAlias = TextContent | ImageContent


@dataclass(slots=True)
class UserMessage:
    content: str | list[UserContentBlock]
    timestamp: int = field(default_factory=now_ms)
    role: Literal["user"] = "user"


@dataclass(slots=True)
class AssistantMessage:
    content: list[AssistantContentBlock]
    api: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = field(default_factory=now_ms)
    role: Literal["assistant"] = "assistant"


@dataclass(slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContentBlock]
    is_error: bool
    details: Any = None
    timestamp: int = field(default_factory=now_ms)
    role: Literal["toolResult"] = "toolResult"


@runtime_checkable
class RoleCarrier(Protocol):
    role: str


CustomAgentMessage: TypeAlias = Mapping[str, Any] | RoleCarrier
AgentMessage: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage | CustomAgentMessage
LlmMessage: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage


@dataclass(slots=True, frozen=True)
class Model:
    id: str
    provider: str
    api: str = "mock"
    base_url: str = ""
    reasoning: bool = False


def default_model() -> Model:
    return Model(id="mock-model", provider="mock", api="mock")


TDetails = TypeVar("TDetails")


@dataclass(slots=True)
class AgentToolResult(Generic[TDetails]):
    content: list[ToolResultContentBlock]
    details: TDetails


AgentToolUpdateCallback: TypeAlias = Callable[[AgentToolResult[Any]], None]


class ToolExecuteFn(Protocol):
    async def __call__(
        self,
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult[Any]: ...


@dataclass(slots=True)
class AgentTool:
    name: str
    label: str
    description: str
    execute: ToolExecuteFn
    parameters: Mapping[str, Any] | None = None


@dataclass(slots=True)
class AgentState:
    system_prompt: str = ""
    model: Model = field(default_factory=default_model)
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None


@dataclass(slots=True)
class AgentContext:
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] | None = None


@dataclass(slots=True)
class LlmContext:
    messages: list[LlmMessage]
    system_prompt: str | None = None
    tools: list[AgentTool] | None = None


@dataclass(slots=True)
class AgentLoopConfig:
    model: Model
    convert_to_llm: ConvertToLlmFn
    transform_context: TransformContextFn | None = None
    get_api_key: GetApiKeyFn | None = None
    get_steering_messages: GetMessagesFn | None = None
    get_follow_up_messages: GetMessagesFn | None = None
    reasoning: ThinkingLevel | None = None
    api_key: str | None = None
    session_id: str | None = None
    thinking_budgets: Mapping[str, int] | None = None
    max_retry_delay_ms: int | None = None


ConvertToLlmFn: TypeAlias = Callable[
    [list[AgentMessage]],
    list[LlmMessage] | Awaitable[list[LlmMessage]],
]
TransformContextFn: TypeAlias = Callable[
    [list[AgentMessage], asyncio.Event | None],
    Awaitable[list[AgentMessage]],
]
GetApiKeyFn: TypeAlias = Callable[[str], str | None | Awaitable[str | None]]
GetMessagesFn: TypeAlias = Callable[
    [],
    list[AgentMessage] | Awaitable[list[AgentMessage]],
]


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


class StartEvent(TypedDict):
    type: Literal["start"]
    partial: AssistantMessage


class TextStartEvent(TypedDict):
    type: Literal["text_start"]
    content_index: int
    partial: AssistantMessage


class TextDeltaEvent(TypedDict):
    type: Literal["text_delta"]
    content_index: int
    delta: str
    partial: AssistantMessage


class TextEndEvent(TypedDict):
    type: Literal["text_end"]
    content_index: int
    content: str
    partial: AssistantMessage


class ThinkingStartEvent(TypedDict):
    type: Literal["thinking_start"]
    content_index: int
    partial: AssistantMessage


class ThinkingDeltaEvent(TypedDict):
    type: Literal["thinking_delta"]
    content_index: int
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(TypedDict):
    type: Literal["thinking_end"]
    content_index: int
    content: str
    partial: AssistantMessage


class ToolCallStartEvent(TypedDict):
    type: Literal["toolcall_start"]
    content_index: int
    partial: AssistantMessage


class ToolCallDeltaEvent(TypedDict):
    type: Literal["toolcall_delta"]
    content_index: int
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(TypedDict):
    type: Literal["toolcall_end"]
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


class DoneEvent(TypedDict):
    type: Literal["done"]
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage


class ErrorEvent(TypedDict):
    type: Literal["error"]
    reason: Literal["error", "aborted"]
    error: AssistantMessage


AssistantMessageEvent: TypeAlias = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)


class AssistantStream(Protocol):
    def __aiter__(self) -> AsyncIterator[AssistantMessageEvent]: ...

    async def result(self) -> AssistantMessage: ...


StreamFn: TypeAlias = Callable[
    [Model, LlmContext, AgentLoopConfig, asyncio.Event | None],
    AssistantStream | Awaitable[AssistantStream],
]


# ---------------------------------------------------------------------------
# Agent lifecycle events
# ---------------------------------------------------------------------------


AgentEvent: TypeAlias = dict[str, Any]


def message_role(message: AgentMessage) -> str:
    if isinstance(message, Mapping):
        role = message.get("role")
        if isinstance(role, str):
            return role
        raise ValueError("Mapping-based message missing string role")

    role = getattr(message, "role", None)
    if isinstance(role, str):
        return role
    raise ValueError("Message missing role")


def is_llm_message(message: AgentMessage) -> bool:
    return message_role(message) in {"user", "assistant", "toolResult"}


def assistant_tool_calls(message: AssistantMessage) -> list[ToolCall]:
    return [block for block in message.content if isinstance(block, ToolCall)]
