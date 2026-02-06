from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from .agent_loop import agent_loop, agent_loop_continue
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    AssistantMessage,
    ConvertToLlmFn,
    GetApiKeyFn,
    ImageContent,
    LlmMessage,
    Model,
    StopReason,
    StreamFn,
    TextContent,
    ThinkingLevel,
    ToolResultMessage,
    TransformContextFn,
    Usage,
    UsageCost,
    UserMessage,
    default_model,
    is_llm_message,
    message_role,
)

Listener = Callable[[AgentEvent], None]
LiteralMode = Literal["all", "one-at-a-time"]


def default_convert_to_llm(messages: list[AgentMessage]) -> list[LlmMessage]:
    llm_messages: list[LlmMessage] = []
    for message in messages:
        if is_llm_message(message) and isinstance(
            message,
            (UserMessage, AssistantMessage, ToolResultMessage),
        ):
            llm_messages.append(message)
    return llm_messages


class Agent:
    def __init__(
        self,
        *,
        initial_state: AgentState | None = None,
        convert_to_llm: ConvertToLlmFn | None = None,
        transform_context: TransformContextFn | None = None,
        steering_mode: LiteralMode = "one-at-a-time",
        follow_up_mode: LiteralMode = "one-at-a-time",
        stream_fn: StreamFn | None = None,
        session_id: str | None = None,
        get_api_key: GetApiKeyFn | None = None,
        thinking_budgets: Mapping[str, int] | None = None,
        max_retry_delay_ms: int | None = None,
    ) -> None:
        self._state = (
            self._clone_state(initial_state)
            if initial_state
            else AgentState(model=default_model())
        )
        self._listeners: set[Listener] = set()

        self._convert_to_llm = convert_to_llm or default_convert_to_llm
        self._transform_context = transform_context
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self.stream_fn = stream_fn

        self._session_id = session_id
        self.get_api_key = get_api_key
        self._thinking_budgets = dict(thinking_budgets) if thinking_budgets else None
        self._max_retry_delay_ms = max_retry_delay_ms

        self._steering_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []
        self._abort_event: asyncio.Event | None = None
        self._running_task: asyncio.Task[None] | None = None

    @staticmethod
    def _clone_state(state: AgentState) -> AgentState:
        return AgentState(
            system_prompt=state.system_prompt,
            model=state.model,
            thinking_level=state.thinking_level,
            tools=list(state.tools),
            messages=list(state.messages),
            is_streaming=state.is_streaming,
            stream_message=state.stream_message,
            pending_tool_calls=set(state.pending_tool_calls),
            error=state.error,
        )

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._session_id = value

    @property
    def thinking_budgets(self) -> dict[str, int] | None:
        return self._thinking_budgets

    @thinking_budgets.setter
    def thinking_budgets(self, value: Mapping[str, int] | None) -> None:
        self._thinking_budgets = dict(value) if value else None

    @property
    def max_retry_delay_ms(self) -> int | None:
        return self._max_retry_delay_ms

    @max_retry_delay_ms.setter
    def max_retry_delay_ms(self, value: int | None) -> None:
        self._max_retry_delay_ms = value

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.add(listener)

        def _unsubscribe() -> None:
            self._listeners.discard(listener)

        return _unsubscribe

    def set_system_prompt(self, prompt: str) -> None:
        self._state.system_prompt = prompt

    def set_model(self, model: Model) -> None:
        self._state.model = model

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._state.thinking_level = level

    def set_tools(self, tools: list[AgentTool]) -> None:
        self._state.tools = list(tools)

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        self._state.messages = list(messages)

    def append_message(self, message: AgentMessage) -> None:
        self._state.messages = [*self._state.messages, message]

    def clear_messages(self) -> None:
        self._state.messages = []

    def reset(self) -> None:
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self._steering_queue = []
        self._follow_up_queue = []

    def steer(self, message: AgentMessage) -> None:
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage) -> None:
        self._follow_up_queue.append(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue = []

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue = []

    def clear_all_queues(self) -> None:
        self._steering_queue = []
        self._follow_up_queue = []

    def set_steering_mode(self, mode: LiteralMode) -> None:
        self._steering_mode = mode

    def get_steering_mode(self) -> LiteralMode:
        return self._steering_mode

    def set_follow_up_mode(self, mode: LiteralMode) -> None:
        self._follow_up_mode = mode

    def get_follow_up_mode(self) -> LiteralMode:
        return self._follow_up_mode

    def abort(self) -> None:
        if self._abort_event is not None:
            self._abort_event.set()

    async def wait_for_idle(self) -> None:
        if self._running_task is not None:
            await self._running_task

    async def prompt(
        self,
        input_value: str | AgentMessage | list[AgentMessage],
        images: Sequence[ImageContent] | None = None,
    ) -> None:
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or "
                "follow_up() to queue messages, or wait for completion."
            )

        messages: list[AgentMessage]
        if isinstance(input_value, list):
            messages = input_value
        elif isinstance(input_value, str):
            content: list[TextContent | ImageContent] = [TextContent(text=input_value)]
            if images:
                content.extend(images)
            messages = [UserMessage(content=content)]
        else:
            messages = [input_value]

        await self._run_loop(messages)

    async def continue_(self) -> None:
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing. Wait for completion before continuing."
            )

        if not self._state.messages:
            raise RuntimeError("No messages to continue from")

        if message_role(self._state.messages[-1]) == "assistant":
            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_loop(None)

    async def _run_loop(self, messages: list[AgentMessage] | None) -> None:
        self._running_task = asyncio.create_task(self._execute(messages))
        try:
            await self._running_task
        finally:
            self._running_task = None

    async def _execute(self, messages: list[AgentMessage] | None) -> None:
        self._abort_event = asyncio.Event()
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

        config = AgentLoopConfig(
            model=self._state.model,
            reasoning=None if self._state.thinking_level == "off" else self._state.thinking_level,
            session_id=self._session_id,
            thinking_budgets=self._thinking_budgets,
            max_retry_delay_ms=self._max_retry_delay_ms,
            convert_to_llm=self._convert_to_llm,
            transform_context=self._transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=self._pull_steering_messages,
            get_follow_up_messages=self._pull_follow_up_messages,
        )

        try:
            if self.stream_fn is None:
                raise RuntimeError("No stream_fn configured")

            stream = (
                agent_loop(messages, context, config, self._abort_event, self.stream_fn)
                if messages is not None
                else agent_loop_continue(context, config, self._abort_event, self.stream_fn)
            )

            async for event in stream:
                event_type = event["type"]

                if event_type in {"message_start", "message_update"}:
                    partial = event["message"]
                    self._state.stream_message = partial

                if event_type == "message_end":
                    self._state.stream_message = None
                    self.append_message(event["message"])

                if event_type == "tool_execution_start":
                    pending = set(self._state.pending_tool_calls)
                    pending.add(event["tool_call_id"])
                    self._state.pending_tool_calls = pending

                if event_type == "tool_execution_end":
                    pending = set(self._state.pending_tool_calls)
                    pending.discard(event["tool_call_id"])
                    self._state.pending_tool_calls = pending

                if event_type == "turn_end":
                    message = event["message"]
                    if isinstance(message, AssistantMessage) and message.error_message:
                        self._state.error = message.error_message

                if event_type == "agent_end":
                    self._state.is_streaming = False
                    self._state.stream_message = None

                self._emit(event)

        except Exception as exc:  # noqa: BLE001
            stop_reason: StopReason = (
                "aborted"
                if self._abort_event is not None and self._abort_event.is_set()
                else "error"
            )
            error_message = AssistantMessage(
                content=[TextContent(text="")],
                api=self._state.model.api,
                provider=self._state.model.provider,
                model=self._state.model.id,
                usage=Usage(
                    input=0,
                    output=0,
                    cache_read=0,
                    cache_write=0,
                    total_tokens=0,
                    cost=UsageCost(),
                ),
                stop_reason=stop_reason,
                error_message=str(exc),
            )
            self.append_message(error_message)
            self._state.error = str(exc)
            self._emit({"type": "agent_end", "messages": [error_message]})

        finally:
            self._state.is_streaming = False
            self._state.stream_message = None
            self._state.pending_tool_calls = set()
            self._abort_event = None

    async def _pull_steering_messages(self) -> list[AgentMessage]:
        if self._steering_mode == "one-at-a-time":
            if self._steering_queue:
                first = self._steering_queue.pop(0)
                return [first]
            return []

        steering = list(self._steering_queue)
        self._steering_queue = []
        return steering

    async def _pull_follow_up_messages(self) -> list[AgentMessage]:
        if self._follow_up_mode == "one-at-a-time":
            if self._follow_up_queue:
                first = self._follow_up_queue.pop(0)
                return [first]
            return []

        follow_ups = list(self._follow_up_queue)
        self._follow_up_queue = []
        return follow_ups

    def _emit(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            listener(event)


__all__ = ["Agent", "default_convert_to_llm"]
