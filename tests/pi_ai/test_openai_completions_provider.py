from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import pytest

from pi_py.agent_core import (
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    AssistantMessageEvent,
    ImageContent,
    LlmContext,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_py.pi_ai import OpenAICompletionsProvider, PiAIRequest
from pi_py.pi_ai.providers.openai_completions import _chunk_to_mapping, _response_to_mapping


async def _noop_execute(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[AgentToolResult[Any]], None] | None = None,
) -> AgentToolResult[Any]:
    del tool_call_id, params, abort_event, on_update
    return AgentToolResult(content=[TextContent(text="unused")], details={})


def _assistant_text(message: AssistantMessage) -> str:
    return " ".join(
        block.text for block in message.content if isinstance(block, TextContent)
    ).strip()


@pytest.mark.asyncio
async def test_openai_completions_provider_emits_tool_call_response() -> None:
    seen_payloads: list[dict[str, Any]] = []

    async def request_fn(
        payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> Mapping[str, Any]:
        seen_payloads.append(payload)
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Tokyo"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
            },
        }

    provider = OpenAICompletionsProvider(request_fn=request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5-mini", provider="openai", api="openai-completions"),
        context=LlmContext(
            system_prompt="Use tools when needed.",
            messages=[UserMessage(content="Weather in Tokyo?")],
            tools=[
                AgentTool(
                    name="get_weather",
                    label="Get Weather",
                    description="Returns weather for a city",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    execute=_noop_execute,
                )
            ],
        ),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    async for _ in stream:
        pass
    message = await stream.result()

    assert message.stop_reason == "toolUse"
    tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == {"city": "Tokyo"}
    assert message.usage.input == 7

    payload = seen_payloads[0]
    assert payload["model"] == "gpt-5-mini"
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["tools"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_openai_completions_provider_emits_text_response() -> None:
    async def request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> Mapping[str, Any]:
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hello from OpenAI Completions",
                    },
                }
            ]
        }

    provider = OpenAICompletionsProvider(request_fn=request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5-mini", provider="openai", api="openai-completions"),
        context=LlmContext(messages=[UserMessage(content="Hello")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    async for _ in stream:
        pass
    message = await stream.result()

    assert message.stop_reason == "stop"
    assert _assistant_text(message) == "Hello from OpenAI Completions"


@pytest.mark.asyncio
async def test_openai_completions_provider_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PI_PY_TEST_OPENAI_KEY", raising=False)

    async def should_not_run(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> Mapping[str, Any]:
        return await _raise_if_called()

    provider = OpenAICompletionsProvider(
        request_fn=should_not_run,
        api_key_env="PI_PY_TEST_OPENAI_KEY",
    )
    request = PiAIRequest(
        model=Model(id="gpt-5-mini", provider="openai", api="openai-completions"),
        context=LlmContext(messages=[UserMessage(content="Hello")]),
    )

    stream = await provider.stream(request)
    events: list[AssistantMessageEvent] = []
    async for event in stream:
        events.append(event)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    error_message = events[0]["error"].error_message or ""
    assert "Missing OpenAI API key" in error_message


@pytest.mark.asyncio
async def test_openai_completions_provider_sends_tool_result_images() -> None:
    seen_payloads: list[dict[str, Any]] = []

    async def request_fn(
        payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> Mapping[str, Any]:
        seen_payloads.append(payload)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Received image",
                    },
                }
            ]
        }

    provider = OpenAICompletionsProvider(request_fn=request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5-mini", provider="openai", api="openai-completions"),
        context=LlmContext(
            messages=[
                UserMessage(content="Show tool image result"),
                ToolResultMessage(
                    tool_call_id="call_image",
                    tool_name="generate_chart",
                    content=[
                        TextContent(text="Chart generated"),
                        ImageContent(data="aGVsbG8=", mime_type="image/png"),
                    ],
                    is_error=False,
                ),
            ]
        ),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    async for _ in stream:
        pass
    _ = await stream.result()

    messages = seen_payloads[0]["messages"]
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_image"
    assert tool_messages[0]["content"] == "Chart generated"

    image_user_messages = [
        message
        for message in messages
        if message.get("role") == "user"
        and isinstance(message.get("content"), list)
        and any(
            isinstance(part, Mapping) and part.get("type") == "image_url"
            for part in message["content"]
        )
    ]
    assert len(image_user_messages) == 1
    user_content = image_user_messages[0]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_openai_completions_provider_normalizes_cached_usage_tokens() -> None:
    async def request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> Mapping[str, Any]:
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Done"},
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 20},
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        }

    provider = OpenAICompletionsProvider(request_fn=request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5-mini", provider="openai", api="openai-completions"),
        context=LlmContext(messages=[UserMessage(content="hello")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    async for _ in stream:
        pass
    message = await stream.result()

    assert message.usage.input == 100
    assert message.usage.output == 35
    assert message.usage.cache_read == 20
    assert message.usage.total_tokens == 150


@pytest.mark.asyncio
async def test_openai_completions_provider_streams_text_deltas() -> None:
    async def stream_request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        async def _events() -> AsyncIterator[Mapping[str, Any]]:
            yield {
                "choices": [
                    {"delta": {"content": "Hel"}, "finish_reason": None},
                ]
            }
            yield {
                "choices": [
                    {"delta": {"content": "lo"}, "finish_reason": None},
                ]
            }
            yield {
                "choices": [
                    {"delta": {}, "finish_reason": "stop"},
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }

        return _events()

    provider = OpenAICompletionsProvider(stream_request_fn=stream_request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5-mini", provider="openai", api="openai-completions"),
        context=LlmContext(messages=[UserMessage(content="Hello")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    event_types: list[str] = []
    text_deltas: list[str] = []
    async for event in stream:
        event_types.append(event["type"])
        if event["type"] == "text_delta":
            text_deltas.append(event["delta"])
    message = await stream.result()

    assert event_types == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert text_deltas == ["Hel", "lo"]
    assert message.stop_reason == "stop"
    assert _assistant_text(message) == "Hello"


@pytest.mark.asyncio
async def test_openai_completions_provider_streams_tool_call_updates() -> None:
    async def stream_request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        async def _events() -> AsyncIterator[Mapping[str, Any]]:
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call_tokyo",
                                    "index": 0,
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Tok',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call_tokyo",
                                    "index": 0,
                                    "type": "function",
                                    "function": {"arguments": 'yo"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {"delta": {}, "finish_reason": "tool_calls"},
                ]
            }

        return _events()

    provider = OpenAICompletionsProvider(stream_request_fn=stream_request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5-mini", provider="openai", api="openai-completions"),
        context=LlmContext(messages=[UserMessage(content="Weather in Tokyo?")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    event_types: list[str] = []
    async for event in stream:
        event_types.append(event["type"])
    message = await stream.result()

    tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
    assert event_types == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    assert message.stop_reason == "toolUse"
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == {"city": "Tokyo"}


@pytest.mark.asyncio
async def test_openai_completions_provider_streams_reasoning_events() -> None:
    async def stream_request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        async def _events() -> AsyncIterator[Mapping[str, Any]]:
            yield {
                "choices": [
                    {
                        "delta": {"reasoning_content": "Reason one\n"},
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {"reasoning_content": "Reason two"},
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {"delta": {"content": "Final answer"}, "finish_reason": None},
                ]
            }
            yield {
                "choices": [
                    {"delta": {}, "finish_reason": "stop"},
                ]
            }

        return _events()

    provider = OpenAICompletionsProvider(stream_request_fn=stream_request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5-mini", provider="openai", api="openai-completions"),
        context=LlmContext(messages=[UserMessage(content="Explain briefly")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    event_types: list[str] = []
    async for event in stream:
        event_types.append(event["type"])
    message = await stream.result()

    thinking_blocks = [
        block for block in message.content if isinstance(block, ThinkingContent)
    ]
    assert "thinking_start" in event_types
    assert "thinking_end" in event_types
    assert message.stop_reason == "stop"
    assert len(thinking_blocks) == 1
    assert "Reason one" in thinking_blocks[0].thinking
    assert "Reason two" in thinking_blocks[0].thinking


def test_chunk_to_mapping_uses_warnings_false_for_model_dump() -> None:
    class FakeChunk:
        def model_dump(self, *, warnings: bool = True) -> Mapping[str, Any]:
            assert warnings is False
            return {"choices": []}

    mapped = _chunk_to_mapping(FakeChunk())
    assert mapped["choices"] == []


def test_response_to_mapping_uses_warnings_false_for_model_dump() -> None:
    class FakeResponse:
        def model_dump(self, *, warnings: bool = True) -> Mapping[str, Any]:
            assert warnings is False
            return {"choices": []}

    mapped = _response_to_mapping(FakeResponse())
    assert mapped["choices"] == []


async def _raise_if_called() -> Mapping[str, Any]:
    raise AssertionError("request_fn should not be called")
