from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from typing import Any

from pi_py.agent_core import (
    Agent,
    AgentEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    Model,
    TextContent,
)
from pi_py.pi_ai import create_agent_stream_fn, create_default_registry


def extract_assistant_text(message: AssistantMessage) -> str:
    return " ".join(
        block.text for block in message.content if isinstance(block, TextContent)
    ).strip()


async def get_weather(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[AgentToolResult[Any]], None] | None = None,
) -> AgentToolResult[Any]:
    del tool_call_id, abort_event
    city = str(params.get("city", "Unknown"))
    result = AgentToolResult(
        content=[TextContent(text=f"Sunny, 22C in {city}")],
        details={"city": city},
    )
    if on_update is not None:
        on_update(result)
    return result


def on_agent_event(event: AgentEvent) -> None:
    event_type = event["type"]

    if event_type == "tool_execution_start":
        print(f"[tool:start] {event['tool_name']} args={event['args']}")
        return

    if event_type == "tool_execution_end":
        print(f"[tool:end] {event['tool_name']} error={event['is_error']}")
        return

    if event_type == "message_end":
        message = event["message"]
        if isinstance(message, AssistantMessage):
            text = extract_assistant_text(message)
            if text:
                print(f"[assistant] {text}")


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it and run `uv run python examples/agent_e2e.py`."
        )

    registry = create_default_registry()

    agent = Agent(
        stream_fn=create_agent_stream_fn(registry),
        session_id="agent-e2e-demo",
    )
    agent.set_model(Model(id="gpt-5-mini", provider="openai", api="openai"))
    agent.set_system_prompt(
        "You are a concise assistant. Use get_weather when users ask weather questions."
    )
    agent.set_tools(
        [
            AgentTool(
                name="get_weather",
                label="Get Weather",
                description="Returns a weather string for a city.",
                execute=get_weather,
            )
        ]
    )
    agent.subscribe(on_agent_event)

    await agent.prompt("What's the weather in San Francisco?")

    final_message = agent.state.messages[-1]
    if isinstance(final_message, AssistantMessage):
        print(f"\nFinal answer: {extract_assistant_text(final_message)}")


if __name__ == "__main__":
    asyncio.run(main())
