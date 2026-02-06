# pi-agent

Python tools for building AI agents and managing LLM deployments.

This repository is an open-source Python reimplementation of core ideas from Pi (JS/TS), starting with a provider-agnostic agent runtime that is composable and SDK-friendly.

## Status

Phase 1 and an initial Phase 2 slice are implemented:

- Core typed domain model for messages, content blocks, tools, usage, model config
- Async event stream primitive with terminal-result handling
- Agent loop runtime with turn/tool orchestration and steering/follow-up queues
- Stateful `Agent` wrapper with prompt/continue/abort/wait APIs
- Provider abstraction (`pi_ai`) with registry + runtime adapter
- Providers:
  - mock provider
  - OpenAI Responses provider (with streaming deltas)
  - OpenAI Completions-compatible provider (with streaming deltas)
- Strict checks and tests (`ruff`, `mypy`, `pytest`)

Roadmap details are tracked in [`PLAN.md`](./PLAN.md).

## Requirements

- Python `>=3.11`
- [`uv`](https://docs.astral.sh/uv/)

## Install

```bash
pip install pi-agent
```

```python
from pi_agent.agent_core import Agent, Model
```

## Development

```bash
uv sync
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

## Package layout

```text
src/pi_agent/agent_core/
  types.py         # domain model + event types + runtime protocols
  event_stream.py  # generic async stream + assistant stream specialization
  agent_loop.py    # turn execution + tool execution loop
  agent.py         # high-level agent state wrapper

src/pi_agent/pi_ai/
  types.py         # provider request + provider protocol
  registry.py      # provider registry + defaults
  runtime.py       # stream/complete APIs + Agent adapter
  providers/mock.py
  providers/openai.py
  providers/openai_completions.py
```

## End-to-end example

Run:

```bash
uv sync --extra openai
export OPENAI_API_KEY=your_key_here
uv run python examples/agent_e2e.py
```

This example uses OpenAI (`gpt-5-mini`), routes calls through `pi_ai`, invokes a tool, and prints the final assistant response.

## Streaming example (OpenAI)

Run:

```bash
uv sync --extra openai
export OPENAI_API_KEY=your_key_here
uv run python examples/openai_streaming.py
```

This example prints streaming delta events (`text_delta`, and tool-call events when present) from the OpenAI provider.

## Build and publish

```bash
uv build
uvx twine check dist/*
uv publish
```

## Releases

Tag-based releases use `.github/workflows/release.yml` and expect:

- `pyproject.toml` version matches the tag (`vX.Y.Z`).
- A matching `CHANGELOG.md` section exists (`## [X.Y.Z] - YYYY-MM-DD`).
