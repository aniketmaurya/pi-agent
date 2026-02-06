# pi-py Roadmap (Python >= 3.11)

This project aims to provide an open source Python library equivalent in spirit to Pi (JS/TS), so developers can build agentic apps, SDKs, CLIs, and integrations on top of a composable core.

## Guiding Principles

- Layered architecture mirroring Pi:
  - `pi_ai`-equivalent abstraction layer for providers/models/streams
  - `pi_agent_core`-equivalent runtime for turns/tool execution/state/events
  - `pi_coding_agent`-equivalent SDK for sessions, tools, resources, and orchestration
- Provider-agnostic, event-driven, extensible by default.
- Strong typing, deterministic tests, and strict compatibility-oriented semantics.

## Immediate Work Plan

### Phase 0: Project Bootstrap (completed)

- Initialize project with `uv`.
- Set Python baseline to `>=3.11`.
- Add dev tooling (`pytest`, `pytest-asyncio`, `ruff`, `mypy`).
- Establish `src/` package layout and test scaffolding.

### Phase 1: Core Agent Runtime (initial slice completed)

Deliver a first usable runtime package with:

1. Core domain types:
   - Messages (`user`, `assistant`, `toolResult`) and content blocks (`text`, `thinking`, `image`, `toolCall`).
   - Tool protocol and tool result/update contract.
2. Event stream primitives:
   - Async iterable stream utility with terminal result extraction.
3. Agent loop:
   - Turn lifecycle orchestration.
   - Tool execution + tool result message injection.
   - Steering/follow-up queue semantics.
4. Agent class:
   - Stateful wrapper around loop.
   - Subscribe/emit lifecycle events.
   - Prompting, continue, abort, wait-for-idle.
5. Test coverage:
   - Event ordering.
   - Tool execution behavior.
   - Steering/follow-up behavior.
   - Error/abort and continue guards.

Current status:

- `uv`-based package scaffold is set up with strict lint/type/test tooling.
- The initial `agent_core` runtime is implemented and validated.
- Remaining Phase 1 work can focus on API polish, richer event typing, and docs/examples.

## Near-Term Plan

### Phase 2: `pi-ai` Python Layer

- Provider/API registry.
- Unified `stream`/`complete` and `stream_simple`/`complete_simple` APIs.
- Initial providers:
  - OpenAI Responses
  - OpenAI Completions-compatible
  - Anthropic
- Tool argument validation and partial JSON parsing for streamed tool calls.
- Model catalog + cost/token accounting foundation.

Current status:

- Initial Phase 2 slice implemented:
  - `pi_ai` provider contract + registry
  - unified runtime APIs (`stream`, `complete`, `stream_simple`, `complete_simple`)
  - Agent adapter (`create_agent_stream_fn`)
  - mock provider for deterministic local development
  - initial OpenAI Responses provider implementation
  - OpenAI Completions-compatible provider implementation
  - OpenAI streaming event parsing for text/tool-call/reasoning deltas
  - End-to-end `Agent` example added and tested.

### Phase 3: SDK Session Layer

- `create_agent_session` and `AgentSession` abstraction.
- JSONL session persistence with tree semantics (`id`/`parentId`).
- Settings/auth/model registry and runtime resource loading.
- Built-in coding tools (`read`, `write`, `edit`, `bash`; then `grep`, `find`, `ls`).

## Future Work

### Phase 4: Reliability + Compatibility Hardening

- Auto-compaction + branch summarization.
- Retry/backoff policies.
- Context transformation and pruning hooks.
- Cross-provider handoff normalization.

### Phase 5: Ecosystem Features

- Extensions/skills/plugin APIs.
- CLI and optional TUI layer.
- Richer provider matrix (Google, Vertex, Bedrock, Azure, OAuth-backed providers).
- Documentation, examples, and versioned migration notes.

## Open Source Quality Bar

- Clear public API boundaries and semantic versioning.
- Compatibility-focused integration tests.
- High-signal docs with architecture and extension guides.
- Contributor-friendly project structure and CI checks.
