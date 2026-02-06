# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-02-06

### Added

- Core `agent_core` runtime with typed messages, content blocks, usage model, and stream event contracts.
- Stateful `Agent` orchestration with prompt/continue/abort/wait APIs.
- Agent loop support for tool execution, tool-result wiring, steering messages, and follow-up messages.
- `pi_ai` provider abstraction with provider registry and runtime APIs (`stream`, `complete`, and simple variants).
- `OpenAIResponsesProvider` with streaming text/tool-call/reasoning parsing and normalized usage extraction.
- `OpenAICompletionsProvider` with streaming text/tool-call/reasoning parsing and stop-reason mapping.
- Tool-result image compatibility handling for OpenAI providers.
- End-to-end examples for `Agent` and OpenAI streaming (`gpt-5-mini`).
- CI workflow for tests and release workflow for tag-based publishing.

### Changed

- Distribution/package identity changed from `pi-py`/`pi_py` to `pi-agent`/`pi_agent`.
- Project metadata and README updated for PyPI publishing and installation.
