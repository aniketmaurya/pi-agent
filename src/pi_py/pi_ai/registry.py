from __future__ import annotations

from ..agent_core.types import Model
from .types import Provider


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register(self, key: str, provider: Provider) -> None:
        self._providers[_normalize_key(key)] = provider

    def unregister(self, key: str) -> None:
        self._providers.pop(_normalize_key(key), None)

    def get(self, key: str) -> Provider | None:
        return self._providers.get(_normalize_key(key))

    def resolve(self, model: Model) -> Provider:
        api_provider = self.get(model.api)
        if api_provider is not None:
            return api_provider

        provider = self.get(model.provider)
        if provider is not None:
            return provider

        raise LookupError(
            "No provider registered for model "
            f"api='{model.api}' or provider='{model.provider}'."
        )

    def keys(self) -> list[str]:
        return sorted(self._providers.keys())


def create_default_registry() -> ProviderRegistry:
    from .providers import (
        MockProvider,
        OpenAICompletionsProvider,
        OpenAIResponsesProvider,
    )

    registry = ProviderRegistry()
    mock_provider = MockProvider()
    openai_provider = OpenAIResponsesProvider()
    openai_completions_provider = OpenAICompletionsProvider()
    registry.register("mock", mock_provider)
    registry.register("mock-api", mock_provider)
    registry.register("mock-provider", mock_provider)
    registry.register("openai", openai_provider)
    registry.register("openai-responses", openai_provider)
    registry.register("openai-completions", openai_completions_provider)
    return registry


def _normalize_key(key: str) -> str:
    return key.strip().lower()
