from __future__ import annotations

import pytest

from pi_py.agent_core import Model
from pi_py.pi_ai import ProviderRegistry, create_default_registry
from pi_py.pi_ai.providers import MockProvider


def test_registry_prefers_api_over_provider() -> None:
    registry = ProviderRegistry()
    provider_from_api = MockProvider()
    provider_from_provider = MockProvider()
    registry.register("mock-api", provider_from_api)
    registry.register("mock-provider", provider_from_provider)

    model = Model(id="model-1", provider="mock-provider", api="mock-api")

    assert registry.resolve(model) is provider_from_api


def test_registry_raises_for_missing_provider() -> None:
    registry = ProviderRegistry()
    model = Model(id="model-1", provider="missing-provider", api="missing-api")

    with pytest.raises(LookupError, match="No provider registered"):
        registry.resolve(model)


def test_default_registry_has_mock_and_openai() -> None:
    registry = create_default_registry()

    assert registry.get("mock") is not None
    assert registry.get("openai") is not None
    assert registry.get("openai-completions") is not None
