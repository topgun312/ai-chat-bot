import asyncio
import warnings

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import ValidationError

from chatbot.client import ChatClient
from chatbot.config import ChatConfig

MODEL_BY_PROVIDER = {
    "ollama": "llama3.2",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}


def make_config(**kwargs) -> ChatConfig:
    provider = kwargs.get("provider", "ollama")
    base = {
        "model": MODEL_BY_PROVIDER.get(provider, "llama3.2"),
        "api_key": "test-anthropic-key",
        "openai_api_key": "test-openai-key",
    }
    base.update(kwargs)
    return ChatConfig(**base)


class TestProviderSelection:
    def test_ollama_uses_local_endpoint(self):
        client = ChatClient(make_config(provider="ollama"))
        assert isinstance(client._client, AsyncOpenAI)
        assert str(client._client.base_url).rstrip("/") == "http://localhost:11434/v1"

    def test_ollama_custom_base_url(self):
        cfg = make_config(provider="ollama", ollama_base_url="http://gpu-box:11434")
        client = ChatClient(cfg)
        assert str(client._client.base_url).rstrip("/") == "http://gpu-box:11434/v1"

    def test_openai_uses_official_endpoint(self):
        client = ChatClient(make_config(provider="openai"))
        assert isinstance(client._client, AsyncOpenAI)
        assert str(client._client.base_url).rstrip("/") == "https://api.openai.com/v1"

    def test_anthropic_uses_anthropic_sdk(self):
        client = ChatClient(make_config(provider="anthropic"))
        assert isinstance(client._client, AsyncAnthropic)

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValidationError):
            make_config(provider="mistral")


class TestProviderKeyValidation:
    def test_anthropic_requires_key(self):
        with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
            ChatConfig(provider="anthropic", api_key="")

    def test_openai_requires_key(self):
        with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
            ChatConfig(provider="openai", openai_api_key="")

    def test_ollama_needs_no_key(self):
        cfg = ChatConfig(provider="ollama", api_key="", openai_api_key="")
        assert cfg.provider == "ollama"


class TestModelProviderMismatch:
    def test_warns_when_model_does_not_match_provider(self):
        with pytest.warns(UserWarning, match="не похожа"):
            ChatConfig(provider="anthropic", model="llama3.2")

    def test_no_warning_for_matching_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ChatConfig(provider="anthropic", model="claude-sonnet-4-6")


class TestStreamDispatch:
    def _run_with_fakes(self, provider: str) -> dict[str, bool]:
        client = ChatClient(make_config(provider=provider))
        called = {"openai": False, "anthropic": False}

        async def fake_openai(*args, **kwargs):
            called["openai"] = True
            yield "x"

        async def fake_anthropic(*args, **kwargs):
            called["anthropic"] = True
            yield "x"

        client._stream_openai = fake_openai  # type: ignore[method-assign]
        client._stream_anthropic = fake_anthropic  # type: ignore[method-assign]

        async def collect():
            return [c async for c in client.stream([], "system")]

        asyncio.run(collect())
        return called

    def test_ollama_routes_to_openai_path(self):
        assert self._run_with_fakes("ollama") == {"openai": True, "anthropic": False}

    def test_openai_routes_to_openai_path(self):
        assert self._run_with_fakes("openai") == {"openai": True, "anthropic": False}

    def test_anthropic_routes_to_anthropic_path(self):
        assert self._run_with_fakes("anthropic") == {"openai": False, "anthropic": True}
