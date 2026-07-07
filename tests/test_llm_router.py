"""Mocks the network boundary (_call) to test fallback/retry orchestration
offline — no real API key needed for these. See llm_router_live smoke test
(scripts/) once a key is set, for an actual network call.
"""

import pytest
from pydantic import BaseModel

from core.llm_router import LLMRouterError, _extract_json, complete

ALL_PROVIDER_VARS = [
    "GROQ_API_KEY", "GROQ_MODEL",
    "OLLAMA_REMOTE_URL", "OLLAMA_REMOTE_MODEL",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
    "OLLAMA_LOCAL_URL", "OLLAMA_LOCAL_MODEL",
]


class _DummySchema(BaseModel):
    value: str


def _clear_all_providers(monkeypatch):
    for var in ALL_PROVIDER_VARS:
        monkeypatch.delenv(var, raising=False)


def test_extract_json_strips_markdown_fences():
    assert _extract_json('```json\n{"value": "ok"}\n```') == '{"value": "ok"}'
    assert _extract_json('{"value": "ok"}') == '{"value": "ok"}'


def test_no_providers_configured_raises(monkeypatch):
    _clear_all_providers(monkeypatch)
    with pytest.raises(LLMRouterError):
        complete("sys", "user", _DummySchema)


def test_complete_returns_on_first_provider_success(monkeypatch):
    _clear_all_providers(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "fake-model")

    monkeypatch.setattr("core.llm_router._call", lambda provider, system, user, schema: '{"value": "ok"}')
    result = complete("sys", "user", _DummySchema)
    assert result.value == "ok"


def test_complete_falls_back_to_second_provider(monkeypatch):
    _clear_all_providers(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "fake-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-2")
    monkeypatch.setenv("OPENROUTER_MODEL", "fake-model-2")

    calls = []

    def fake_call(provider, system, user, schema):
        calls.append(provider.name)
        if provider.name == "groq":
            raise RuntimeError("rate limited")
        return '{"value": "ok"}'

    monkeypatch.setattr("core.llm_router._call", fake_call)
    result = complete("sys", "user", _DummySchema, max_retries_per_provider=0)
    assert result.value == "ok"
    assert calls == ["groq", "openrouter"]


def test_complete_retries_before_falling_back(monkeypatch):
    _clear_all_providers(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "fake-model")

    attempts = {"n": 0}

    def fake_call(provider, system, user, schema):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient error")
        return '{"value": "ok"}'

    monkeypatch.setattr("core.llm_router._call", fake_call)
    result = complete("sys", "user", _DummySchema, max_retries_per_provider=1)
    assert result.value == "ok"
    assert attempts["n"] == 2


def test_complete_raises_after_exhausting_everything(monkeypatch):
    _clear_all_providers(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("GROQ_MODEL", "fake-model")

    def always_fails(provider, system, user, schema):
        raise RuntimeError("always fails")

    monkeypatch.setattr("core.llm_router._call", always_fails)
    with pytest.raises(LLMRouterError, match="groq"):
        complete("sys", "user", _DummySchema, max_retries_per_provider=0)
