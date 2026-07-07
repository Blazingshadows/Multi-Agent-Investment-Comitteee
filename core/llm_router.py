"""Common interface for every LLM-backed agent/critic: Groq -> remote Ollama
(GPU box) -> OpenRouter -> local Ollama, tried in order with automatic
fallback and one retry per provider on a malformed/invalid response.
Everything downstream calls complete() instead of a provider SDK directly,
so adding/reordering/removing a provider never touches agent code.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()


class LLMRouterError(Exception):
    """Every configured provider failed or returned an unparseable/invalid
    response. Callers should catch this and fall back to a neutral,
    zero-confidence output — never let it crash a trading cycle.
    """


class _Provider:
    def __init__(self, name: str, model: str | None, base_url: str, api_key: str | None = None, requires_key: bool = True):
        self.name = name
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.requires_key = requires_key

    @property
    def available(self) -> bool:
        if not self.model or not self.base_url:
            return False
        if self.requires_key and not self.api_key:
            return False
        return True

    def client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key or "ollama", base_url=self.base_url)


def _load_providers() -> list[_Provider]:
    return [
        _Provider("groq", os.getenv("GROQ_MODEL"), "https://api.groq.com/openai/v1", os.getenv("GROQ_API_KEY")),
        _Provider("ollama-remote", os.getenv("OLLAMA_REMOTE_MODEL"), os.getenv("OLLAMA_REMOTE_URL", ""), requires_key=False),
        _Provider("openrouter", os.getenv("OPENROUTER_MODEL"), "https://openrouter.ai/api/v1", os.getenv("OPENROUTER_API_KEY")),
        _Provider("ollama-local", os.getenv("OLLAMA_LOCAL_MODEL"), os.getenv("OLLAMA_LOCAL_URL", ""), requires_key=False),
    ]


def _extract_json(text: str) -> str:
    """Strips markdown code fences some models wrap JSON in despite
    instructions not to.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _call(provider: _Provider, system: str, user: str, schema: type[BaseModel]) -> str:
    client = provider.client()
    schema_hint = json.dumps(schema.model_json_schema())
    response = client.chat.completions.create(
        model=provider.model,
        messages=[
            {
                "role": "system",
                "content": f"{system}\n\nRespond with ONLY valid JSON matching this schema, no markdown fences, no commentary:\n{schema_hint}",
            },
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def complete(system: str, user: str, schema: type[BaseModel], max_retries_per_provider: int = 1) -> BaseModel:
    """Tries each available provider in order; within a provider, retries
    once on a response that fails JSON parsing or schema validation. Returns
    a validated instance of `schema`. Raises LLMRouterError only if every
    provider and retry is exhausted.
    """
    providers = [p for p in _load_providers() if p.available]
    if not providers:
        raise LLMRouterError(
            "No LLM provider configured — set at least one of GROQ_API_KEY, "
            "OLLAMA_REMOTE_URL, OPENROUTER_API_KEY, OLLAMA_LOCAL_URL in .env"
        )

    errors = []
    for provider in providers:
        for attempt in range(max_retries_per_provider + 1):
            try:
                raw = _call(provider, system, user, schema)
                return schema.model_validate_json(_extract_json(raw))
            except Exception as exc:  # provider/network/parse errors should all trigger fallback, not crash the cycle
                errors.append(f"{provider.name} attempt {attempt + 1}: {exc}")
                continue

    raise LLMRouterError("All providers exhausted: " + " | ".join(errors))
