"""Shared LLM call schema for both critics."""

from pydantic import BaseModel, Field


class LLMCriticCall(BaseModel):
    verdict: str
    alternative_stocks: list[str] = Field(default_factory=list)
