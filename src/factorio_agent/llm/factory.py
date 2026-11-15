"""Build the configured LLM client(s)."""
from __future__ import annotations

import os

from factorio_agent.config import Config
from factorio_agent.llm.client import AnthropicClient, FallbackClient, LLMClient, LLMUnavailable, OllamaClient


class _Unavailable(LLMClient):
    """Placeholder for a provider that cannot be constructed (e.g. missing API key)."""

    def __init__(self, reason: str):
        self.reason = reason

    def generate(self, messages):
        raise LLMUnavailable(self.reason)


def _make(provider: str, cfg: Config) -> LLMClient:
    if provider == "ollama":
        return OllamaClient(cfg.ollama_url, cfg.model, cfg.num_ctx, cfg.keep_alive,
                            cfg.llm_timeout, cfg.llm_retries, cfg.think, cfg.temperature)
    if provider == "anthropic":
        try:
            return AnthropicClient(os.environ.get("ANTHROPIC_API_KEY", ""), cfg.anthropic_model, cfg.anthropic_url,
                                   cfg.anthropic_max_tokens, cfg.llm_timeout, cfg.llm_retries, cfg.temperature)
        except LLMUnavailable as exc:
            return _Unavailable(str(exc))
    raise ValueError(f"Unknown LLM provider '{provider}' (use 'ollama' or 'anthropic')")


def create_llm(cfg: Config) -> LLMClient:
    primary = _make(cfg.llm_provider, cfg)
    if cfg.llm_fallback and cfg.llm_fallback != cfg.llm_provider:
        return FallbackClient(primary, _make(cfg.llm_fallback, cfg))
    return primary
