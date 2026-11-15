"""LLM client abstraction and Ollama implementation (design 8.3)."""
from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMUnavailable(LLMError):
    """Ollama is unreachable, timed out, or the model is missing."""


class LLMResponseError(LLMError):
    """Ollama answered, but the response was malformed or empty."""


class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages: list[dict]) -> str:
        """Send chat messages ([{"role","content"}]) and return the reply text."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "<think>" in text:                    # unterminated block
        text = text.split("<think>", 1)[0]
    return text.strip()


class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b",
                 num_ctx: int = 8192, keep_alive: str = "5m", timeout: float = 180.0,
                 retries: int = 1, think: bool = False, temperature: float = 0.2):
        self.base_url = base_url.rstrip("/")
        self.model, self.num_ctx, self.keep_alive = model, num_ctx, keep_alive
        self.timeout, self.retries, self.think, self.temperature = timeout, retries, think, temperature

    # ------------------------------------------------------------------ transport
    def _request(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", "replace").strip()
            if exc.code == 404:
                raise LLMUnavailable(f"Model '{self.model}' not found; run `ollama pull {self.model}`. {detail}") from exc
            raise LLMUnavailable(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise LLMUnavailable(f"Ollama timed out after {timeout or self.timeout:.0f}s") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LLMUnavailable(f"Ollama unreachable at {self.base_url}: {exc}") from exc
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise LLMResponseError(f"Ollama returned invalid JSON: {exc}") from exc

    # ------------------------------------------------------------------ API
    def generate(self, messages: list[dict]) -> str:
        body = {"model": self.model, "messages": messages, "stream": False, "think": self.think,
                "keep_alive": self.keep_alive,
                "options": {"num_ctx": self.num_ctx, "temperature": self.temperature}}
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                payload = self._request("/api/chat", body)
                content = (payload.get("message") or {}).get("content")
                if not isinstance(content, str):
                    raise LLMResponseError("Ollama response has no message content")
                text = strip_thinking(content)
                if not text:
                    raise LLMResponseError("Ollama returned an empty response")
                return text
            except LLMResponseError as exc:      # malformed -> retry; unavailability is not retried
                last = exc
        raise LLMResponseError(f"{last} (after {self.retries + 1} attempts)")

    def is_available(self) -> bool:
        try:
            self._request("/api/tags", timeout=5)
            return True
        except LLMError:
            return False

    def list_models(self) -> list[str]:
        return [m.get("name", "") for m in self._request("/api/tags", timeout=5).get("models", [])]

    def running_models(self) -> list[dict]:
        """Loaded models incl. `size` and `size_vram` - used to verify GPU inference."""
        return self._request("/api/ps", timeout=5).get("models", [])


class AnthropicClient(LLMClient):
    """Claude via the Anthropic Messages API (stdlib HTTP, no SDK needed)."""

    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5", base_url: str = "https://api.anthropic.com",
                 max_tokens: int = 1024, timeout: float = 120.0, retries: int = 1, temperature: float = 0.2):
        if not api_key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.max_tokens, self.timeout, self.retries, self.temperature = max_tokens, timeout, retries, temperature

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + "/v1/messages", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": self.API_VERSION})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", "replace").strip()
            hint = {401: " (check ANTHROPIC_API_KEY)", 429: " (rate limited)", 529: " (API overloaded)"}.get(exc.code, "")
            raise LLMUnavailable(f"Anthropic API HTTP {exc.code}{hint}: {detail}") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise LLMUnavailable(f"Anthropic API timed out after {self.timeout:.0f}s") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LLMUnavailable(f"Anthropic API unreachable: {exc}") from exc
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise LLMResponseError(f"Anthropic API returned invalid JSON: {exc}") from exc

    def generate(self, messages: list[dict]) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]}
        if system:
            body["system"] = system
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                payload = self._post(body)
                blocks = payload.get("content")
                if not isinstance(blocks, list):
                    raise LLMResponseError("Anthropic response has no content blocks")
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                if not text:
                    raise LLMResponseError("Anthropic returned an empty response")
                return text
            except LLMResponseError as exc:
                last = exc
        raise LLMResponseError(f"{last} (after {self.retries + 1} attempts)")


class FallbackClient(LLMClient):
    """Try `primary`; if it is unavailable or fails, try `secondary`."""

    def __init__(self, primary: LLMClient, secondary: LLMClient):
        self.primary, self.secondary = primary, secondary

    def generate(self, messages: list[dict]) -> str:
        try:
            return self.primary.generate(messages)
        except LLMError as first:
            try:
                return self.secondary.generate(messages)
            except LLMError as second:
                raise LLMUnavailable(f"primary LLM failed ({first}); fallback failed ({second})") from second
