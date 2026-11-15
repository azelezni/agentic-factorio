import json
import unittest

from helpers import StubServer
from factorio_agent.llm.client import (AnthropicClient, FallbackClient, LLMResponseError, LLMUnavailable,
                                       OllamaClient, strip_thinking)
from factorio_agent.llm.factory import create_llm
from factorio_agent.config import Config
from helpers import FakeLLM


def reply(content):
    return 200, {"message": {"role": "assistant", "content": content}}


class OllamaClientTests(unittest.TestCase):
    def test_generate_sends_expected_request_and_strips_thinking(self):
        srv = StubServer(lambda m, p, b: reply("<think>hmm</think>Hello there"))
        self.addCleanup(srv.close)
        c = OllamaClient(srv.url, "qwen3:8b", num_ctx=8192, keep_alive="5m")
        self.assertEqual(c.generate([{"role": "user", "content": "hi"}]), "Hello there")
        method, path, body = srv.requests[0]
        body = json.loads(body)
        self.assertEqual((method, path), ("POST", "/api/chat"))
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertFalse(body["stream"])
        self.assertEqual(body["options"]["num_ctx"], 8192)
        self.assertEqual(body["keep_alive"], "5m")

    def test_retries_malformed_response_then_succeeds(self):
        answers = iter([reply(""), reply("ok")])
        srv = StubServer(lambda m, p, b: next(answers))
        self.addCleanup(srv.close)
        self.assertEqual(OllamaClient(srv.url, retries=1).generate([]), "ok")
        self.assertEqual(len(srv.requests), 2)

    def test_persistent_malformed_response_is_controlled_error(self):
        srv = StubServer(lambda m, p, b: (200, b"not json"))
        self.addCleanup(srv.close)
        with self.assertRaises(LLMResponseError):
            OllamaClient(srv.url, retries=1).generate([])
        self.assertEqual(len(srv.requests), 2)

    def test_unreachable_is_unavailable_and_not_retried(self):
        with self.assertRaises(LLMUnavailable):
            OllamaClient("http://127.0.0.1:9", timeout=2).generate([])

    def test_missing_model_gives_pull_hint(self):
        srv = StubServer(lambda m, p, b: (404, {"error": "model not found"}))
        self.addCleanup(srv.close)
        with self.assertRaisesRegex(LLMUnavailable, "ollama pull"):
            OllamaClient(srv.url, "nope").generate([])

    def test_timeout_is_unavailable(self):
        import time
        srv = StubServer(lambda m, p, b: (time.sleep(1.5), reply("late"))[1])
        self.addCleanup(srv.close)
        with self.assertRaisesRegex(LLMUnavailable, "timed out"):
            OllamaClient(srv.url, timeout=0.3).generate([])

    def test_running_models_reports_vram(self):
        srv = StubServer(lambda m, p, b: (200, {"models": [{"name": "qwen3:8b", "size": 100, "size_vram": 100}]}))
        self.addCleanup(srv.close)
        self.assertEqual(OllamaClient(srv.url).running_models()[0]["size_vram"], 100)

    def test_strip_thinking_handles_unterminated_block(self):
        self.assertEqual(strip_thinking("answer<think>never closed"), "answer")


class AnthropicClientTests(unittest.TestCase):
    def test_request_shape_and_text_extraction(self):
        srv = StubServer(lambda m, p, b: (200, {"content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]}))
        self.addCleanup(srv.close)
        c = AnthropicClient("sk-test", "claude-sonnet-5-5", srv.url, max_tokens=500)
        out = c.generate([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])
        self.assertEqual(out, "Hello world")
        method, path, body = srv.requests[0]
        body = json.loads(body)
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(body["system"], "be brief")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])   # system moved out of messages
        self.assertEqual((body["model"], body["max_tokens"]), ("claude-sonnet-5-5", 500))

    def test_auth_header_and_version_sent(self):
        seen = {}
        srv = StubServer(lambda m, p, b: (200, {"content": [{"type": "text", "text": "ok"}]}))
        self.addCleanup(srv.close)
        import urllib.request as ur
        orig = ur.urlopen
        def spy(req, timeout=None):
            seen.update({k.lower(): v for k, v in req.header_items()})
            return orig(req, timeout=timeout)
        ur.urlopen = spy
        self.addCleanup(setattr, ur, "urlopen", orig)
        AnthropicClient("sk-test", base_url=srv.url).generate([{"role": "user", "content": "x"}])
        self.assertEqual(seen["x-api-key"], "sk-test")
        self.assertEqual(seen["anthropic-version"], "2023-06-01")

    def test_missing_key_is_unavailable(self):
        with self.assertRaisesRegex(LLMUnavailable, "ANTHROPIC_API_KEY"):
            AnthropicClient("")

    def test_http_errors_are_controlled(self):
        srv = StubServer(lambda m, p, b: (401, {"error": {"message": "invalid x-api-key"}}))
        self.addCleanup(srv.close)
        with self.assertRaisesRegex(LLMUnavailable, "check ANTHROPIC_API_KEY"):
            AnthropicClient("bad", base_url=srv.url).generate([{"role": "user", "content": "x"}])

    def test_empty_content_retried_then_error(self):
        srv = StubServer(lambda m, p, b: (200, {"content": []}))
        self.addCleanup(srv.close)
        with self.assertRaises(LLMResponseError):
            AnthropicClient("k", base_url=srv.url, retries=1).generate([{"role": "user", "content": "x"}])
        self.assertEqual(len(srv.requests), 2)


class ProviderSelectionTests(unittest.TestCase):
    def test_fallback_used_only_when_primary_fails(self):
        ok, bad = FakeLLM("secondary"), FakeLLM(fail=True)
        self.assertEqual(FallbackClient(FakeLLM("primary"), ok).generate([]), "primary")
        self.assertEqual(FallbackClient(bad, ok).generate([]), "secondary")
        with self.assertRaisesRegex(LLMUnavailable, "fallback failed"):
            FallbackClient(bad, FakeLLM(fail=True)).generate([])

    def test_factory(self):
        import os
        from unittest import mock
        self.assertIsInstance(create_llm(Config()), OllamaClient)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}):
            self.assertIsInstance(create_llm(Config(llm_provider="anthropic")), AnthropicClient)
            self.assertIsInstance(create_llm(Config(llm_fallback="anthropic")), FallbackClient)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with self.assertRaisesRegex(LLMUnavailable, "ANTHROPIC_API_KEY"):
                create_llm(Config(llm_provider="anthropic")).generate([])   # fails cleanly at use, not at startup
        with self.assertRaises(ValueError):
            create_llm(Config(llm_provider="nope"))


if __name__ == "__main__":
    unittest.main()
