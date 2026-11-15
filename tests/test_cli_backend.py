import contextlib
import io
import os
import unittest
from unittest import mock

from helpers import StubServer
from factorio_agent.cli.commands import Runtime, apply_overrides, backend_command, build_parser, cmd_chat
from factorio_agent.config import Config
from factorio_agent.llm.client import AnthropicClient, FallbackClient, OllamaClient


def parse(*argv):
    return build_parser().parse_args(list(argv))


class BackendSelectionTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test",
                                         "FACTORIO_AGENT_PROMETHEUS_URL": "http://127.0.0.1:9"})
        p.start()
        self.addCleanup(p.stop)

    def test_flags_override_config_in_either_position(self):
        for argv in (("--llm", "anthropic", "status"), ("status", "--llm", "anthropic")):
            cfg = apply_overrides(Config(), parse(*argv))
            self.assertEqual(cfg.llm_provider, "anthropic")

    def test_model_flag_targets_selected_backend(self):
        cfg = apply_overrides(Config(), parse("--llm", "anthropic", "--model", "claude-haiku-5-5", "status"))
        self.assertEqual((cfg.anthropic_model, cfg.model), ("claude-haiku-5-5", "qwen3:8b"))
        cfg = apply_overrides(Config(), parse("--model", "llama3.1:8b", "status"))
        self.assertEqual((cfg.model, cfg.llm_provider), ("llama3.1:8b", "ollama"))

    def test_fallback_flag_and_none(self):
        self.assertEqual(apply_overrides(Config(), parse("--fallback", "anthropic", "chat")).llm_fallback, "anthropic")
        self.assertEqual(apply_overrides(Config(llm_fallback="anthropic"), parse("--fallback", "none", "chat")).llm_fallback, "")

    def test_invalid_backend_from_config_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_overrides(Config(llm_provider="gpt"), parse("status"))

    def test_runtime_switch_updates_llm_and_investigator(self):
        rt = Runtime(Config())
        self.assertIsInstance(rt.llm, OllamaClient)
        rt.investigator = mock.Mock()
        msg = rt.set_backend("anthropic", "claude-haiku-5-5")
        self.assertIn("anthropic (claude-haiku-5-5)", msg)
        self.assertIsInstance(rt.llm, AnthropicClient)
        self.assertIs(rt.investigator.llm, rt.llm)
        with self.assertRaises(ValueError):
            rt.set_backend("nope")
        self.assertIsInstance(rt.llm, AnthropicClient)          # failed switch changes nothing

    def test_backend_command_text(self):
        rt = Runtime(Config())
        self.assertIn("Current backend: ollama (qwen3:8b)", backend_command(rt, ""))
        self.assertIn("Unknown backend", backend_command(rt, " gpt"))
        self.assertIn("Switched to anthropic", backend_command(rt, " ANTHROPIC"))

    def test_chat_can_switch_backend_midway(self):
        ollama = StubServer(lambda m, p, b: (200, {"message": {"content": "from-ollama"}}))
        claude = StubServer(lambda m, p, b: (200, {"content": [{"type": "text", "text": "from-claude"}]}))
        self.addCleanup(ollama.close)
        self.addCleanup(claude.close)
        cfg = Config(ollama_url=ollama.url, anthropic_url=claude.url)
        rt = Runtime(cfg)
        inputs = iter(["hello", "/backend anthropic", "hello again", "/backend", "/quit"])
        buf = io.StringIO()
        with mock.patch("builtins.input", lambda _="": next(inputs)), contextlib.redirect_stdout(buf):
            cmd_chat(rt, None)
        out = buf.getvalue()
        self.assertLess(out.index("from-ollama"), out.index("Switched to anthropic"))
        self.assertLess(out.index("Switched to anthropic"), out.index("from-claude"))
        self.assertIn("Current backend: anthropic", out)
        # conversation history carried over to the new backend
        import json
        sent = json.loads(claude.requests[0][2])["messages"]
        self.assertEqual([m["content"] for m in sent], ["hello", "from-ollama", "hello again"])


if __name__ == "__main__":
    unittest.main()
