"""Acceptance test: stub Prometheus + stub Ollama driven through the real CLI and HTTP clients."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse

from helpers import StubServer
from factorio_agent.cli.commands import main

# (current, short baseline, long baseline, last-10m) items/min; simulated copper shortage.
PROD = {
    "electronic-circuit": (69, 120, 120, 69), "copper-cable": (110, 200, 200, 110),
    "copper-plate": (100, 204, 204, 100), "copper-ore": (210, 215, 215, 210),
    "iron-plate": (300, 300, 300, 300),
}
CONS = {"copper-plate": (200, 205, 205, 200), "copper-cable": (220, 230, 230, 220), "iron-plate": (250, 250, 250, 250)}


def ok(data):
    return 200, {"status": "success", "data": data}


def prom_handler(method, path, body):
    u = urlparse(path)
    q = parse_qs(u.query)
    if u.path == "/-/ready":
        return 200, b"ready"
    if u.path == "/api/v1/label/__name__/values":
        return ok(["wt_production_total", "wt_consumption_total", "go_gc_duration"])
    if u.path == "/api/v1/labels":
        return ok(["__name__", "item"])
    if u.path == "/api/v1/metadata":
        return ok({})
    if u.path == "/api/v1/label/item/values":
        return ok(sorted(set(PROD) | set(CONS)))
    expr = q["query"][0]
    table = PROD if "wt_production_total" in expr else CONS
    if u.path == "/api/v1/query_range":
        vals = lambda t: [[1000 + 60 * i, str(t[1] if i < 20 else t[0])] for i in range(31)]
        return ok({"resultType": "matrix", "result": [{"metric": {"item": k}, "values": vals(v)} for k, v in table.items()]})
    if expr.startswith("time() - max(timestamp("):
        return ok({"resultType": "vector", "result": [{"metric": {}, "value": [0, "5"]}]})
    if expr.startswith("time() - max by"):
        return ok({"resultType": "vector", "result": [{"metric": {"item": k}, "value": [0, "5"]} for k in table]})
    idx = 3 if "[10m])" in expr else 1 if "[30m] offset" in expr else 2 if "[120m] offset" in expr else 0
    return ok({"resultType": "vector", "result": [{"metric": {"item": k}, "value": [0, str(v[idx])]} for k, v in table.items()]})


def ollama_handler(method, path, body):
    if path == "/api/chat":
        msg = json.loads(body)["messages"][-1]["content"]
        assert "copper-plate" in msg
        return 200, {"message": {"role": "assistant", "content":
                     "Observed: copper-plate production fell 51%.\nInferred: copper supply is the bottleneck.\n"
                     "Hypothesis: ore supply.\nRecommendation: inspect copper smelting."}}
    if path == "/api/tags":
        return 200, {"models": [{"name": "qwen3:8b"}]}
    if path == "/api/ps":
        return 200, {"models": [{"name": "qwen3:8b", "size": 100, "size_vram": 100}]}
    return 404, {}


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.prom, self.ollama = StubServer(prom_handler), StubServer(ollama_handler)
        self.addCleanup(self.prom.close)
        self.addCleanup(self.ollama.close)
        tmp = tempfile.mkdtemp()
        env = {"FACTORIO_AGENT_PROMETHEUS_URL": self.prom.url, "FACTORIO_AGENT_OLLAMA_URL": self.ollama.url,
               "FACTORIO_AGENT_LOG_DIR": os.path.join(tmp, "log"), "FACTORIO_AGENT_STATE_FILE": os.path.join(tmp, "a.json")}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)
        self.log_dir = env["FACTORIO_AGENT_LOG_DIR"]

    def run_cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()

    def test_status(self):
        code, out = self.run_cli("status")
        self.assertEqual(code, 0, out)
        self.assertIn("auto-detected", out)
        self.assertIn("100% in GPU VRAM", out)
        self.assertIn("Telemetry:   OK", out)

    def test_investigate_item_finds_copper_bottleneck(self):
        code, out = self.run_cli("investigate", "--item", "electronic-circuit", "--window", "1h")
        self.assertEqual(code, 0, out)
        self.assertIn("Likely bottleneck: copper-plate", out)
        self.assertIn("-51%", out)
        self.assertIn("Observed:", out)
        with open(os.path.join(self.log_dir, "investigations.jsonl")) as fh:
            self.assertEqual(json.loads(fh.readline())["subject"], "electronic-circuit")

    def test_llm_flag_selects_anthropic_backend_per_run(self):
        claude = StubServer(lambda m, p, b: (200, {"content": [{"type": "text", "text":
                            "Observed: copper-plate -51%.\nRecommendation: check smelters."}]}))
        self.addCleanup(claude.close)
        env = {"ANTHROPIC_API_KEY": "sk-test", "FACTORIO_AGENT_ANTHROPIC_URL": claude.url}
        with mock.patch.dict(os.environ, env):
            code, out = self.run_cli("--llm", "anthropic", "investigate", "--item", "electronic-circuit")
            code2, out2 = self.run_cli("investigate", "--item", "electronic-circuit", "--llm", "anthropic")  # flag after subcommand
        self.assertEqual((code, code2), (0, 0), out)
        self.assertIn("check smelters", out)
        self.assertIn("check smelters", out2)
        self.assertEqual(sum(1 for r in claude.requests if r[1] == "/v1/messages"), 2)
        self.assertFalse(any(r[1] == "/api/chat" for r in self.ollama.requests))      # Ollama never used

    def test_llm_flag_defaults_to_config_and_rejects_unknown(self):
        code, out = self.run_cli("investigate", "--item", "electronic-circuit")
        self.assertIn("Observed:", out)
        self.assertTrue(any(r[1] == "/api/chat" for r in self.ollama.requests))
        with self.assertRaises(SystemExit):
            self.run_cli("--llm", "gpt", "status")

    def test_investigate_factory(self):
        code, out = self.run_cli("investigate", "--window", "30m")
        self.assertEqual(code, 0, out)
        self.assertIn("Investigation: factory", out)

    def test_metrics_and_query_commands(self):
        code, out = self.run_cli("metrics")
        self.assertIn("wt_production_total", out)
        self.assertNotIn("go_gc", out)
        code, out = self.run_cli("query", "wt_production_total")
        self.assertIn("electronic-circuit", out)

    def test_prometheus_down_is_reported_not_crashed(self):
        with mock.patch.dict(os.environ, {"FACTORIO_AGENT_PROMETHEUS_URL": "http://127.0.0.1:9"}):
            code, out = self.run_cli("investigate", "--item", "electronic-circuit")
        self.assertEqual(code, 1)
        self.assertIn("Cannot investigate", out)


if __name__ == "__main__":
    unittest.main()
