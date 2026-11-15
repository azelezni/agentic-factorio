"""Shared test doubles."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from factorio_agent.config import Config
from factorio_agent.llm.client import LLMClient, LLMUnavailable
from factorio_agent.models.metrics import DataStatus, FactorySnapshot, FlowStats, ItemSnapshot, Sample


def flow(cur, short, long=None, recent=None, cv=None):
    return FlowStats(cur, short, short if long is None else long, cur if recent is None else recent, cv)


def item(name, prod=None, cons=None, fresh=True):
    return ItemSnapshot(name, prod, cons, fresh, 10.0 if fresh else 999.0)


def snapshot(*items, ts=10_000.0, status=DataStatus.OK, detail=""):
    return FactorySnapshot(ts, status, detail, {i.item: i for i in items})


def series(start, values, step=60):
    return [Sample(start + i * step, v) for i, v in enumerate(values)]


class FakeTelemetry:
    def __init__(self, snap, histories=None):
        self.snap, self.histories, self.calls = snap, histories or {}, 0

    def snapshot(self):
        self.calls += 1
        return self.snap

    def history(self, flow_name, item_name, minutes):
        return self.histories.get((flow_name, item_name), [])

    def items(self):
        return sorted(self.snap.items)


class FakeLLM(LLMClient):
    def __init__(self, reply="Observed: x\nInferred: y\nHypothesis: z\nRecommendation: w", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def generate(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise LLMUnavailable("Ollama unreachable (test)")
        return self.reply


class StubServer:
    """Tiny HTTP server: handler(path, query, body) -> (status, payload | bytes)."""

    def __init__(self, handler):
        outer = self
        self.requests = []

        class H(BaseHTTPRequestHandler):
            def _do(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append((self.command, self.path, body))
                status, payload = handler(self.command, self.path, body)
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client timed out (intentional in timeout tests)

            do_GET = do_POST = _do

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def cfg(**kw):
    return Config(**kw)
