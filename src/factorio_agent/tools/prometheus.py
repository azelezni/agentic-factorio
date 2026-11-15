"""Read-only Prometheus query layer (design 7).

Only the four tool functions from the design plus a few discovery helpers are
exposed; the LLM never issues HTTP requests itself.
"""
from __future__ import annotations

import json
import math
import socket
import urllib.error
import urllib.parse
import urllib.request

from factorio_agent.models.metrics import Sample, Series


class PrometheusError(Exception):
    """Prometheus is unreachable, returned an error, or sent an unusable response."""


class PrometheusClient:
    def __init__(self, base_url: str = "http://localhost:9090", timeout: float = 10.0,
                 max_bytes: int = 8_000_000):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_bytes = max_bytes

    # ------------------------------------------------------------------ transport
    def _get(self, path: str, params: dict | None = None) -> dict:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                raw = resp.read(self.max_bytes + 1)
        except urllib.error.HTTPError as exc:
            body = exc.read(2000).decode("utf-8", "replace")
            raise PrometheusError(f"HTTP {exc.code} from Prometheus: {body.strip()}") from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            raise PrometheusError(f"Prometheus unreachable at {self.base_url}: {exc}") from exc
        if len(raw) > self.max_bytes:
            raise PrometheusError(f"Prometheus response exceeded {self.max_bytes} bytes")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise PrometheusError(f"Prometheus returned invalid JSON: {exc}") from exc
        if payload.get("status") != "success":
            raise PrometheusError(f"Prometheus error: {payload.get('errorType')}: {payload.get('error')}")
        return payload

    # ------------------------------------------------------------------ queries
    def query_instant(self, query: str, timestamp: str | None = None) -> list[Series]:
        params = {"query": query}
        if timestamp is not None:
            params["time"] = str(timestamp)
        return _normalize(self._get("/api/v1/query", params)["data"])

    def query_range(self, query: str, start: str, end: str, step: str) -> list[Series]:
        params = {"query": query, "start": str(start), "end": str(end), "step": step}
        return _normalize(self._get("/api/v1/query_range", params)["data"])

    # ------------------------------------------------------------------ discovery
    def list_metrics(self) -> list[str]:
        return sorted(self._get("/api/v1/label/__name__/values")["data"])

    def get_metric_labels(self, metric_name: str) -> list[str]:
        data = self._get("/api/v1/labels", {"match[]": metric_name})["data"]
        return sorted(label for label in data if label != "__name__")

    def label_values(self, label: str, metric_name: str | None = None) -> list[str]:
        params = {"match[]": metric_name} if metric_name else None
        return sorted(self._get(f"/api/v1/label/{urllib.parse.quote(label)}/values", params)["data"])

    def metadata(self) -> dict[str, dict]:
        """{metric: {"type": ..., "help": ...}} - empty if the server has no metadata."""
        try:
            data = self._get("/api/v1/metadata")["data"]
        except PrometheusError:
            return {}
        return {name: {"type": e[0].get("type", ""), "help": e[0].get("help", "")}
                for name, e in data.items() if e}

    def is_ready(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/-/ready", timeout=self.timeout) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False


def _to_float(raw) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None   # NaN / Inf == "no usable value"


def _samples(pairs) -> list[Sample]:
    out = []
    for ts, raw in pairs:
        value = _to_float(raw)
        if value is not None:
            out.append(Sample(float(ts), value))
    return out


def _normalize(data: dict) -> list[Series]:
    rtype, result = data.get("resultType"), data.get("result")
    if rtype == "scalar":
        series = [Series({}, _samples([result]))]
    elif rtype in ("vector", "matrix"):
        series = []
        for entry in result:
            pairs = entry["values"] if rtype == "matrix" else [entry["value"]]
            series.append(Series(dict(entry.get("metric", {})), _samples(pairs)))
    else:
        raise PrometheusError(f"Unsupported Prometheus result type: {rtype!r}")
    return [s for s in series if s.samples]
