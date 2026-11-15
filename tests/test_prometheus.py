import math
import unittest
from urllib.parse import parse_qs, urlparse

from helpers import StubServer
from factorio_agent.config import Config
from factorio_agent.tools.catalog import MetricCatalog, SchemaError, resolve_schema
from factorio_agent.tools.prometheus import PrometheusClient, PrometheusError


def ok(data):
    return 200, {"status": "success", "data": data}


class PrometheusClientTests(unittest.TestCase):
    def serve(self, handler):
        srv = StubServer(handler)
        self.addCleanup(srv.close)
        return srv, PrometheusClient(srv.url, timeout=2, max_bytes=5000)

    def test_instant_query_normalizes_vector(self):
        srv, prom = self.serve(lambda m, p, b: ok({"resultType": "vector", "result": [
            {"metric": {"item": "iron-plate"}, "value": [1700000000, "12.5"]}]}))
        res = prom.query_instant("up", timestamp="1700000000")
        self.assertEqual(res[0].labels, {"item": "iron-plate"})
        self.assertEqual(res[0].latest.value, 12.5)
        q = parse_qs(urlparse(srv.requests[0][1]).query)
        self.assertEqual((q["query"], q["time"]), (["up"], ["1700000000"]))

    def test_range_query_normalizes_matrix_and_drops_nan(self):
        _, prom = self.serve(lambda m, p, b: ok({"resultType": "matrix", "result": [
            {"metric": {"item": "a"}, "values": [[1, "1"], [2, "NaN"], [3, "3"]]}]}))
        res = prom.query_range("x", "0", "10", "60s")
        self.assertEqual([s.value for s in res[0].samples], [1.0, 3.0])

    def test_empty_result_is_empty_list_not_zero(self):
        _, prom = self.serve(lambda m, p, b: ok({"resultType": "vector", "result": []}))
        self.assertEqual(prom.query_instant("missing_metric"), [])

    def test_all_nan_series_is_dropped(self):
        _, prom = self.serve(lambda m, p, b: ok({"resultType": "vector", "result": [
            {"metric": {}, "value": [1, "NaN"]}]}))
        self.assertEqual(prom.query_instant("x"), [])

    def test_prometheus_error_payload(self):
        _, prom = self.serve(lambda m, p, b: (200, {"status": "error", "errorType": "bad_data", "error": "parse error"}))
        with self.assertRaisesRegex(PrometheusError, "parse error"):
            prom.query_instant("(((")

    def test_http_error_and_unreachable(self):
        _, prom = self.serve(lambda m, p, b: (500, b"boom"))
        with self.assertRaisesRegex(PrometheusError, "HTTP 500"):
            prom.query_instant("x")
        with self.assertRaises(PrometheusError):
            PrometheusClient("http://127.0.0.1:9", timeout=1).query_instant("x")

    def test_response_size_limit(self):
        _, prom = self.serve(lambda m, p, b: (200, b"x" * 6000))
        with self.assertRaisesRegex(PrometheusError, "exceeded"):
            prom.query_instant("x")

    def test_metric_and_label_discovery(self):
        def handler(m, path, b):
            if path.startswith("/api/v1/label/__name__/values"):
                return ok(["factorio_item_production_total", "go_goroutines"])
            if path.startswith("/api/v1/labels"):
                return ok(["__name__", "item", "force"])
            if path.startswith("/api/v1/label/item/values"):
                return ok(["iron-plate", "copper-plate"])
            if path.startswith("/api/v1/metadata"):
                return ok({"factorio_item_production_total": [{"type": "counter", "help": "produced"}]})
            return 404, b""
        _, prom = self.serve(handler)
        self.assertEqual(prom.get_metric_labels("factorio_item_production_total"), ["force", "item"])
        self.assertEqual(prom.label_values("item"), ["copper-plate", "iron-plate"])
        cat = MetricCatalog.build(prom)
        self.assertEqual(list(cat.metrics), ["factorio_item_production_total"])   # go_* filtered
        self.assertEqual(cat.metrics["factorio_item_production_total"].type, "counter")


class SchemaTests(unittest.TestCase):
    def catalog(self, *names, labels=("item",)):
        from factorio_agent.tools.catalog import MetricInfo
        return MetricCatalog({n: MetricInfo(n, labels=list(labels)) for n in names})

    def test_auto_detects_unique_metrics(self):
        s = resolve_schema(Config(), self.catalog("wt_production_total", "wt_consumption_total"))
        self.assertEqual((s.production_metric, s.consumption_metric, s.item_label),
                         ("wt_production_total", "wt_consumption_total", "item"))

    def test_ambiguous_schema_raises_with_guidance(self):
        with self.assertRaisesRegex(SchemaError, "Cannot uniquely"):
            resolve_schema(Config(), self.catalog("a_production", "b_production", "c_consumption"))

    def test_explicit_config_needs_no_discovery(self):
        s = resolve_schema(Config(production_metric="p", consumption_metric="c", item_label="name"), None)
        self.assertEqual(s.item_label, "name")


if __name__ == "__main__":
    unittest.main()
