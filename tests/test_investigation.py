import json
import os
import tempfile
import unittest

from helpers import FakeLLM, FakeTelemetry, cfg, flow, item, series, snapshot
from factorio_agent.investigation.engine import Investigator, TelemetryUnavailable, parse_window
from factorio_agent.investigation.evidence import find_decline_start
from factorio_agent.investigation.relations import RecipeGraph
from factorio_agent.investigation.report import ReportLog
from factorio_agent.models.metrics import DataStatus, Sample
from factorio_agent.tools.catalog import MetricSchema
from factorio_agent.tools.factory import FactoryTelemetry

C = cfg(log_dir="")


def copper_scenario():
    """Circuits -42%, cable -45%, copper plate -51%; copper demand stays high; gears unaffected."""
    return snapshot(
        item("electronic-circuit", flow(69, 120), flow(69, 120)),
        item("copper-cable", flow(110, 200), flow(220, 230)),
        item("copper-plate", flow(100, 204), flow(200, 205)),
        item("copper-ore", flow(210, 215), flow(210, 215)),
        item("iron-plate", flow(300, 300), flow(250, 250)),
        item("iron-gear-wheel", flow(60, 60)),
        item("automation-science-pack", flow(30, 60), flow(30, 60)),   # competes for copper-plate
    )


def histories():
    t0 = 100_000
    return {
        ("production", "electronic-circuit"): series(t0, [120] * 30 + [70] * 10),
        ("production", "copper-plate"): series(t0, [205] * 28 + [100] * 12),
    }


class InvestigationTests(unittest.TestCase):
    def setUp(self):
        self.llm = FakeLLM()
        self.inv = Investigator(FakeTelemetry(copper_scenario(), histories()), self.llm, C)

    def test_identifies_upstream_copper_bottleneck_with_evidence(self):
        report = self.inv.investigate_item("electronic-circuit", 60)
        ev = report.evidence
        self.assertEqual(ev.candidates[0].item, "copper-plate")       # root-most problem, not copper-cable
        self.assertEqual(ev.candidates[0].kind, "production_drop")
        self.assertEqual({i.item for i in ev.inputs}, {"iron-plate", "copper-cable", "copper-plate", "copper-ore"})
        self.assertIn("automation-science-pack", {c.item for c in ev.competitors})
        self.assertEqual(ev.confidence, "High")                        # input fell first, demand stayed high
        self.assertEqual(ev.scope, "partial")                          # science packs also affected
        text = report.format()
        self.assertIn("Likely bottleneck: copper-plate", text)
        self.assertIn("-51%", text)

    def test_llm_receives_evidence_and_reports_distinguish_sections(self):
        self.inv.investigate_item("electronic-circuit", 60, question="Why is it dropping?")
        system, user = self.llm.calls[0]
        self.assertIn("Observed", system["content"])
        self.assertIn("Hypothesis", system["content"])
        self.assertIn("copper-plate", user["content"])
        self.assertIn("Why is it dropping?", user["content"])

    def test_decline_start_is_found(self):
        ev = self.inv.investigate_item("electronic-circuit", 60).evidence
        self.assertEqual(ev.target.drop_started_at, 100_000 + 30 * 60)
        self.assertLessEqual(ev.candidates[0].drop_started_at, ev.target.drop_started_at)

    def test_llm_unavailable_still_returns_deterministic_findings(self):
        inv = Investigator(FakeTelemetry(copper_scenario(), histories()), FakeLLM(fail=True), C)
        report = inv.investigate_item("electronic-circuit", 60)
        self.assertFalse(report.llm_ok)
        out = report.format()
        self.assertIn("LLM analysis unavailable", out)
        self.assertIn("copper-plate", out)

    def test_unavailable_telemetry_raises_instead_of_reporting_zero(self):
        inv = Investigator(FakeTelemetry(snapshot(status=DataStatus.FAILED, detail="down")), self.llm, C)
        with self.assertRaises(TelemetryUnavailable):
            inv.investigate_item("electronic-circuit")
        self.assertEqual(self.llm.calls, [])

    def test_unknown_item_reports_gap_not_zero(self):
        ev = self.inv.investigate_item("rocket-fuel", 60).evidence
        self.assertIsNone(ev.target)
        self.assertTrue(any("No telemetry" in g for g in ev.gaps))

    def test_healthy_item_has_low_confidence_and_no_candidates(self):
        snap = snapshot(item("iron-gear-wheel", flow(60, 60)), item("iron-plate", flow(300, 300)))
        ev = Investigator(FakeTelemetry(snap), self.llm, C).investigate_item("iron-gear-wheel").evidence
        self.assertEqual((ev.candidates, ev.confidence), ([], "Low"))

    def test_scope_ignores_downstream_effects_and_explains_healthy_inputs(self):
        snap = snapshot(item("copper-plate", flow(100, 204)), item("copper-ore", flow(210, 215)),
                        item("copper-cable", flow(110, 200)), item("electronic-circuit", flow(69, 120)))
        ev = Investigator(FakeTelemetry(snap), self.llm, C).investigate_item("copper-plate").evidence
        self.assertEqual(ev.scope, "isolated")          # cable/circuit are consequences, not separate problems
        self.assertEqual(ev.candidates, [])
        self.assertTrue(any("own production step" in n for n in ev.notes))

    def test_widespread_scope(self):
        snap = snapshot(item("a", flow(10, 100)), item("b", flow(10, 100)), item("c", flow(10, 100)),
                        item("iron-gear-wheel", flow(60, 60)))
        ev = Investigator(FakeTelemetry(snap), self.llm, C).investigate_item("iron-gear-wheel").evidence
        self.assertEqual(ev.scope, "widespread")

    def test_factory_investigation_ranks_stopped_first(self):
        snap = snapshot(item("steel-plate", flow(0, 50, long=50, recent=0)),
                        item("copper-plate", flow(80, 80), flow(150, 150)),
                        item("iron-plate", flow(100, 100)))
        report = Investigator(FakeTelemetry(snap), self.llm, C).investigate_factory(60)
        self.assertEqual(report.evidence.candidates[0].item, "steel-plate")
        self.assertIn("copper-plate", {c.item for c in report.evidence.competitors})

    def test_evidence_is_logged(self):
        d = tempfile.mkdtemp()
        inv = Investigator(FakeTelemetry(copper_scenario(), histories()), self.llm, C, log=ReportLog(d))
        inv.investigate_item("electronic-circuit")
        with open(os.path.join(d, "investigations.jsonl")) as fh:
            rec = json.loads(fh.readline())
        self.assertEqual(rec["subject"], "electronic-circuit")
        self.assertEqual(rec["evidence"]["candidates"][0]["item"], "copper-plate")

    def test_ask_matches_items_and_windows(self):
        rep = self.inv.ask("Why is electronic circuit production dropping over the last 2 hours?")
        self.assertEqual((rep.subject, rep.evidence.window_minutes), ("electronic-circuit", 120))
        self.assertEqual(self.inv.ask("Which resources are limiting production?").subject, "factory")
        self.assertFalse(self.inv.is_factory_question("tell me a joke"))
        self.assertTrue(self.inv.is_factory_question("explain the biggest bottleneck"))


class HelperTests(unittest.TestCase):
    def test_parse_window(self):
        self.assertEqual([parse_window(t, 5) for t in ("1h", "30 minutes", "last hour", "2 hours", "hello")],
                         [60, 30, 60, 120, 5])

    def test_find_decline_start_requires_current_decline(self):
        self.assertIsNone(find_decline_start(series(0, [100] * 30), 0.3))
        self.assertIsNone(find_decline_start(series(0, [100] * 20 + [50] * 3 + [100] * 3), 0.3))  # recovered
        self.assertEqual(find_decline_start(series(0, [100] * 20 + [50] * 10), 0.3), 20 * 60)

    def test_recipe_graph(self):
        g = RecipeGraph()
        self.assertEqual({n for n, _ in g.upstream("electronic-circuit")},
                         {"iron-plate", "copper-cable", "copper-plate", "iron-ore", "copper-ore"})
        self.assertIn("electronic-circuit", g.consumers("iron-plate"))


class FakeProm:
    """Answers the FactoryTelemetry queries from a dict keyed by a substring of the expression."""
    base_url = "fake"

    def __init__(self, ready=True, age=10.0, items=("iron-plate",), up=None):
        self.ready, self.age, self.items, self.up, self.queries = ready, age, items, up, []

    def is_ready(self):
        return self.ready

    def query_instant(self, q, timestamp=None):
        from factorio_agent.models.metrics import Series
        self.queries.append(q)
        if q.startswith("up{"):
            return [] if self.up is None else [Series({}, [Sample(0, self.up)])]
        if q.startswith("time() - max(timestamp("):
            return [] if self.age is None else [Series({}, [Sample(0, self.age)])]
        if q.startswith("time() - max by"):
            return [Series({"item": i}, [Sample(0, self.age or 0)]) for i in self.items]
        val = 0.0 if "offset" not in q and "[10m]" in q else 120.0
        return [Series({"item": i}, [Sample(0, val)]) for i in self.items]

    def query_range(self, q, start, end, step):
        from factorio_agent.models.metrics import Series
        self.queries.append(q)
        return [Series({"item": i}, series(0, [100.0, 120.0, 80.0, 110.0, 90.0, 100.0])) for i in self.items]

    def label_values(self, label, metric=None):
        return list(self.items)


class FactoryTelemetryTests(unittest.TestCase):
    schema = MetricSchema("prod_total", "cons_total", "item")

    def tel(self, prom, **kw):
        return FactoryTelemetry(prom, cfg(**kw), self.schema, clock=lambda: 10_000.0)

    def test_snapshot_builds_rates_and_expressions(self):
        prom = FakeProm()
        snap = self.tel(prom).snapshot()
        self.assertTrue(snap.ok)
        st = snap.items["iron-plate"]
        self.assertTrue(st.fresh)
        self.assertEqual((st.production.current, st.production.short), (120.0, 120.0))
        self.assertIsNotNone(st.production.cv)
        self.assertIn("sum by (item) (rate(prod_total[5m])) * 60", prom.queries)
        self.assertIn("sum by (item) (rate(prod_total[30m] offset 5m)) * 60", prom.queries)
        self.assertIn("sum by (item) (rate(prod_total[2h] offset 5m)) * 60".replace("2h", "120m"), prom.queries)

    def test_valid_zero_is_preserved(self):
        snap = self.tel(FakeProm()).snapshot()
        self.assertEqual(snap.items["iron-plate"].production.recent, 0.0)   # 10m window stub returns 0

    def test_missing_series_is_missing_not_zero(self):
        snap = self.tel(FakeProm(age=None)).snapshot()
        self.assertEqual(snap.status, DataStatus.MISSING)
        self.assertEqual(snap.items, {})

    def test_stale_metric(self):
        self.assertEqual(self.tel(FakeProm(age=600)).snapshot().status, DataStatus.STALE)

    def test_prometheus_down_is_failed(self):
        self.assertEqual(self.tel(FakeProm(ready=False)).snapshot().status, DataStatus.FAILED)

    def test_bridge_down_detected_via_up(self):
        snap = self.tel(FakeProm(up=0), bridge_job="wiretap").snapshot()
        self.assertEqual(snap.status, DataStatus.FAILED)
        self.assertIn("Wiretap", snap.detail)

    def test_gauge_mode_expression(self):
        prom = FakeProm()
        self.tel(prom, metric_kind="gauge", gauge_scale=60.0).snapshot()
        self.assertIn("sum by (item) (avg_over_time(prod_total[5m])) * 60.0", prom.queries)

    def test_item_name_is_escaped_in_selectors(self):
        t = self.tel(FakeProm())
        self.assertIn('item="a\\"b"', t._selector("m", 'a"b'))


if __name__ == "__main__":
    unittest.main()
