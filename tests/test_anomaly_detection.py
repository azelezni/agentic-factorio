import os
import tempfile
import unittest

from helpers import FakeLLM, FakeTelemetry, cfg, flow, item, snapshot
from factorio_agent.models.anomalies import AlertState, AnomalyType
from factorio_agent.models.metrics import DataStatus
from factorio_agent.monitoring import rules
from factorio_agent.monitoring.alert_manager import AlertManager
from factorio_agent.monitoring.engine import MonitoringEngine

C = cfg()


def types(snap):
    return {a.type for a in rules.evaluate_item(snap, C)}


class RuleTests(unittest.TestCase):
    def test_production_drop_vs_short_baseline(self):
        a = rules.evaluate_item(item("electronic-circuit", flow(69, 120)), C)[0]
        self.assertEqual(a.type, AnomalyType.PRODUCTION_DROP)
        self.assertAlmostEqual(a.change, -0.425, places=2)
        self.assertEqual((a.baseline, a.current), (120, 69))

    def test_small_change_is_not_an_anomaly(self):
        self.assertEqual(types(item("x", flow(100, 120))), set())

    def test_production_stopped_takes_precedence_over_drop(self):
        t = types(item("x", flow(0, 100, long=100, recent=0)))
        self.assertEqual(t, {AnomalyType.PRODUCTION_STOPPED})

    def test_valid_zero_for_idle_item_is_not_stopped(self):
        self.assertEqual(types(item("x", flow(0, 0, long=0, recent=0))), set())

    def test_missing_values_never_become_zero(self):
        from factorio_agent.models.metrics import FlowStats
        self.assertEqual(types(item("x", FlowStats(None, None, None, None, None))), set())
        self.assertEqual(types(item("x", FlowStats(None, 100, 100, None, None))), set())

    def test_stale_item_is_skipped(self):
        self.assertEqual(types(item("x", flow(0, 100, recent=0), fresh=False)), set())

    def test_throughput_degradation_vs_long_baseline(self):
        t = types(item("x", flow(100, 105, long=200)))
        self.assertEqual(t, {AnomalyType.THROUGHPUT_DEGRADATION})

    def test_consumption_drop(self):
        t = types(item("x", flow(100, 100), flow(40, 100)))
        self.assertIn(AnomalyType.CONSUMPTION_DROP, t)

    def test_resource_shortage(self):
        a = [x for x in rules.evaluate_item(item("copper-plate", flow(80, 80), flow(120, 120)), C)
             if x.type is AnomalyType.RESOURCE_SHORTAGE]
        self.assertEqual(len(a), 1)

    def test_unstable_production(self):
        self.assertIn(AnomalyType.UNSTABLE_PRODUCTION, types(item("x", flow(100, 100, cv=0.9))))

    def test_thresholds_are_configurable(self):
        strict = cfg(drop_threshold=0.10)
        self.assertEqual(rules.evaluate_item(item("x", flow(85, 100)), strict)[0].type, AnomalyType.PRODUCTION_DROP)

    def test_failed_snapshot_yields_no_anomalies(self):
        self.assertEqual(rules.evaluate_all(snapshot(status=DataStatus.FAILED), C), [])


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def drop(change_to=69):
    return rules.evaluate_item(item("electronic-circuit", flow(change_to, 120)), C)


class AlertManagerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.mgr = AlertManager(cfg(), None, self.clock)
        self.obs = {"electronic-circuit"}

    def test_new_anomaly_is_detected_once(self):
        r1 = self.mgr.update(drop(), self.obs)
        self.assertEqual([a.state for a in r1.new], [AlertState.DETECTED])
        self.mgr.mark_investigating(r1.new[0])
        self.mgr.finish_investigation(r1.new[0], True)
        self.clock.t += 60
        r2 = self.mgr.update(drop(), self.obs)
        self.assertEqual((r2.new, r2.reinvestigate), ([], []))       # deduplicated
        self.assertEqual(self.mgr.active_alerts()[0].state, AlertState.ACTIVE)

    def test_reinvestigates_after_interval(self):
        a = self.mgr.update(drop(), self.obs).new[0]
        self.mgr.finish_investigation(a, True)
        self.clock.t += 901
        self.assertEqual(len(self.mgr.update(drop(), self.obs).reinvestigate), 1)

    def test_reinvestigates_on_meaningful_change(self):
        a = self.mgr.update(drop(69), self.obs).new[0]            # -42.5%
        self.mgr.finish_investigation(a, True)
        self.clock.t += 60
        self.assertEqual(self.mgr.update(drop(66), self.obs).reinvestigate, [])        # tiny change
        self.assertEqual(len(self.mgr.update(drop(30), self.obs).reinvestigate), 1)    # -75%

    def test_failed_investigation_is_retried_later(self):
        a = self.mgr.update(drop(), self.obs).new[0]
        self.mgr.mark_investigating(a)
        self.mgr.finish_investigation(a, False)                   # e.g. LLM timeout
        self.clock.t += 60
        self.assertEqual(self.mgr.update(drop(), self.obs).reinvestigate, [])
        self.clock.t += 300
        self.assertEqual(len(self.mgr.update(drop(), self.obs).reinvestigate), 1)

    def test_resolves_after_consecutive_clear_cycles(self):
        self.mgr.update(drop(), self.obs)
        for _ in range(2):
            self.assertEqual(self.mgr.update([], self.obs).resolved, [])
        r = self.mgr.update([], self.obs)
        self.assertEqual(r.resolved[0].state, AlertState.RESOLVED)
        self.assertEqual(self.mgr.active_alerts(), [])

    def test_unobserved_items_are_not_resolved(self):
        self.mgr.update(drop(), self.obs)
        for _ in range(10):
            self.mgr.update([], set())          # telemetry for the item is missing/stale
        self.assertEqual(len(self.mgr.active_alerts()), 1)

    def test_restart_resumes_without_duplicate_detection(self):
        path = os.path.join(tempfile.mkdtemp(), "alerts.json")
        m1 = AlertManager(cfg(), path, self.clock)
        a = m1.update(drop(), self.obs).new[0]
        m1.mark_investigating(a)                # crash mid-investigation
        m2 = AlertManager(cfg(), path, self.clock)
        self.assertEqual(m2.active_alerts()[0].state, AlertState.ACTIVE)
        self.assertEqual(m2.update(drop(), self.obs).new, [])


class RecordingInvestigator:
    def __init__(self, ok=True, graph=None):
        self.calls, self.ok, self.graph = [], ok, graph

    def investigate_anomaly(self, anomaly, snapshot=None):
        from factorio_agent.investigation.evidence import Evidence
        from factorio_agent.investigation.report import Report
        self.calls.append(anomaly.key)
        return Report(anomaly.item, Evidence(anomaly.item, 60, 0), "analysis" if self.ok else None, None if self.ok else "down")


class MonitoringEngineTests(unittest.TestCase):
    def make(self, snap, inv=None):
        self.lines = []
        self.tel = FakeTelemetry(snap)
        self.inv = inv or RecordingInvestigator()
        self.clock = Clock()
        eng = MonitoringEngine(self.tel, AlertManager(cfg(), None, self.clock), self.inv, cfg(),
                               out=self.lines.append, synchronous=True)
        return eng

    def test_healthy_factory_never_invokes_llm(self):
        eng = self.make(snapshot(item("iron-plate", flow(100, 100))))
        for _ in range(3):
            eng.run_cycle()
        self.assertEqual(self.inv.calls, [])

    def test_anomaly_triggers_exactly_one_investigation(self):
        eng = self.make(snapshot(item("electronic-circuit", flow(69, 120))))
        for _ in range(4):
            eng.run_cycle()
        self.assertEqual(self.inv.calls, ["production_drop:electronic-circuit"])
        text = "\n".join(self.lines)
        self.assertIn("ANOMALY DETECTED", text)
        self.assertIn("Investigation complete", text)
        self.assertEqual(text.count("ANOMALY DETECTED"), 1)

    def test_one_root_cause_triggers_one_investigation(self):
        from factorio_agent.investigation.relations import RecipeGraph
        snap = snapshot(item("electronic-circuit", flow(69, 120)), item("copper-cable", flow(110, 200)),
                        item("copper-plate", flow(100, 204), flow(230, 205)), item("iron-plate", flow(300, 300)))
        eng = self.make(snap, RecordingInvestigator(graph=RecipeGraph()))
        eng.run_cycle()
        self.assertEqual(self.inv.calls, ["production_drop:electronic-circuit"])   # not one per upstream item
        self.assertIn("also covers upstream: copper-cable, copper-plate", "\n".join(self.lines))
        states = {a.state for a in eng.alerts.active_alerts()}
        self.assertEqual(states, {AlertState.ACTIVE})                              # covered alerts are not re-triggered
        eng.run_cycle()
        self.assertEqual(len(self.inv.calls), 1)

    def test_independent_items_are_investigated_separately(self):
        from factorio_agent.investigation.relations import RecipeGraph
        snap = snapshot(item("iron-gear-wheel", flow(30, 60)), item("sulfur", flow(5, 50)))
        eng = self.make(snap, RecordingInvestigator(graph=RecipeGraph()))
        eng.run_cycle()
        self.assertEqual(sorted(self.inv.calls), ["production_drop:iron-gear-wheel", "production_drop:sulfur"])

    def test_monitoring_continues_when_llm_unavailable(self):
        eng = self.make(snapshot(item("electronic-circuit", flow(69, 120))), RecordingInvestigator(ok=False))
        eng.run_cycle()
        self.assertIn("LLM analysis unavailable", "\n".join(self.lines))
        self.assertIsNotNone(eng.run_cycle())          # next cycle still runs

    def test_unavailable_telemetry_raises_no_alerts_and_reports_once(self):
        eng = self.make(snapshot(status=DataStatus.MISSING, detail="no series"))
        self.assertIsNone(eng.run_cycle())
        eng.run_cycle()
        text = "\n".join(self.lines)
        self.assertEqual(text.count("TELEMETRY UNAVAILABLE"), 1)
        self.assertNotIn("ANOMALY", text)
        self.assertEqual(self.inv.calls, [])

    def test_stale_item_is_reported_and_skipped(self):
        eng = self.make(snapshot(item("copper-plate", flow(0, 100, recent=0), fresh=False)))
        eng.run_cycle()
        text = "\n".join(self.lines)
        self.assertIn("missing/stale", text)
        self.assertNotIn("ANOMALY", text)


if __name__ == "__main__":
    unittest.main()
