"""Continuous monitoring loop (design 4.1).

Deterministic and independent of the LLM: rules run every cycle; the
investigator (and therefore the LLM) is only invoked for new or materially
changed anomalies, on a single worker thread so a slow model never stalls
metric collection.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from factorio_agent.config import Config
from factorio_agent.models.anomalies import Alert
from factorio_agent.models.metrics import FactorySnapshot
from factorio_agent.monitoring import rules
from factorio_agent.monitoring.alert_manager import AlertManager, UpdateResult


def stamp() -> str:
    return time.strftime("[%H:%M:%S]")


class MonitoringEngine:
    def __init__(self, telemetry, alerts: AlertManager, investigator, cfg: Config,
                 out: Callable[[str], None] = print, synchronous: bool = False):
        self.telemetry, self.alerts, self.investigator, self.cfg = telemetry, alerts, investigator, cfg
        self._out_fn = out
        self._print_lock = threading.Lock()
        self._known_items: set[str] = set()
        self._missing_reported: set[str] = set()
        self._telemetry_down = False
        self._synchronous = synchronous
        self._executor = None if synchronous else ThreadPoolExecutor(max_workers=1)

    def out(self, text: str) -> None:
        with self._print_lock:
            self._out_fn(text)

    # ------------------------------------------------------------------ one cycle
    def run_cycle(self) -> UpdateResult | None:
        snapshot = self.telemetry.snapshot()
        if not snapshot.ok:
            # Never convert missing/stale/failed data into zero production.
            if not self._telemetry_down:
                self.out(f"{stamp()} TELEMETRY UNAVAILABLE ({snapshot.status.value}): {snapshot.detail}\n"
                         "Rules are paused; no alerts will be raised or resolved until data returns.")
            self._telemetry_down = True
            return None
        if self._telemetry_down:
            self.out(f"{stamp()} Telemetry recovered.")
            self._telemetry_down = False

        self._report_missing(snapshot)
        observed = {n for n, s in snapshot.items.items() if s.fresh}
        anomalies = rules.evaluate_all(snapshot, self.cfg)
        result = self.alerts.update(anomalies, observed)
        self.out(f"{stamp()} Factory metrics collected ({len(snapshot.items)} items, "
                 f"{len(self.alerts.active_alerts())} active alerts).")

        for alert in result.new:
            self.out(self.format_anomaly(alert))
        for alert in result.resolved:
            self.out(f"{stamp()} RESOLVED: {alert.anomaly.message}")
        self._start_investigations(result.to_investigate, snapshot)
        return result

    def _report_missing(self, snapshot: FactorySnapshot) -> None:
        present = set(snapshot.items)
        bad = (self._known_items - present) | {n for n, s in snapshot.items.items() if not s.fresh}
        for name in sorted(bad - self._missing_reported):
            self.out(f"{stamp()} Telemetry for '{name}' is missing/stale; skipping its rules (not treated as zero).")
        self._missing_reported = bad          # recovered items drop out, so a recurrence is reported again
        self._known_items |= present

    @staticmethod
    def format_anomaly(alert: Alert) -> str:
        a = alert.anomaly
        lines = [f"\n{stamp()} ANOMALY DETECTED [{a.severity.upper()}]",
                 f"Product: {a.item}", f"Type: {a.type.value}", a.message]
        if a.change is not None:       # drop-style anomalies only; others carry their numbers in the message
            lines.append(f"Change: {a.change * 100:+.0f}%")
            if a.baseline is not None and a.current is not None:
                lines += [f"Baseline: {a.baseline:.1f}/min", f"Current: {a.current:.1f}/min"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ investigations
    def _plan(self, alerts: list[Alert]) -> list[tuple[Alert, list[Alert]]]:
        """Group alerts into investigations: [(primary alert, all alerts it covers)].

        One investigation per item, and an item whose problem lies upstream of another
        alerting item is covered by that item's investigation (its evidence walks the chain),
        so a single root cause does not trigger a cascade of LLM calls.
        """
        by_item: dict[str, list[Alert]] = {}
        for alert in alerts:
            by_item.setdefault(alert.anomaly.item, []).append(alert)
        graph = getattr(self.investigator, "graph", None)
        upstream = {i: ({n for n, _ in graph.upstream(i)} if graph else set()) for i in by_item}
        tops = [i for i in by_item if not any(i in upstream[z] for z in by_item if z != i)] or list(by_item)
        covered: dict[str, list[Alert]] = {t: list(by_item[t]) for t in tops}
        for item, group in by_item.items():
            if item not in covered:
                owner = next((t for t in tops if item in upstream[t]), tops[0])
                covered[owner].extend(group)
        plan = []
        for top, group in covered.items():
            primary = max(by_item[top], key=lambda a: (a.anomaly.severity == "critical", a.anomaly.magnitude))
            plan.append((primary, group))
        return plan

    def _start_investigations(self, alerts: list[Alert], snapshot: FactorySnapshot) -> None:
        for primary, group in self._plan(alerts):
            if self.investigator is None:
                for alert in group:
                    self.alerts.finish_investigation(alert, success=True)
                continue
            for alert in group:
                self.alerts.mark_investigating(alert)
            extra = sorted({a.anomaly.item for a in group} - {primary.anomaly.item})
            self.out(f"{stamp()} Starting investigation of {primary.anomaly.item}..."
                     + (f" (also covers upstream: {', '.join(extra)})" if extra else ""))
            if self._synchronous:
                self._investigate(primary, group, snapshot)
            else:
                self._executor.submit(self._investigate, primary, group, snapshot)

    def _investigate(self, primary: Alert, group: list[Alert], snapshot: FactorySnapshot) -> None:
        try:
            report = self.investigator.investigate_anomaly(primary.anomaly, snapshot=snapshot)
        except Exception as exc:  # noqa: BLE001 - the monitor must survive any investigation failure
            for alert in group:
                self.alerts.finish_investigation(alert, success=False)
            self.out(f"{stamp()} Investigation of {primary.anomaly.item} failed: {exc}")
            return
        for alert in group:
            self.alerts.finish_investigation(alert, success=report.llm_ok)
        self.out(f"{stamp()} Investigation complete.\n\n{report.format()}")
        if not report.llm_ok:
            self.out(f"(LLM analysis unavailable; will retry in {self.cfg.investigation_retry_seconds}s "
                     "if the anomaly persists.)")

    # ------------------------------------------------------------------ loop
    def run_forever(self, sleep: Callable[[float], None] = time.sleep) -> None:
        self.out(f"{stamp()} Monitoring started (every {self.cfg.poll_interval}s). Ctrl+C to stop.")
        try:
            while True:
                started = time.monotonic()
                try:
                    self.run_cycle()
                except Exception as exc:  # noqa: BLE001
                    self.out(f"{stamp()} Monitoring cycle failed: {exc}")
                sleep(max(1.0, self.cfg.poll_interval - (time.monotonic() - started)))
        except KeyboardInterrupt:
            self.out(f"\n{stamp()} Monitoring stopped.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def wait(self) -> None:
        """Block until queued investigations finish (used by tests)."""
        if self._executor:
            self._executor.submit(lambda: None).result()
