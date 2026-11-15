"""Factory-level telemetry on top of the raw Prometheus client.

Turns the exporter's per-item counters into per-minute production/consumption
rates over the configured windows, while keeping "valid zero", "missing",
"stale" and "failed" distinct (design 11).
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

from factorio_agent.config import Config
from factorio_agent.models.metrics import (DataStatus, FactorySnapshot, FlowStats,
                                           ItemSnapshot, Sample)
from factorio_agent.tools.catalog import MetricSchema
from factorio_agent.tools.prometheus import PrometheusClient, PrometheusError


@dataclass
class TelemetryHealth:
    status: DataStatus
    detail: str = ""
    age_seconds: float | None = None


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class FactoryTelemetry:
    def __init__(self, prom: PrometheusClient, cfg: Config, schema: MetricSchema,
                 clock: Callable[[], float] = time.time):
        self.prom, self.cfg, self.schema, self.clock = prom, cfg, schema, clock

    # ------------------------------------------------------------------ expressions
    def _selector(self, metric: str, item: str | None = None) -> str:
        if item is None:
            return metric
        return f'{metric}{{{self.schema.item_label}="{_quote(item)}"}}'

    def _rate_expr(self, metric: str, window_min: int, offset_min: int = 0, item: str | None = None) -> str:
        """Per-item items/minute over `window_min`, ending `offset_min` ago."""
        sel = self._selector(metric, item)
        off = f" offset {offset_min}m" if offset_min else ""
        by = self.schema.item_label
        if self.cfg.metric_kind == "gauge":
            return f"sum by ({by}) (avg_over_time({sel}[{window_min}m]{off})) * {self.cfg.gauge_scale}"
        return f"sum by ({by}) (rate({sel}[{window_min}m]{off})) * 60"

    def _step_expr(self, metric: str, item: str | None = None) -> str:
        sel = self._selector(metric, item)
        by = self.schema.item_label
        if self.cfg.metric_kind == "gauge":
            return f"sum by ({by}) ({sel}) * {self.cfg.gauge_scale}"
        return f"sum by ({by}) (rate({sel}[2m])) * 60"

    def _instant_map(self, expr: str) -> dict[str, float]:
        out = {}
        for s in self.prom.query_instant(expr):
            name = s.labels.get(self.schema.item_label)
            if name is not None and s.latest is not None:
                out[name] = s.latest.value
        return out

    # ------------------------------------------------------------------ health
    def health(self) -> TelemetryHealth:
        c = self.cfg
        try:
            if not self.prom.is_ready():
                return TelemetryHealth(DataStatus.FAILED, f"Prometheus not ready at {self.prom.base_url}")
            if c.bridge_job:
                up = self.prom.query_instant(f'up{{job="{_quote(c.bridge_job)}"}}')
                if up and all(s.latest.value == 0 for s in up):
                    return TelemetryHealth(DataStatus.FAILED,
                                           f"Prometheus cannot scrape the Wiretap bridge (job '{c.bridge_job}' is down)")
            res = self.prom.query_instant(f"time() - max(timestamp({self.schema.production_metric}))")
        except PrometheusError as exc:
            return TelemetryHealth(DataStatus.FAILED, str(exc))
        if not res:
            return TelemetryHealth(DataStatus.MISSING,
                                   f"No series for {self.schema.production_metric} - telemetry is missing "
                                   "(this is NOT zero production)")
        age = res[0].latest.value
        if age > c.stale_seconds:
            return TelemetryHealth(DataStatus.STALE,
                                   f"{self.schema.production_metric} last updated {age:.0f}s ago", age)
        return TelemetryHealth(DataStatus.OK, "", age)

    # ------------------------------------------------------------------ snapshot
    def items(self) -> list[str]:
        names: set[str] = set()
        for metric in (self.schema.production_metric, self.schema.consumption_metric):
            names.update(self.prom.label_values(self.schema.item_label, metric))
        return sorted(names)

    def _collect_flow(self, metric: str) -> tuple[dict[str, FlowStats], dict[str, float]]:
        c = self.cfg
        cur = self._instant_map(self._rate_expr(metric, c.current_window_minutes))
        short = self._instant_map(self._rate_expr(metric, c.short_window_minutes, c.current_window_minutes))
        long = self._instant_map(self._rate_expr(metric, c.long_window_minutes, c.current_window_minutes))
        recent = self._instant_map(self._rate_expr(metric, c.stopped_minutes))
        ages = self._instant_map(f"time() - max by ({self.schema.item_label}) (timestamp({metric}))")

        cvs: dict[str, float] = {}
        end = self.clock()
        for s in self.prom.query_range(self._step_expr(metric), str(end - c.short_window_minutes * 60),
                                       str(end), "60s"):
            name = s.labels.get(self.schema.item_label)
            values = [x.value for x in s.samples]
            if name and len(values) >= 5 and statistics.fmean(values) > 0:
                cvs[name] = statistics.pstdev(values) / statistics.fmean(values)

        stats = {}
        for name in set(cur) | set(short) | set(long) | set(recent) | set(ages):
            stats[name] = FlowStats(cur.get(name), short.get(name), long.get(name),
                                    recent.get(name), cvs.get(name))
        return stats, ages

    def snapshot(self) -> FactorySnapshot:
        now = self.clock()
        health = self.health()
        if health.status is not DataStatus.OK:
            return FactorySnapshot(now, health.status, health.detail)
        try:
            prod, prod_age = self._collect_flow(self.schema.production_metric)
            cons, cons_age = self._collect_flow(self.schema.consumption_metric)
        except PrometheusError as exc:
            return FactorySnapshot(now, DataStatus.FAILED, str(exc))

        items = {}
        for name in set(prod) | set(cons):
            age = prod_age.get(name, cons_age.get(name))
            items[name] = ItemSnapshot(name, prod.get(name), cons.get(name),
                                       fresh=age is not None and age <= self.cfg.stale_seconds,
                                       age_seconds=age)
        return FactorySnapshot(now, DataStatus.OK, "", items)

    # ------------------------------------------------------------------ history
    def history(self, flow: str, item: str, minutes: int) -> list[Sample]:
        """Per-minute rate history of `flow` ("production" | "consumption") for one item."""
        metric = self.schema.production_metric if flow == "production" else self.schema.consumption_metric
        end = self.clock()
        series = self.prom.query_range(self._step_expr(metric, item), str(end - minutes * 60), str(end), "60s")
        return series[0].samples if series else []
