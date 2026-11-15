"""Metric discovery and schema resolution (design 7.3).

The agent never assumes metric names: it inspects the exporter, builds a
catalog, and resolves which metrics carry per-item production / consumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from factorio_agent.config import Config
from factorio_agent.tools.prometheus import PrometheusClient

INTERNAL_PREFIXES = ("go_", "process_", "prometheus_", "promhttp_", "scrape_", "net_", "up", "alertmanager_")
ITEM_LABEL_CANDIDATES = ("item", "item_name", "name", "product", "prototype", "type")


class SchemaError(Exception):
    """The configured/discovered metric schema is unusable."""


@dataclass
class MetricInfo:
    name: str
    help: str = ""
    type: str = ""
    labels: list[str] = field(default_factory=list)


@dataclass
class MetricSchema:
    production_metric: str
    consumption_metric: str
    item_label: str
    notes: list[str] = field(default_factory=list)


class MetricCatalog:
    def __init__(self, metrics: dict[str, MetricInfo]):
        self.metrics = metrics

    @classmethod
    def build(cls, prom: PrometheusClient, prefix: str = "") -> "MetricCatalog":
        names = [n for n in prom.list_metrics()
                 if (n.startswith(prefix) if prefix else not n.startswith(INTERNAL_PREFIXES))]
        meta = prom.metadata()
        metrics = {}
        for name in names:
            info = meta.get(name, {})
            metrics[name] = MetricInfo(name, info.get("help", ""), info.get("type", ""),
                                       prom.get_metric_labels(name))
        return cls(metrics)

    def find(self, *keywords: str, exclude: tuple[str, ...] = ()) -> list[MetricInfo]:
        return [m for m in self.metrics.values()
                if any(k in m.name.lower() for k in keywords)
                and not any(x in m.name.lower() for x in exclude)
                and any(label in ITEM_LABEL_CANDIDATES for label in m.labels)]


def resolve_schema(cfg: Config, catalog: MetricCatalog | None) -> MetricSchema:
    """Use explicit config where given; otherwise infer from the catalog."""
    notes: list[str] = []
    prod, cons, label = cfg.production_metric, cfg.consumption_metric, cfg.item_label

    if not (prod and cons and label):
        if catalog is None:
            raise SchemaError("Metric schema not configured and discovery is unavailable.")
        if not prod:
            prod = _pick(catalog.find("produc", exclude=("consum",)), "production")
            notes.append(f"production metric auto-detected: {prod}")
        if not cons:
            cons = _pick(catalog.find("consum"), "consumption")
            notes.append(f"consumption metric auto-detected: {cons}")
        if not label:
            labels = catalog.metrics[prod].labels if prod in catalog.metrics else []
            label = next((c for c in ITEM_LABEL_CANDIDATES if c in labels), "")
            if not label:
                raise SchemaError(f"Cannot infer item label for {prod}; labels: {labels}. Set item_label.")
            notes.append(f"item label auto-detected: {label}")
    return MetricSchema(prod, cons, label, notes)


def _pick(candidates: list[MetricInfo], what: str) -> str:
    if len(candidates) == 1:
        return candidates[0].name
    totals = [c for c in candidates if c.name.endswith("_total")]
    if len(totals) == 1:
        return totals[0].name
    names = [c.name for c in candidates] or "none found"
    raise SchemaError(
        f"Cannot uniquely identify the {what} metric (candidates: {names}). "
        f"Set FACTORIO_AGENT_{what.upper()}_METRIC (and FACTORIO_AGENT_ITEM_LABEL) or the matching "
        "keys in factorio_agent.toml. Run `python -m factorio_agent metrics` to see the exporter's metrics.")
