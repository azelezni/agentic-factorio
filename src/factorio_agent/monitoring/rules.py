"""Deterministic, configurable health rules (design 5.2, 5.3).

Rules only ever see *fresh* items with real values; missing/None values are
skipped, never treated as zero (design 11).
"""
from __future__ import annotations

from factorio_agent.config import Config
from factorio_agent.models.anomalies import Anomaly, AnomalyType
from factorio_agent.models.metrics import FactorySnapshot, ItemSnapshot

EPS = 0.01  # items/min treated as "zero"


def _pct(x: float) -> str:
    return f"{x * 100:+.0f}%"


def evaluate_item(snap: ItemSnapshot, cfg: Config) -> list[Anomaly]:
    if not snap.fresh:
        return []
    out: list[Anomaly] = []
    item, p, c = snap.item, snap.production, snap.consumption

    if p is not None:
        baseline = max(v for v in (p.short, p.long) if v is not None) if (p.short is not None or p.long is not None) else None
        stopped = (baseline is not None and baseline >= cfg.min_rate
                   and p.recent is not None and p.recent <= EPS
                   and p.current is not None and p.current <= EPS)
        if stopped:
            out.append(Anomaly(AnomalyType.PRODUCTION_STOPPED, item,
                               f"{item} production stopped (0/min for {cfg.stopped_minutes} min, baseline {baseline:.1f}/min)",
                               p.current, baseline, -1.0, 1.0, "critical"))
        else:
            ch, ch_long = p.change_vs_short(), p.change_vs_long()
            if ch is not None and ch <= -cfg.drop_threshold and (p.short or 0) >= cfg.min_rate:
                out.append(Anomaly(AnomalyType.PRODUCTION_DROP, item,
                                   f"{item} production {_pct(ch)} vs {cfg.short_window_minutes}-min baseline "
                                   f"({p.current:.1f}/min vs {p.short:.1f}/min)",
                                   p.current, p.short, ch, abs(ch), "critical" if ch <= -0.6 else "warning"))
            elif ch_long is not None and ch_long <= -cfg.degradation_threshold and (p.long or 0) >= cfg.min_rate:
                out.append(Anomaly(AnomalyType.THROUGHPUT_DEGRADATION, item,
                                   f"{item} throughput {_pct(ch_long)} vs {cfg.long_window_minutes}-min baseline "
                                   f"({p.current:.1f}/min vs {p.long:.1f}/min)",
                                   p.current, p.long, ch_long, abs(ch_long), "warning"))
        if (not stopped and p.cv is not None and p.cv > cfg.instability_cv
                and (p.short or 0) >= cfg.min_rate):
            out.append(Anomaly(AnomalyType.UNSTABLE_PRODUCTION, item,
                               f"{item} production is unstable (coefficient of variation {p.cv:.2f})",
                               p.current, p.short, None, p.cv, "warning", {"cv": round(p.cv, 3)}))

    if c is not None:
        ch = c.change_vs_short()
        if ch is not None and ch <= -cfg.consumption_drop_threshold and (c.short or 0) >= cfg.min_rate:
            out.append(Anomaly(AnomalyType.CONSUMPTION_DROP, item,
                               f"{item} consumption {_pct(ch)} vs {cfg.short_window_minutes}-min baseline "
                               f"({c.current:.1f}/min vs {c.short:.1f}/min)",
                               c.current, c.short, ch, abs(ch), "warning"))
        if (p is not None and c.short is not None and p.short is not None
                and c.short >= cfg.min_rate and c.short > p.short * cfg.shortage_ratio):
            ratio = c.short / max(p.short, EPS)
            out.append(Anomaly(AnomalyType.RESOURCE_SHORTAGE, item,
                               f"{item} consumption ({c.short:.1f}/min) exceeds production ({p.short:.1f}/min)",
                               p.short, c.short, None, min(ratio, 99.0), "warning",
                               {"consumption": round(c.short, 2), "production": round(p.short, 2)}))
    return out


def evaluate_all(snapshot: FactorySnapshot, cfg: Config) -> list[Anomaly]:
    if not snapshot.ok:
        return []
    anomalies = []
    for name in sorted(snapshot.items):
        anomalies.extend(evaluate_item(snapshot.items[name], cfg))
    return anomalies
