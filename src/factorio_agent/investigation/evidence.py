"""Structured evidence packages (design 6.1, 6.3).

All numbers are computed deterministically here; the LLM only interprets them.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from factorio_agent.config import Config
from factorio_agent.models.anomalies import Anomaly, AnomalyType
from factorio_agent.models.metrics import FactorySnapshot, FlowStats, ItemSnapshot, Sample
from factorio_agent.monitoring import rules
from factorio_agent.investigation.relations import RecipeGraph, normalize_item

PROBLEM_TYPES = {AnomalyType.PRODUCTION_STOPPED, AnomalyType.PRODUCTION_DROP,
                 AnomalyType.THROUGHPUT_DEGRADATION}
ASSUMPTION = "Input relationships use assumed vanilla recipes (telemetry does not expose recipes)."


@dataclass
class ItemEvidence:
    item: str
    role: str                      # target | input | competitor | concern
    production: FlowStats | None = None
    consumption: FlowStats | None = None
    depth: int = 0                 # upstream distance from the target
    note: str = ""
    drop_started_at: float | None = None

    def to_dict(self) -> dict:
        return {"item": self.item, "role": self.role, "depth": self.depth, "note": self.note,
                "production": self.production.to_dict() if self.production else None,
                "consumption": self.consumption.to_dict() if self.consumption else None,
                "drop_started_at": self.drop_started_at}


@dataclass
class Candidate:
    item: str
    depth: int
    kind: str                      # AnomalyType value
    magnitude: float
    observations: list[str]
    drop_started_at: float | None = None

    def to_dict(self) -> dict:
        return {"item": self.item, "depth": self.depth, "kind": self.kind,
                "magnitude": round(self.magnitude, 3), "observations": self.observations,
                "drop_started_at": self.drop_started_at}


@dataclass
class Evidence:
    subject: str                   # item name or "factory"
    window_minutes: int
    generated_at: float
    target: ItemEvidence | None = None
    inputs: list[ItemEvidence] = field(default_factory=list)
    competitors: list[ItemEvidence] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    scope: str = "unknown"         # isolated | partial | widespread | unknown
    confidence: str = "n/a"
    gaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"subject": self.subject, "window_minutes": self.window_minutes,
                "generated_at": self.generated_at,
                "target": self.target.to_dict() if self.target else None,
                "inputs": [i.to_dict() for i in self.inputs],
                "competitors": [c.to_dict() for c in self.competitors],
                "anomalies": [a.to_dict() for a in self.anomalies],
                "candidates": [c.to_dict() for c in self.candidates],
                "scope": self.scope, "confidence": self.confidence,
                "gaps": self.gaps, "notes": self.notes}

    def to_text(self) -> str:
        """Compact, LLM-friendly rendering (kept small for an 8k context)."""
        L = [f"Subject: {self.subject} | analysis window: {self.window_minutes} min | "
             f"scope: {self.scope} | rule-based confidence: {self.confidence}"]

        def row(e: ItemEvidence) -> str:
            parts = [f"- {e.item} [{e.role}{f', upstream depth {e.depth}' if e.depth else ''}]"]
            if e.production:
                p = e.production
                parts.append(f"production now {_n(p.current)}/min, {_n(p.short)} (30m baseline), {_n(p.long)} (long baseline)"
                             f" [{_pct(p.change_vs_short())} vs short]")
            if e.consumption:
                c = e.consumption
                parts.append(f"consumption now {_n(c.current)}/min, {_n(c.short)} (30m baseline)"
                             f" [{_pct(c.change_vs_short())} vs short]")
            if e.drop_started_at:
                parts.append(f"decline began ~{_clock(e.drop_started_at)}")
            if e.note:
                parts.append(e.note)
            return "; ".join(parts)

        if self.target:
            L += ["", "TARGET:", row(self.target)]
        if self.inputs:
            L += ["", "UPSTREAM INPUTS:"] + [row(i) for i in self.inputs[:10]]
        if self.competitors:
            L += ["", "OTHER CONSUMERS OF THE SAME INPUTS:"] + [row(c) for c in self.competitors[:6]]
        if self.candidates:
            L += ["", "BOTTLENECK CANDIDATES (ranked, rule-based):"]
            for i, c in enumerate(self.candidates[:4], 1):
                L.append(f"{i}. {c.item} ({c.kind}): " + " ".join(c.observations))
        if self.anomalies:
            L += ["", "ACTIVE ANOMALIES FACTORY-WIDE:"] + [f"- {a.message}" for a in self.anomalies[:12]]
        if self.gaps:
            L += ["", "DATA GAPS:"] + [f"- {g}" for g in self.gaps]
        if self.notes:
            L += ["", "NOTES:"] + [f"- {n}" for n in self.notes]
        return "\n".join(L)


def _n(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:+.0f}%"


def _clock(ts: float) -> str:
    import time
    return time.strftime("%H:%M", time.localtime(ts))


def find_decline_start(samples: list[Sample], drop_threshold: float) -> float | None:
    """Timestamp where the trailing low-rate run began, or None if not currently declined.

    Baseline = median of the first half of the window; "low" = below (1 - threshold) * baseline.
    """
    if len(samples) < 6:
        return None
    half = samples[: len(samples) // 2]
    baseline = statistics.median(s.value for s in half)
    if baseline <= 0:
        return None
    level = baseline * (1 - drop_threshold)
    start = None
    for s in reversed(samples):
        if s.value < level:
            start = s.timestamp
        else:
            break
    if start is None or (samples[-1].timestamp - start) < 60:
        return None
    return start


class EvidenceBuilder:
    def __init__(self, telemetry, graph: RecipeGraph, cfg: Config, max_depth: int = 3):
        self.telemetry, self.graph, self.cfg, self.max_depth = telemetry, graph, cfg, max_depth

    # ------------------------------------------------------------------ helpers
    def _item_evidence(self, snap: ItemSnapshot, role: str, depth: int = 0, note: str = "") -> ItemEvidence:
        return ItemEvidence(snap.item, role, snap.production, snap.consumption, depth, note)

    def _decline_start(self, item: str, window: int) -> float | None:
        try:
            return find_decline_start(self.telemetry.history("production", item, window),
                                      self.cfg.drop_threshold)
        except Exception:  # noqa: BLE001 - history is a nice-to-have, never fatal
            return None

    def _candidate(self, snap: ItemSnapshot, depth: int) -> Candidate | None:
        anoms = [a for a in rules.evaluate_item(snap, self.cfg)
                 if a.type in PROBLEM_TYPES or a.type is AnomalyType.RESOURCE_SHORTAGE]
        if not anoms:
            return None
        anoms.sort(key=lambda a: (a.type not in PROBLEM_TYPES, -a.magnitude))
        top = anoms[0]
        return Candidate(snap.item, depth, top.type.value, top.magnitude, [a.message + "." for a in anoms])

    # ------------------------------------------------------------------ item evidence
    def for_item(self, item: str, window_minutes: int, snapshot: FactorySnapshot) -> Evidence:
        item = normalize_item(item)
        cfg = self.cfg
        ev = Evidence(item, window_minutes, snapshot.timestamp, notes=[ASSUMPTION])
        ev.anomalies = rules.evaluate_all(snapshot, cfg)

        snap = snapshot.items.get(item)
        if snap is None:
            ev.gaps.append(f"No telemetry for '{item}' (not produced/consumed yet, or the name differs).")
        else:
            ev.target = self._item_evidence(snap, "target")
            if not snap.fresh:
                ev.gaps.append(f"Telemetry for '{item}' is stale ({snap.age_seconds or 0:.0f}s old).")
            ev.target.drop_started_at = self._decline_start(item, window_minutes)

        chain = self.graph.upstream(item, self.max_depth)
        if not chain:
            ev.gaps.append(f"No known inputs for '{item}' in the recipe table; upstream analysis skipped.")
        candidates: list[Candidate] = []
        for name, depth in chain:
            s = snapshot.items.get(name)
            if s is None:
                if depth == 1:
                    ev.gaps.append(f"No telemetry for input '{name}'.")
                continue
            ev.inputs.append(self._item_evidence(s, "input", depth))
            cand = self._candidate(s, depth) if s.fresh else None
            if cand:
                candidates.append(cand)
        candidates.sort(key=lambda c: (c.kind == AnomalyType.RESOURCE_SHORTAGE.value, -c.depth, -c.magnitude))
        ev.candidates = candidates

        # Resource competition: other consumers of the direct inputs and of the top bottleneck candidates.
        contested = self.graph.inputs(item) + [c.item for c in candidates[:2]]
        for inp in dict.fromkeys(contested):
            for other in self.graph.consumers(inp):
                s = snapshot.items.get(other)
                if other != item and s is not None and all(c.item != other for c in ev.competitors):
                    ev.competitors.append(self._item_evidence(s, "competitor", 0, f"also consumes {inp}"))

        # Isolated or part of a bigger problem?
        related = {item} | {n for n, _ in chain} | {n for n, _ in self.graph.downstream(item, self.max_depth)}
        others = {a.item for a in ev.anomalies if a.type in PROBLEM_TYPES and a.item not in related}
        ev.scope = "widespread" if len(others) >= 3 else "partial" if others else "isolated"
        if others:
            ev.notes.append("Other items with production problems: " + ", ".join(sorted(others)[:8]) + ".")

        if ev.target and not candidates and ev.inputs and any(
                a.item == item and a.type in PROBLEM_TYPES for a in ev.anomalies):
            ev.notes.append("None of its upstream inputs show production problems or shortages, so the limit "
                            "is likely at this item's own production step (machines, power, output blockage) "
                            "- the telemetry cannot confirm which.")
        self._confidence(ev)
        return ev

    def _confidence(self, ev: Evidence) -> None:
        tgt = ev.target
        if not ev.candidates:
            ev.confidence = "Low"
            return
        top = ev.candidates[0]
        top.drop_started_at = self._decline_start(top.item, ev.window_minutes)
        points = 1
        if top.kind != AnomalyType.RESOURCE_SHORTAGE.value:
            points += 1                                        # the input itself fell
        if tgt and tgt.drop_started_at and top.drop_started_at and top.drop_started_at <= tgt.drop_started_at + 120:
            points += 1                                        # input fell first (or together)
        demand = next((i for i in ev.inputs if i.item == top.item), None)
        if demand and demand.consumption and demand.consumption.current is not None \
                and demand.consumption.short and demand.consumption.current >= 0.8 * demand.consumption.short:
            points += 1                                        # demand for the input stayed high
        if ev.gaps and points > 2:
            points = 2
        ev.confidence = "High" if points >= 4 else "Medium" if points >= 2 else "Low"

    # ------------------------------------------------------------------ factory evidence
    def for_factory(self, window_minutes: int, snapshot: FactorySnapshot) -> Evidence:
        ev = Evidence("factory", window_minutes, snapshot.timestamp, notes=[ASSUMPTION])
        ev.anomalies = rules.evaluate_all(snapshot, self.cfg)
        weight = {AnomalyType.PRODUCTION_STOPPED: 3, AnomalyType.PRODUCTION_DROP: 2,
                  AnomalyType.THROUGHPUT_DEGRADATION: 2, AnomalyType.RESOURCE_SHORTAGE: 1}
        ranked = sorted(ev.anomalies, key=lambda a: (-weight.get(a.type, 0), -a.magnitude))
        seen: set[str] = set()
        for a in ranked:
            if a.item not in seen and len(seen) < 8:
                seen.add(a.item)
                ev.competitors.append(self._item_evidence(snapshot.items[a.item], "concern", 0, a.type.value))
        for item in sorted(seen):
            snap = snapshot.items[item]
            cand = self._candidate(snap, 0)
            if cand:
                ev.candidates.append(cand)
        ev.candidates.sort(key=lambda c: (c.kind == AnomalyType.RESOURCE_SHORTAGE.value, -c.magnitude))
        ev.scope = "widespread" if len({a.item for a in ev.anomalies if a.type in PROBLEM_TYPES}) >= 3 \
            else "partial" if ev.anomalies else "isolated"
        if not ev.anomalies:
            ev.notes.append("No anomalies detected by the configured rules.")
        return ev
