"""Investigation engine (design 6)."""
from __future__ import annotations

import re

from factorio_agent.config import Config
from factorio_agent.investigation.evidence import Evidence, EvidenceBuilder
from factorio_agent.investigation.relations import RecipeGraph, normalize_item
from factorio_agent.investigation.report import Report, ReportLog
from factorio_agent.llm.client import LLMClient, LLMError
from factorio_agent.llm.prompts import build_investigation_messages
from factorio_agent.models.anomalies import Anomaly
from factorio_agent.models.metrics import DataStatus, FactorySnapshot

_WINDOW_RE = re.compile(r"(\d+)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)\b", re.I)
FACTORY_WORDS = ("bottleneck", "stalled", "stall", "limiting", "factory", "production", "resource",
                 "shortage", "starved", "dropping", "dropped", "throughput")


class TelemetryUnavailable(Exception):
    pass


def parse_window(text: str, default: int) -> int:
    """'1h' / '30 minutes' / 'last hour' -> minutes."""
    m = _WINDOW_RE.search(text)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        return max(1, n * 60 if unit.startswith("h") else n)
    low = text.lower()
    if "last hour" in low or "past hour" in low:
        return 60
    return default


class Investigator:
    def __init__(self, telemetry, llm: LLMClient | None, cfg: Config,
                 graph: RecipeGraph | None = None, log: ReportLog | None = None):
        self.telemetry, self.llm, self.cfg = telemetry, llm, cfg
        self.graph = graph or RecipeGraph()
        self.builder = EvidenceBuilder(telemetry, self.graph, cfg)
        self.log = log or ReportLog(cfg.log_dir)

    # ------------------------------------------------------------------ public API
    def investigate_item(self, item: str, window_minutes: int | None = None,
                         snapshot: FactorySnapshot | None = None, question: str | None = None) -> Report:
        window = window_minutes or 60
        snap = self._snapshot(snapshot)
        return self._finish(self.builder.for_item(item, window, snap), question)

    def investigate_factory(self, window_minutes: int | None = None,
                            snapshot: FactorySnapshot | None = None, question: str | None = None) -> Report:
        window = window_minutes or 60
        snap = self._snapshot(snapshot)
        return self._finish(self.builder.for_factory(window, snap), question)

    def investigate_anomaly(self, anomaly: Anomaly, snapshot: FactorySnapshot | None = None) -> Report:
        return self.investigate_item(anomaly.item, 60, snapshot)

    def ask(self, question: str) -> Report:
        """Interpret a natural-language request deterministically and investigate."""
        window = parse_window(question, 60)
        item = self.match_item(question)
        if item:
            return self.investigate_item(item, window, question=question)
        return self.investigate_factory(window, question=question)

    # ------------------------------------------------------------------ intent helpers
    def known_items(self) -> set[str]:
        items = set(self.graph.known_items())
        try:
            items |= set(self.telemetry.items())
        except Exception:  # noqa: BLE001
            pass
        return items

    def match_item(self, text: str) -> str | None:
        low = text.lower()
        best: tuple[int, str] | None = None
        for item in self.known_items():
            for variant in {item, item.replace("-", " "), item.replace("-", "")}:
                pos = low.find(variant)
                if pos >= 0 and (best is None or pos < best[0] or (pos == best[0] and len(item) > len(best[1]))):
                    best = (pos, item)
        return best[1] if best else None

    def is_factory_question(self, text: str) -> bool:
        low = text.lower()
        return self.match_item(text) is not None or any(w in low for w in FACTORY_WORDS)

    # ------------------------------------------------------------------ internals
    def _snapshot(self, snapshot: FactorySnapshot | None) -> FactorySnapshot:
        snap = snapshot or self.telemetry.snapshot()
        if not snap.ok:
            raise TelemetryUnavailable(f"Telemetry {snap.status.value}: {snap.detail}")
        return snap

    def _finish(self, evidence: Evidence, question: str | None) -> Report:
        report = Report(evidence.subject, evidence)
        if self.llm is None:
            report.error = "no LLM configured"
        else:
            try:
                report.analysis = self.llm.generate(build_investigation_messages(evidence.to_text(), question))
            except LLMError as exc:
                report.error = str(exc)
        self.log.write(report)
        return report
