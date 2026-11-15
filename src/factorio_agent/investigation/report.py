"""Investigation reports (design 6.3, 9.1) and the evidence/audit log (design 12)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from factorio_agent.investigation.evidence import Evidence
from factorio_agent.models.anomalies import AnomalyType

CHECKS = {
    AnomalyType.PRODUCTION_STOPPED.value: "Check power, ingredient supply, output blockage and whether machines/furnaces are still running.",
    AnomalyType.PRODUCTION_DROP.value: "Inspect the supply chain feeding it (ore patches, belts, smelter/assembler counts, power satisfaction).",
    AnomalyType.THROUGHPUT_DEGRADATION.value: "Look for slow depletion (ore patches), growing belt congestion, or added consumers.",
    AnomalyType.RESOURCE_SHORTAGE.value: "Demand exceeds supply: add production capacity or reduce competing consumers.",
}


@dataclass
class Report:
    subject: str
    evidence: Evidence
    analysis: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)

    @property
    def llm_ok(self) -> bool:
        return self.analysis is not None

    def findings(self) -> list[str]:
        """Deterministic findings - always available, even with the LLM down."""
        ev, out = self.evidence, []
        if ev.target and ev.target.production:
            p = ev.target.production
            ch = p.change_vs_short()
            out.append(f"{ev.subject} production: {p.current:.1f}/min now"
                       + (f" vs {p.short:.1f}/min baseline ({ch * 100:+.0f}%)" if ch is not None and p.short is not None else "")
                       if p.current is not None else f"{ev.subject}: no current production reading")
            if ev.target.drop_started_at:
                out.append("Decline began around " + time.strftime("%H:%M", time.localtime(ev.target.drop_started_at)))
        for c in ev.candidates[:3]:
            out += [f"{c.item}: {o}" for o in c.observations]
        if ev.subject == "factory":
            out += [a.message for a in ev.anomalies[:10]]
        return out

    def format(self) -> str:
        ev, lines = self.evidence, []
        lines.append(f"=== Investigation: {self.subject} (window {ev.window_minutes} min) ===")
        if ev.candidates:
            top = ev.candidates[0]
            label = "Likely bottleneck" if ev.subject != "factory" else "Most significant issue"
            lines.append(f"{label}: {top.item} ({top.kind.replace('_', ' ')})")
        if ev.confidence != "n/a":
            lines.append(f"Confidence: {ev.confidence}   Scope: {ev.scope}")
        facts = self.findings()
        if facts:
            lines += ["", "Evidence:"] + [f"- {f}" for f in facts]
        if ev.gaps:
            lines += ["", "Data gaps:"] + [f"- {g}" for g in ev.gaps]
        if ev.candidates:
            lines += ["", "Suggested checks (rule-based): " + CHECKS.get(ev.candidates[0].kind, "")]
        lines.append("")
        if self.analysis:
            lines += ["Analysis:", self.analysis]
        else:
            lines.append(f"LLM analysis unavailable: {self.error or 'unknown error'}. Showing rule-based findings only.")
        return "\n".join(lines)

    def to_record(self) -> dict:
        return {"created": self.created, "subject": self.subject, "analysis": self.analysis,
                "error": self.error, "evidence": self.evidence.to_dict()}


class ReportLog:
    """Append-only JSONL audit log of investigations and their evidence."""

    def __init__(self, directory: str):
        self.path = Path(directory) / "investigations.jsonl" if directory else None

    def write(self, report: Report) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(report.to_record()) + "\n")
        except OSError:
            pass  # logging must never break an investigation
