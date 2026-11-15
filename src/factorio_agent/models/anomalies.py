"""Anomaly and alert models (design 5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class AnomalyType(str, Enum):
    PRODUCTION_DROP = "production_drop"
    PRODUCTION_STOPPED = "production_stopped"
    RESOURCE_SHORTAGE = "resource_shortage"
    CONSUMPTION_DROP = "consumption_drop"
    UNSTABLE_PRODUCTION = "unstable_production"
    THROUGHPUT_DEGRADATION = "throughput_degradation"


class AlertState(str, Enum):
    DETECTED = "DETECTED"
    INVESTIGATING = "INVESTIGATING"
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"


@dataclass
class Anomaly:
    type: AnomalyType
    item: str
    message: str
    current: float | None = None
    baseline: float | None = None
    change: float | None = None        # fractional change, e.g. -0.42
    magnitude: float = 0.0             # type-specific size used to detect "meaningful change"
    severity: str = "warning"          # "warning" | "critical"
    details: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.type.value}:{self.item}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Anomaly":
        d = dict(d)
        d["type"] = AnomalyType(d["type"])
        return cls(**d)


@dataclass
class Alert:
    anomaly: Anomaly
    state: AlertState = AlertState.DETECTED
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_attempt: float | None = None        # last time an investigation was started
    last_investigated: float | None = None   # last time one completed successfully
    last_magnitude: float | None = None      # anomaly magnitude at last investigation
    clear_cycles: int = 0

    @property
    def key(self) -> str:
        return self.anomaly.key

    def to_dict(self) -> dict:
        d = asdict(self)
        d["anomaly"] = self.anomaly.to_dict()
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Alert":
        d = dict(d)
        d["anomaly"] = Anomaly.from_dict(d["anomaly"])
        d["state"] = AlertState(d["state"])
        return cls(**d)
