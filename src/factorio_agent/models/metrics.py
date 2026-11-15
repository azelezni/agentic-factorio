"""Normalized metric data structures.

Every value the agent reasons about carries enough context to tell apart:
a valid zero, a missing series, a stale metric and a failed query (design 11).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class DataStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"   # no matching time series
    STALE = "stale"       # series exists but has stopped updating
    FAILED = "failed"     # query / service failure


@dataclass(frozen=True)
class Sample:
    timestamp: float
    value: float


@dataclass
class Series:
    labels: dict[str, str]
    samples: list[Sample]

    @property
    def latest(self) -> Sample | None:
        return self.samples[-1] if self.samples else None


def relative_change(current: float | None, baseline: float | None) -> float | None:
    """(current - baseline) / baseline, or None when it cannot be computed."""
    if current is None or baseline is None or baseline <= 0:
        return None
    return (current - baseline) / baseline


@dataclass
class FlowStats:
    """Per-minute rates of one flow (production or consumption) of one item."""
    current: float | None = None   # last `current_window_minutes`
    short: float | None = None     # short baseline (ends where `current` starts)
    long: float | None = None      # long baseline
    recent: float | None = None    # rate over the "stopped" interval
    cv: float | None = None        # coefficient of variation of the rate (instability)

    def change_vs_short(self) -> float | None:
        return relative_change(self.current, self.short)

    def change_vs_long(self) -> float | None:
        return relative_change(self.current, self.long)

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class ItemSnapshot:
    item: str
    production: FlowStats | None = None
    consumption: FlowStats | None = None
    fresh: bool = True
    age_seconds: float | None = None


@dataclass
class FactorySnapshot:
    timestamp: float
    status: DataStatus = DataStatus.OK
    detail: str = ""
    items: dict[str, ItemSnapshot] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is DataStatus.OK
