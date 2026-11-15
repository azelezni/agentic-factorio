"""Alert lifecycle and deduplication (design 5.4).

DETECTED -> INVESTIGATING -> ACTIVE -> RESOLVED.  An active anomaly is only
re-investigated on a meaningful change or after the re-investigation interval.
State is persisted so an agent restart does not re-announce active alerts.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from factorio_agent.config import Config
from factorio_agent.models.anomalies import Alert, AlertState, Anomaly


@dataclass
class UpdateResult:
    new: list[Alert] = field(default_factory=list)
    reinvestigate: list[Alert] = field(default_factory=list)
    resolved: list[Alert] = field(default_factory=list)

    @property
    def to_investigate(self) -> list[Alert]:
        return self.new + self.reinvestigate


class AlertManager:
    def __init__(self, cfg: Config, state_path: str | None = None, clock: Callable[[], float] = time.time):
        self.cfg, self.clock = cfg, clock
        self.state_path = Path(state_path) if state_path else None
        self._alerts: dict[str, Alert] = {}
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
            for raw in data.get("alerts", []):
                alert = Alert.from_dict(raw)
                if alert.state is not AlertState.RESOLVED:
                    # An interrupted investigation is simply retried later; nothing is re-announced.
                    alert.state = AlertState.ACTIVE
                    self._alerts[alert.key] = alert
        except (ValueError, KeyError, TypeError):
            pass  # corrupt state file: start clean rather than crash the monitor

    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.state_path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump({"alerts": [a.to_dict() for a in self._alerts.values()]}, fh, indent=2)
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------ lifecycle
    def update(self, anomalies: list[Anomaly], observed_items: set[str]) -> UpdateResult:
        """Fold one monitoring cycle into alert state.

        `observed_items` are items with trustworthy data this cycle; alerts for
        other items are left untouched (unknown is not "resolved").
        """
        now, result = self.clock(), UpdateResult()
        with self._lock:
            seen = set()
            for anomaly in anomalies:
                seen.add(anomaly.key)
                alert = self._alerts.get(anomaly.key)
                if alert is None:
                    alert = Alert(anomaly, AlertState.DETECTED, now, now)
                    self._alerts[anomaly.key] = alert
                    result.new.append(alert)
                    continue
                alert.anomaly, alert.last_seen, alert.clear_cycles = anomaly, now, 0
                if alert.state is AlertState.ACTIVE and self._due(alert, now):
                    result.reinvestigate.append(alert)

            for key, alert in list(self._alerts.items()):
                if key in seen or alert.anomaly.item not in observed_items:
                    continue
                alert.clear_cycles += 1
                if alert.clear_cycles >= self.cfg.resolve_after_cycles:
                    alert.state = AlertState.RESOLVED
                    result.resolved.append(alert)
                    del self._alerts[key]
            self._save()
        return result

    def _due(self, alert: Alert, now: float) -> bool:
        c = self.cfg
        if alert.last_investigated is None:        # previous attempt failed / never completed
            return now - (alert.last_attempt or alert.first_seen) >= c.investigation_retry_seconds
        if now - alert.last_investigated >= c.reinvestigate_seconds:
            return True
        return (alert.last_magnitude is not None
                and abs(alert.anomaly.magnitude - alert.last_magnitude) >= c.reinvestigate_change)

    def mark_investigating(self, alert: Alert) -> None:
        with self._lock:
            alert.state, alert.last_attempt = AlertState.INVESTIGATING, self.clock()
            self._save()

    def finish_investigation(self, alert: Alert, success: bool) -> None:
        with self._lock:
            alert.state = AlertState.ACTIVE
            if success:
                alert.last_investigated = self.clock()
                alert.last_magnitude = alert.anomaly.magnitude
            self._save()

    def active_alerts(self) -> list[Alert]:
        with self._lock:
            return list(self._alerts.values())
