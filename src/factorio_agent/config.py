"""Configuration: defaults < TOML file < FACTORIO_AGENT_<NAME> environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


@dataclass
class Config:
    # --- LLM provider selection ---
    llm_provider: str = "ollama"     # "ollama" (local) or "anthropic" (Claude API)
    llm_fallback: str = ""           # optional second provider used if the first is unavailable

    # --- Anthropic API (key is read from the ANTHROPIC_API_KEY environment variable, never the config) ---
    anthropic_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_max_tokens: int = 1024

    # --- Ollama (design 8.1) ---
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    num_ctx: int = 8192
    keep_alive: str = "5m"
    llm_timeout: float = 180.0
    llm_retries: int = 1          # retries on malformed responses
    think: bool = False           # Qwen3 "thinking" mode; off for latency
    temperature: float = 0.2

    # --- Prometheus (design 7) ---
    prometheus_url: str = "http://localhost:9090"
    prometheus_timeout: float = 10.0
    prometheus_max_bytes: int = 8_000_000

    # --- Metric schema. Empty strings => auto-discover from the actual exporter (7.3) ---
    production_metric: str = ""
    consumption_metric: str = ""
    item_label: str = ""
    metric_kind: str = "counter"   # "counter" (cumulative) or "gauge" (already a rate)
    gauge_scale: float = 1.0       # multiplier to convert gauge values to items/minute
    metric_prefix: str = ""        # restrict discovery to metrics with this prefix
    bridge_job: str = "wiretap"    # Prometheus job name of the Wiretap bridge (for the `up` check)

    # --- Monitoring (design 4.1, 5.3) ---
    poll_interval: int = 60
    current_window_minutes: int = 5
    short_window_minutes: int = 30
    long_window_minutes: int = 120
    stopped_minutes: int = 10
    drop_threshold: float = 0.30
    degradation_threshold: float = 0.30
    consumption_drop_threshold: float = 0.30
    instability_cv: float = 0.6
    shortage_ratio: float = 1.15
    min_rate: float = 5.0          # items/min; ignore anything slower than this
    stale_seconds: int = 180

    # --- Alert lifecycle (design 5.4) ---
    reinvestigate_seconds: int = 900
    reinvestigate_change: float = 0.15
    investigation_retry_seconds: int = 300
    resolve_after_cycles: int = 3

    # --- Persistence ---
    state_file: str = "data/alerts.json"
    log_dir: str = "data/investigations"
    recipes_file: str = ""         # optional JSON overriding the built-in recipe table

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        known = {f.name for f in fields(cls)}
        path = path or os.environ.get("FACTORIO_AGENT_CONFIG")
        if not path and Path("factorio_agent.toml").exists():
            path = "factorio_agent.toml"
        if path:
            if tomllib is None:
                raise RuntimeError("TOML config files require Python 3.11+; use environment variables instead.")
            with open(path, "rb") as fh:
                for key, value in tomllib.load(fh).items():
                    if key not in known:
                        raise ValueError(f"Unknown config key in {path}: {key}")
                    setattr(cfg, key, _coerce(value, getattr(cfg, key)))
        for name in known:
            env = os.environ.get("FACTORIO_AGENT_" + name.upper())
            if env is not None:
                setattr(cfg, name, _coerce(env, getattr(cfg, name)))
        return cfg


def _coerce(value, default):
    if isinstance(default, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return str(value)
