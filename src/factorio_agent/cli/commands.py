"""Command-line interface (design 9)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

from factorio_agent.config import Config
from factorio_agent.investigation.engine import Investigator, TelemetryUnavailable, parse_window
from factorio_agent.investigation.relations import RecipeGraph, normalize_item
from factorio_agent.llm.client import LLMError, OllamaClient
from factorio_agent.llm.factory import create_llm
from factorio_agent.llm.prompts import CHAT_SYSTEM
from factorio_agent.monitoring.alert_manager import AlertManager
from factorio_agent.monitoring.engine import MonitoringEngine
from factorio_agent.tools.catalog import MetricCatalog, SchemaError, resolve_schema
from factorio_agent.tools.factory import FactoryTelemetry
from factorio_agent.tools.prometheus import PrometheusClient, PrometheusError


PROVIDERS = ("ollama", "anthropic")


def describe_backend(cfg: Config) -> str:
    model = cfg.anthropic_model if cfg.llm_provider == "anthropic" else cfg.model
    text = f"{cfg.llm_provider} ({model})"
    if cfg.llm_fallback and cfg.llm_fallback != cfg.llm_provider:
        text += f", fallback: {cfg.llm_fallback}"
    return text


def backend_command(rt: "Runtime", arg: str) -> str:
    """Handle `/backend [name [model]]` in chat."""
    parts = arg.split()
    if not parts:
        return (f"Current backend: {describe_backend(rt.cfg)}\n"
                f"Switch with /backend <{'|'.join(PROVIDERS)}> [model]")
    try:
        return "Switched to " + rt.set_backend(parts[0].lower(), parts[1] if len(parts) > 1 else None)
    except ValueError as exc:
        return str(exc)


class Runtime:
    """Wires up clients; schema resolution failures are kept, not fatal, so chat/status still work."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm = create_llm(cfg)
        self.prom = PrometheusClient(cfg.prometheus_url, cfg.prometheus_timeout, cfg.prometheus_max_bytes)
        self.schema = None
        self.schema_error: str | None = None
        self.telemetry = None
        self.investigator = None
        self._resolve()

    def set_backend(self, provider: str, model: str | None = None) -> str:
        """Switch the LLM backend at runtime (investigator and chat pick it up immediately)."""
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown backend '{provider}'. Choose from: {', '.join(PROVIDERS)}")
        new_cfg = replace(self.cfg, llm_provider=provider)
        if model:
            setattr(new_cfg, "anthropic_model" if provider == "anthropic" else "model", model)
        llm = create_llm(new_cfg)           # validate before committing
        self.cfg, self.llm = new_cfg, llm
        if self.investigator is not None:
            self.investigator.llm = llm
        return describe_backend(new_cfg)

    def _resolve(self) -> None:
        try:
            catalog = None
            if not (self.cfg.production_metric and self.cfg.consumption_metric and self.cfg.item_label):
                catalog = MetricCatalog.build(self.prom, self.cfg.metric_prefix)
            self.schema = resolve_schema(self.cfg, catalog)
        except (SchemaError, PrometheusError) as exc:
            self.schema_error = str(exc)
            return
        self.telemetry = FactoryTelemetry(self.prom, self.cfg, self.schema)
        self.investigator = Investigator(self.telemetry, self.llm, self.cfg, RecipeGraph.load(self.cfg.recipes_file))


def trim_history(messages: list[dict], max_chars: int) -> list[dict]:
    """Keep the system prompt plus the newest messages that fit a rough character budget."""
    system, rest = messages[:1], messages[1:]
    kept, total = [], len(system[0]["content"]) if system else 0
    for msg in reversed(rest):
        total += len(msg["content"])
        if total > max_chars and kept:
            break
        kept.append(msg)
    return system + kept[::-1]


# ---------------------------------------------------------------------- commands
def cmd_status(rt: Runtime, args) -> int:
    cfg, bad = rt.cfg, 0
    providers = {cfg.llm_provider, cfg.llm_fallback} - {""}
    print(f"LLM:         provider={cfg.llm_provider}" + (f", fallback={cfg.llm_fallback}" if cfg.llm_fallback else ""))
    if "anthropic" in providers:
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        print(f"Anthropic:   model {cfg.anthropic_model}; ANTHROPIC_API_KEY " + ("set" if has_key else "NOT SET"))
        bad += not has_key
    if "ollama" in providers:
        ollama = OllamaClient(cfg.ollama_url, cfg.model, timeout=5)
        try:
            models = ollama.list_models()
            have = any(m == cfg.model or m.split(":")[0] == cfg.model for m in models)
            line = f"Ollama:      OK ({cfg.ollama_url}); model {cfg.model} " + ("installed" if have else "NOT installed - run `ollama pull`")
            bad += not have
            for m in ollama.running_models():
                if m.get("name", "").startswith(cfg.model.split(":")[0]) and m.get("size"):
                    line += f"; loaded, {100 * m.get('size_vram', 0) / m['size']:.0f}% in GPU VRAM"
            print(line)
        except LLMError as exc:
            print(f"Ollama:      UNAVAILABLE - {exc}")
            bad += 1
    if rt.prom.is_ready():
        print(f"Prometheus:  OK ({cfg.prometheus_url})")
    else:
        print(f"Prometheus:  UNAVAILABLE ({cfg.prometheus_url})")
        return 1
    if rt.schema is None:
        print(f"Metrics:     schema unresolved - {rt.schema_error}")
        return 1
    print(f"Metrics:     production={rt.schema.production_metric} consumption={rt.schema.consumption_metric} "
          f"item label={rt.schema.item_label}")
    for note in rt.schema.notes:
        print(f"             ({note})")
    health = rt.telemetry.health()
    print(f"Telemetry:   {health.status.value.upper()}" + (f" - {health.detail}" if health.detail else
          f" (newest sample {health.age_seconds:.0f}s old)" if health.age_seconds is not None else ""))
    return int(bad > 0 or health.status.value != "ok")


def cmd_metrics(rt: Runtime, args) -> int:
    try:
        if args.name:
            print(f"{args.name} labels: {', '.join(rt.prom.get_metric_labels(args.name)) or '(none)'}")
            for label in rt.prom.get_metric_labels(args.name):
                vals = rt.prom.label_values(label, args.name)
                print(f"  {label}: {len(vals)} values, e.g. {', '.join(vals[:6])}")
            return 0
        catalog = MetricCatalog.build(rt.prom, rt.cfg.metric_prefix)
    except PrometheusError as exc:
        print(f"Error: {exc}")
        return 1
    for m in catalog.metrics.values():
        print(f"{m.name} [{m.type or '?'}] labels={m.labels}" + (f"  # {m.help}" if m.help else ""))
    print(f"\n{len(catalog.metrics)} metrics")
    return 0


def cmd_query(rt: Runtime, args) -> int:
    try:
        if args.range:
            end = time.time()
            series = rt.prom.query_range(args.promql, str(end - parse_window(args.range, 60) * 60), str(end), args.step)
        else:
            series = rt.prom.query_instant(args.promql)
    except PrometheusError as exc:
        print(f"Error: {exc}")
        return 1
    if not series:
        print("No data (missing series - not zero).")
    for s in series:
        print(s.labels or "{}")
        shown = s.samples if len(s.samples) <= 12 else s.samples[:3] + s.samples[-9:]
        for x in shown:
            print(f"  {time.strftime('%H:%M:%S', time.localtime(x.timestamp))}  {x.value:.3f}")
    return 0


def cmd_investigate(rt: Runtime, args) -> int:
    if rt.investigator is None:
        print(f"Cannot investigate: {rt.schema_error}")
        return 1
    window = parse_window(args.window, 60)
    try:
        print("Investigating..." )
        report = (rt.investigator.investigate_item(normalize_item(args.item), window) if args.item
                  else rt.investigator.investigate_factory(window))
    except (TelemetryUnavailable, PrometheusError) as exc:
        print(f"Error: {exc}")
        return 1
    print(report.format())
    return 0


def cmd_monitor(rt: Runtime, args) -> int:
    if rt.telemetry is None:
        print(f"Cannot monitor: {rt.schema_error}")
        return 1
    alerts = AlertManager(rt.cfg, rt.cfg.state_file)
    MonitoringEngine(rt.telemetry, alerts, rt.investigator, rt.cfg).run_forever()
    return 0


def cmd_chat(rt: Runtime, args) -> int:
    cfg = rt.cfg
    messages = [{"role": "system", "content": CHAT_SYSTEM}]
    print(f"Factorio agent chat - backend: {describe_backend(cfg)}\n"
          "/backend [name] switches LLM, /reset clears history, /quit exits.")
    if rt.investigator is None:
        print(f"(Factory telemetry unavailable: {rt.schema_error}\n general chat only.)")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text.startswith("/backend"):
            print(backend_command(rt, text[len("/backend"):]) + "\n")
            cfg = rt.cfg
            continue
        if text == "/reset":
            messages = messages[:1]
            print("History cleared.")
            continue
        try:
            if rt.investigator and rt.investigator.is_factory_question(text):
                print("Investigating...")
                report = rt.investigator.ask(text)
                reply = report.format()
                memo = report.analysis or "\n".join(report.findings())
                messages += [{"role": "user", "content": text},
                             {"role": "assistant", "content": f"[factory evidence summary]\n{memo}"}]
            else:
                messages.append({"role": "user", "content": text})
                messages[:] = trim_history(messages, cfg.num_ctx * 3)
                reply = rt.llm.generate(messages)
                messages.append({"role": "assistant", "content": reply})
            messages[:] = trim_history(messages, cfg.num_ctx * 3)
            print(reply + "\n")
        except (TelemetryUnavailable, PrometheusError, LLMError) as exc:
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            print(f"Error: {exc}\n")


# ---------------------------------------------------------------------- entry point
def _common() -> argparse.ArgumentParser:
    c = argparse.ArgumentParser(add_help=False)
    d = argparse.SUPPRESS      # so a flag given in one position is not overwritten by the other
    c.add_argument("--config", default=d, help="path to a TOML config file")
    c.add_argument("--llm", choices=PROVIDERS, default=d, help="LLM backend for this run (overrides config)")
    c.add_argument("--fallback", choices=PROVIDERS + ("none",), default=d,
                   help="fallback backend if the primary fails ('none' disables)")
    c.add_argument("--model", default=d, help="model name for the selected backend")
    return c


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    p = argparse.ArgumentParser(prog="factorio_agent", parents=[common],
                                description="Local AI observability agent for Factorio (read-only).")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("chat", parents=[common], help="interactive conversation").set_defaults(fn=cmd_chat)
    sub.add_parser("status", parents=[common], help="agent and data-source health").set_defaults(fn=cmd_status)
    sub.add_parser("monitor", parents=[common], help="start continuous monitoring").set_defaults(fn=cmd_monitor)
    inv = sub.add_parser("investigate", parents=[common], help="investigate recent factory behaviour or one item")
    inv.add_argument("--window", default="1h", help="analysis window, e.g. 30m, 1h (default 1h)")
    inv.add_argument("--item", help="item to investigate, e.g. electronic-circuit")
    inv.set_defaults(fn=cmd_investigate)
    met = sub.add_parser("metrics", parents=[common], help="list discovered metrics (or labels of one metric)")
    met.add_argument("name", nargs="?")
    met.set_defaults(fn=cmd_metrics)
    q = sub.add_parser("query", parents=[common], help="run a read-only PromQL query")
    q.add_argument("promql")
    q.add_argument("--range", help="range window, e.g. 1h (omit for an instant query)")
    q.add_argument("--step", default="60s")
    q.set_defaults(fn=cmd_query)
    return p


def apply_overrides(cfg: Config, args) -> Config:
    """Command-line flags beat the config file and environment."""
    if getattr(args, "llm", None):
        cfg.llm_provider = args.llm
    if getattr(args, "fallback", None):
        cfg.llm_fallback = "" if args.fallback == "none" else args.fallback
    if getattr(args, "model", None):
        setattr(cfg, "anthropic_model" if cfg.llm_provider == "anthropic" else "model", args.model)
    for name in (cfg.llm_provider, cfg.llm_fallback):
        if name and name not in PROVIDERS:
            raise ValueError(f"Unknown LLM backend '{name}'. Choose from: {', '.join(PROVIDERS)}")
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = apply_overrides(Config.load(getattr(args, "config", None)), args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    return args.fn(Runtime(cfg), args)
