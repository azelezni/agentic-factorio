# Factorio AI Agent

A local, **read-only** observability agent for Factorio. Deterministic monitoring and
time-series analysis find production problems; a local LLM (Ollama, `qwen3:8b`) explains
the evidence. *Deterministic code collects and validates facts; the LLM reasons about them.*

```
Factorio + Wiretap mod -> stats.json -> Wiretap bridge -> Prometheus -> Python agent -> Ollama
```

## Setup

1. **Monitoring stack** (Docker): put the real Wiretap bridge into `docker/wiretap/`
   (the Dockerfile there is a stub), set `FACTORIO_SCRIPT_OUTPUT` in `docker/.env`, then
   `docker compose -f docker/docker-compose.yml up -d`.
2. **Ollama** (native): `ollama pull qwen3:8b`.
3. **Agent**: Python 3.10+, no third-party runtime dependencies.
   `pip install -e .` (or run with `PYTHONPATH=src`).
4. `python -m factorio_agent status`, then `python -m factorio_agent metrics` to see what the
   exporter actually exposes.

## Commands

| Command | Description |
|---|---|
| `python -m factorio_agent chat` | Interactive chat; factory questions trigger investigations; `/backend` switches LLM |
| `python -m factorio_agent status` | Selected LLM backend(s) (Ollama incl. GPU VRAM use / API key), Prometheus, metric schema, telemetry freshness |
| `python -m factorio_agent monitor` | Continuous monitoring (60 s cycle, LLM only on anomalies) |
| `python -m factorio_agent investigate --window 1h` | Factory-wide investigation |
| `python -m factorio_agent investigate --item electronic-circuit` | Investigate one product |
| `python -m factorio_agent metrics [NAME]` | Discover metrics / labels |
| `python -m factorio_agent query 'PROMQL' [--range 1h]` | Read-only PromQL query |

## Configuration

Defaults < TOML file (`--config`, `$FACTORIO_AGENT_CONFIG` or `./factorio_agent.toml`, Python 3.11+)
< environment variables `FACTORIO_AGENT_<KEY>`. All keys are in `src/factorio_agent/config.py`.

**The Wiretap metric schema is discovered, not assumed.** If auto-detection is ambiguous, set:

```toml
production_metric  = "wiretap_item_production_total"   # your real names
consumption_metric = "wiretap_item_consumption_total"
item_label         = "item"
metric_kind        = "counter"   # or "gauge" (+ gauge_scale to convert to items/min)
```

Thresholds, windows (5 min current / 30 min short / 2 h long), re-investigation interval etc.
are configurable too. Recipe relationships use a built-in vanilla table; override with
`recipes_file` (JSON `{"item": ["input", ...]}`) for modded/changed recipes.

## Choosing the LLM backend

Three ways, highest priority first:

1. **Per run, on the command line** (works before or after the subcommand):
   ```
   python -m factorio_agent --llm anthropic investigate --item electronic-circuit
   python -m factorio_agent monitor --llm ollama --fallback anthropic
   python -m factorio_agent chat --llm anthropic --model claude-haiku-5-5
   ```
   `--fallback none` disables a configured fallback; `--model` applies to the selected backend.
2. **Live, inside `chat`:** `/backend` shows the current one; `/backend anthropic`,
   `/backend ollama` or `/backend anthropic <model>` switch immediately and keep the conversation.
3. **Default, in config / env:**
   ```toml
   llm_provider = "ollama"      # "ollama" (local) or "anthropic" (Claude API)
   llm_fallback = "anthropic"   # optional; used only if the primary is unavailable
   model = "qwen3:8b"           # Ollama model
   anthropic_model = "claude-sonnet-5-5"
   ```

The Claude API needs a Claude Platform API key (claude.ai chat plans do not include API access),
supplied only as an environment variable: `ANTHROPIC_API_KEY=sk-ant-...` (billed per token). With
`anthropic`, investigation evidence (item names and production rates, not your save file) is sent to
Anthropic, which departs from the "fully local" goal; the default stays `ollama`.
Only the LLM is swappable: monitoring and anomaly detection never use it.

## Behaviour notes

* **Missing data is never zero.** Failed queries, missing series, stale series and a down bridge
  pause the rules and are reported; they never raise or resolve alerts.
* **Alert lifecycle** `DETECTED -> INVESTIGATING -> ACTIVE -> RESOLVED`, persisted in
  `data/alerts.json`, so restarts do not re-announce active alerts. One root cause yields one
  investigation (upstream items are covered by the most downstream alerting item).
* **LLM down?** Monitoring continues; reports fall back to rule-based findings and the
  investigation is retried later. Investigations + evidence are logged to
  `data/investigations/investigations.jsonl`.
* Investigations run on a single worker thread so a slow model never stalls metric collection.

## Tests

`PYTHONPATH=src:tests python -m unittest discover -s tests` (also works with `pytest`).
Includes an end-to-end acceptance test with stub Prometheus/Ollama servers and a simulated
copper-plate production drop.
