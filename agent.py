"""Phase 1 smoke test: verifies Python -> Ollama -> Qwen3 works.

Superseded by the package entry point (`python -m factorio_agent chat`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from factorio_agent.config import Config  # noqa: E402
from factorio_agent.llm.client import LLMError, OllamaClient  # noqa: E402

cfg = Config.load()
client = OllamaClient(cfg.ollama_url, cfg.model, cfg.num_ctx, cfg.keep_alive, cfg.llm_timeout)
try:
    print(client.generate([{"role": "user", "content": "Reply with one short sentence confirming you are working."}]))
    for m in client.running_models():
        if m.get("size"):
            print(f"{m['name']}: {100 * m.get('size_vram', 0) / m['size']:.0f}% in GPU VRAM")
except LLMError as exc:
    sys.exit(f"Smoke test failed: {exc}")
