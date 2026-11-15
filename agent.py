import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

SYSTEM_PROMPT = """
You are a helpful assistant specializing in Factorio 2.1.
Help the user understand factory design, production ratios,
resource bottlenecks, and optimization.

Be technically accurate. If you are unsure about something,
say so rather than inventing game mechanics.
"""


class OllamaAgent:
    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def ask(self, prompt: str) -> str:
        self.messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
            "keep_alive": "5m",
            "options": {
                "num_ctx": 8192
            }
        }

        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))

            answer = result["message"]["content"]

            self.messages.append({
                "role": "assistant",
                "content": answer
            })

            return answer

        except HTTPError as exc:
            self.messages.pop()
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}: {details}"
            ) from exc

        except URLError as exc:
            self.messages.pop()
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}: {exc.reason}"
            ) from exc

        except (KeyError, json.JSONDecodeError) as exc:
            self.messages.pop()
            raise RuntimeError(
                f"Unexpected response from Ollama: {exc}"
            ) from exc

    def clear_history(self):
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]


def main():
    print("Factorio AI Agent")
    print(f"Model: {MODEL}")
    print(f"Ollama: {OLLAMA_HOST}")
    print("Commands: /clear, /exit")
    print("-" * 50)

    agent = OllamaAgent(MODEL, OLLAMA_HOST)

    while True:
        try:
            prompt = input("\nYou: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ("/exit", "/quit"):
                print("Goodbye!")
                break

            if prompt.lower() == "/clear":
                agent.clear_history()
                print("Conversation history cleared.")
                continue

            print("\nAgent: ", end="", flush=True)
            answer = agent.ask(prompt)
            print(answer)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except RuntimeError as exc:
            print(f"\nError: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
