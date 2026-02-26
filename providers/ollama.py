import os
import json
import urllib.request
from typing import Optional

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


def ollama_chat(prompt: str, system: Optional[str] = None, model: str = DEFAULT_MODEL) -> str:
    """
    Minimal Ollama chat call via HTTP.
    Returns the assistant message text.
    """
    url = f"{DEFAULT_OLLAMA_URL}/api/chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            # keep it stable-ish
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.0")),
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=int(os.getenv("OLLAMA_TIMEOUT", "120"))) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return (data.get("message") or {}).get("content", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")
