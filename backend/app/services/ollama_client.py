import requests
import os

def _ollama_chat(messages: list[dict], temperature: float = 0.0, model: str | None = None) -> str:
    OLLAMA_MODEL = model or os.getenv("OLLAMA_MODEL", "phi4")
    OLLAMA_URL = os.getenv("OLLAMA_URL_LOCAL", "http://localhost:11434/api/chat")
    payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            },
    }
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120   
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    data = response.json()
    return data.get("message", {}).get("content", "")
