# app/services/ollama_client.py
import os
import requests
from typing import List, Dict, Optional

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2-uncensored")  # set whatever you pulled

def chat(messages: List[Dict], model: Optional[str] = None, stream: bool = False, options: Optional[Dict] = None) -> str:
    """
    Call Ollama /api/chat and return the assistant text.
    messages: [{"role":"system"/"user"/"assistant","content":"..."}]
    """
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": 0, **(options or {})},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # newer Ollama: {"message":{"role":"assistant","content":"..."}}
    # older: {"response":"..."}
    return (data.get("message") or {}).get("content") or data.get("response") or ""
