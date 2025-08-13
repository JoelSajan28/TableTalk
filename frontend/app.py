import os
import requests
import streamlit as st
import pandas as pd

# Backends
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama2-uncensored")  # change if you pulled a different model

st.set_page_config(page_title="TableTalk — Ingest + Chat", layout="wide")
st.title("📊 TableTalk — Excel → SQLite Ingestion")

st.caption(f"API: {BACKEND_URL} • Ollama: {OLLAMA_URL}")

# ---------------------------
# Ingest panel (your existing part)
# ---------------------------
with st.sidebar:
    st.header("Upload")
    dataset = st.text_input("Dataset name (prefix for tables)", value="demo_dataset")
    uploaded = st.file_uploader("Choose an Excel file (.xlsx or .xls)", type=["xlsx", "xls"])
    run = st.button("Ingest via API")

if run and not uploaded:
    st.warning("Please select a file first.")
elif run and not dataset.strip():
    st.warning("Please enter a dataset name.")

if uploaded and run and dataset.strip():
    st.info("Uploading to backend…")
    files = {
        "file": (
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    data = {"dataset": dataset.strip()}

    try:
        resp = requests.post(f"{BACKEND_URL}/ingest/excel", files=files, data=data, timeout=180)
        if resp.status_code != 200:
            st.error(f"Backend error {resp.status_code}: {resp.text}")
        else:
            payload = resp.json()
            st.success("Ingested successfully!")

            st.subheading = st.subheader("Ingestion Summary")
            st.write(f"**File:** {payload.get('filename')}")
            st.write(f"**SQLite path:** {payload.get('sqlite_path')}")
            st.write(f"**Dataset:** `{payload.get('dataset')}`")

            tables = payload.get("tables", [])
            st.write(f"**Tables created:** {len(tables)}")

            if tables:
                df = pd.DataFrame(tables)
                if "columns" in df.columns:
                    df["columns"] = df["columns"].apply(lambda cols: ", ".join(map(str, cols)))
                st.dataframe(df, use_container_width=True)
    except Exception as e:
        print(f"⚠️ Backend error: {e}")
# ---------------------------
# Chat (Ollama) panel
# ---------------------------
st.markdown("---")
st.header("💬 Chat (Ollama)")

# Model selector + system prompt
with st.sidebar:
    st.header("Chat Settings")
    model = st.text_input("Ollama model", value=DEFAULT_MODEL, help="e.g., llama3, mistral, qwen2, etc.")
    sys_prompt = st.text_area(
        "System prompt (optional)",
        value="You are a helpful assistant.",
        height=80,
        help="Set the assistant behavior.",
    )

# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "system", "content": sys_prompt or "You are a helpful assistant."}
    ]

# Render past messages (skip the internal system message)
for m in [m for m in st.session_state.chat_messages if m["role"] != "system"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_input = st.chat_input("Type a message (e.g., 'how are you?')")

def ask_ollama(messages, model_name: str):
    """
    Call Ollama /api/chat with a messages array.
    We set stream=false to get a single consolidated response.
    """
    try:
        payload = {
            "model": "llama2-uncensored",
            "stream": False,
            "messages": [
                { "role": "user", "content": user_input }
            ]
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Ollama returns: { "message": {"role": "assistant", "content": "..."} , ... }
        content = (data.get("message") or {}).get("content", "")
        if not content:
            content = data.get("response", "")  # fallback for older formats
        return content
    except Exception as e:
        return f"⚠️ Ollama error: {e}"

if user_input:
    # Update the stored system prompt if changed
    if st.session_state.chat_messages and st.session_state.chat_messages[0]["role"] == "system":
        st.session_state.chat_messages[0]["content"] = sys_prompt or "You are a helpful assistant."

    # Show user message
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build a short context for Ollama (system + last N turns)
    # Keeping it small so local models respond quickly
    history = st.session_state.chat_messages[-8:]  # last few turns is enough

    # Ask Ollama
    answer = ask_ollama(history, model)
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

st.caption("Tip: change `OLLAMA_URL` or `OLLAMA_MODEL` via env vars if your setup differs.")
