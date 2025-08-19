import os
import requests
import streamlit as st
import pandas as pd

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="TableTalk — Ingest + Chat", layout="wide")
st.title("📊 TableTalk — Excel → SQLite + Chat")

# ---------------- Ingest ----------------
with st.sidebar:
    st.header("📂 Upload Excel → SQLite")
    dataset_input = st.text_input("Dataset name (prefix for tables)", value="")
    uploaded = st.file_uploader("Choose Excel (.xlsx or .xls)", type=["xlsx", "xls"])
    run = st.button("Ingest")

# Model selection
with st.sidebar:
    st.markdown("---")
    st.header("⚙️ Model")
    model_choice = st.selectbox(
        "Choose LLM (for NL→SQL)",
        options=["phi4", "deepseek-r1", "llama2-uncensored"],
        index=0,  # default = phi4
        help="Model used to convert your question into SQL"
    )

# keep session dataset synced with the text input when non-empty
if dataset_input.strip():
    st.session_state["dataset"] = dataset_input.strip()

def set_active_dataset_from_payload(payload: dict, fallback: str):
    name = (payload.get("dataset") or fallback or "").strip()
    if name:
        st.session_state["dataset"] = name

if run:
    if not uploaded:
        st.warning("⚠️ Please select a file first.")
    elif not (dataset_input or "").strip():
        st.warning("⚠️ Please enter a dataset name.")
    else:
        st.info("⏳ Uploading to backend…")
        files = {
            "file": (
                uploaded.name,
                uploaded.getvalue(),
                uploaded.type or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        }
        data = {"dataset": dataset_input.strip()}
        try:
            resp = requests.post(f"{BACKEND_URL}/ingest/excel", files=files, data=data, timeout=180)
            if resp.status_code != 200:
                st.error(f"❌ Backend error {resp.status_code}: {resp.text}")
            else:
                payload = resp.json()
                st.success("✅ Ingested successfully!")
                set_active_dataset_from_payload(payload, dataset_input)

                st.subheader("Ingestion Summary")
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
            st.error(f"⚠️ Backend error: {e}")

# ---------------- Chat (NL→SQL via backend) ----------------
st.markdown("---")
st.header("💬 Chat with your dataset")

# init chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

def current_dataset() -> str:
    return (st.session_state.get("dataset") or dataset_input or "").strip()

st.caption(f"Active dataset: `{current_dataset() or '— not set —'}` • Model: `{model_choice}`")

# render history
for m in st.session_state.chat_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def ask_backend(ds: str, question: str, model: str, max_rows: int = 50):
    if not ds:
        return {"error": "No active dataset. Please ingest an Excel file or enter a dataset name."}
    try:
        r = requests.post(
            f"{BACKEND_URL}/ask",
            json={"dataset": ds, "question": question, "max_rows": max_rows, "model": model},
            timeout=180,
        )
        if r.status_code != 200:
            return {"error": f"Backend returned {r.status_code}: {r.text}"}
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

prompt = st.chat_input("Ask (e.g., 'get me the first 2 customers', 'show top 3 orders by amount')")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    ds = current_dataset()
    out = ask_backend(ds, prompt, model_choice, 50)

    if "error" in out:
        answer_md = f"⚠️ {out['error']}"
        st.session_state.chat_messages.append({"role": "assistant", "content": answer_md})
        with st.chat_message("assistant"):
            st.markdown(answer_md)
    else:
        sql = out.get("sql", "")
        rows = out.get("rows", [])
        answer_md = f"**SQL**\n```sql\n{sql}\n```\n"
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            answer_md += f"\nReturned **{len(rows)}** row(s)."
        else:
            answer_md += "_No rows returned._"

        st.session_state.chat_messages.append({"role": "assistant", "content": answer_md})
        with st.chat_message("assistant"):
            st.markdown(answer_md)
