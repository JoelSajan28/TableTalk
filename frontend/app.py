import os
import requests
import streamlit as st
import pandas as pd

# ---------------- Config ----------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="TableTalk — Ingest + Chat", layout="wide")
st.title("📊 TableTalk — Excel → SQLite + Chat")

# ---------------- Ingest ----------------
with st.sidebar:
    st.header("📂 Upload Excel → SQLite")
    dataset = st.text_input("Dataset name (prefix for tables)", value="demo_dataset")
    uploaded = st.file_uploader("Choose Excel (.xlsx or .xls)", type=["xlsx", "xls"])
    run = st.button("Ingest")

if run and not uploaded:
    st.warning("⚠️ Please select a file first.")
elif run and not dataset.strip():
    st.warning("⚠️ Please enter a dataset name.")

if run and uploaded and dataset.strip():
    st.info("⏳ Uploading to backend…")
    files = {
        "file": (
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    }
    data = {"dataset": dataset.strip()}
    try:
        resp = requests.post(f"{BACKEND_URL}/ingest/excel", files=files, data=data, timeout=180)
        if resp.status_code != 200:
            st.error(f"❌ Backend error {resp.status_code}: {resp.text}")
        else:
            payload = resp.json()
            st.success("✅ Ingested successfully!")
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

# persist dataset name
if "dataset" not in st.session_state:
    st.session_state.dataset = dataset.strip()

# persist chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# render history
for m in st.session_state.chat_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# helper: call backend
def ask_backend(dataset: str, question: str, max_rows: int = 50):
    try:
        r = requests.post(
            f"{BACKEND_URL}/ask",
            json={"dataset": dataset, "question": question, "max_rows": max_rows},
            timeout=180,
        )
        if r.status_code != 200:
            return {"error": f"Backend returned {r.status_code}: {r.text}"}
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# input
prompt = st.chat_input("Ask me (e.g., 'get me the first 2 customers', 'show top 3 orders by amount')")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    out = ask_backend(st.session_state.dataset, prompt, 50)

    if "error" in out:
        answer_md = f"⚠️ {out['error']}"
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
