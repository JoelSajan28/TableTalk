import os
import requests
import streamlit as st
import pandas as pd

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="TableTalk — Excel → SQLite + Chat", layout="wide")
st.title("TableTalk")

# =========================
# Sidebar — Ingestion & Model
# =========================
with st.sidebar:
    st.header("Upload Excel → SQLite")
    dataset_input = st.text_input("Dataset name (prefix)", value="")
    uploaded = st.file_uploader("Excel file (.xlsx / .xls)", type=["xlsx", "xls"])
    run = st.button("Ingest", use_container_width=True)

with st.sidebar:
    st.divider()
    st.header("Models")

    # Model used for NL -> SQL
    model_choice = st.selectbox(
        "LLM for NL → SQL (query → SQL)",
        options=[
            "phi4",
            "mxbai-embed-large",
            "granite3.2-vision",
            "deepseek-r1",
            "llama3.2-visio",
            "granite3.1-dense",
            "nomic-embed-text",
            "llama3.1:8b",
        ],
        index=0,
        help="Model used to translate your question into SQL.",
    )

    # Optional: separate model for SQL -> NL
    nl_model_choice = st.selectbox(
        "LLM for SQL → NL (rows → answer)",
        options=[
            "(same as above)",
            "phi4",
            "llama3.1:8b",
            "granite3.1-dense",
        ],
        index=0,
        help="Model used to turn SQL rows into a natural-language answer.",
    )

    st.caption("Tip: start with `phi4`. You can later split models if you want different styles/perf.")

with st.sidebar:
    st.divider()
    st.header("Answer style")
    tone_choice = st.selectbox(
        "Tone",
        options=["chatty", "precise"],
        index=0,
        help="Chatty = friendly ChatGPT-like; Precise = terse analyst tone.",
    )
    allow_table_choice = st.checkbox(
        "Include mini table when ≤ 10 rows",
        value=True,
        help="Append a compact Markdown table for small results.",
    )

# Persist chosen dataset in session
if dataset_input.strip():
    st.session_state["dataset"] = dataset_input.strip()

def _set_dataset_from_payload(payload: dict, fallback: str) -> None:
    name = (payload.get("dataset") or fallback or "").strip()
    if name:
        st.session_state["dataset"] = name

def _clear_chat_session(reason: str | None = None) -> None:
    st.session_state.pop("chat_messages", None)
    st.session_state.pop("last_ingested_key", None)
    if reason:
        st.info(reason)

# =========================
# Ingest flow
# =========================
if run:
    if not uploaded:
        st.warning("Please select a file first.")
    elif not (dataset_input or "").strip():
        st.warning("Please enter a dataset name.")
    elif not uploaded.name.lower().endswith((".xlsx", ".xls")):
        st.error("❌ Invalid file format. Please upload an Excel file with extension .xlsx or .xls.")
    else:
        with st.spinner("Uploading to backend…"):
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
            except Exception as e:
                st.error(f"Backend error: {e}")
            else:
                if resp.status_code != 200:
                    st.error(f"Backend error {resp.status_code}: {resp.text}")
                else:
                    payload = resp.json() or {}

                    # Wipe old conversation when a new Excel is ingested
                    new_key = f"{payload.get('dataset','')}::{payload.get('filename','')}"
                    if st.session_state.get("last_ingested_key") != new_key:
                        _clear_chat_session("Previous chat cleared for the new dataset.")
                    st.session_state["last_ingested_key"] = new_key
                    _set_dataset_from_payload(payload, dataset_input)

                    st.success("Ingest completed.")
                    with st.container():
                        col1, col2, col3 = st.columns([3, 3, 4])
                        with col1:
                            st.markdown(f"**File:** {payload.get('filename','—')}")
                        with col2:
                            st.markdown(f"**Dataset:** `{payload.get('dataset','—')}`")
                        with col3:
                            st.markdown(f"**SQLite path:** {payload.get('sqlite_path','—')}")

                    # Tables overview
                    tables = payload.get("tables", []) or []
                    st.markdown(f"**Tables created:** {len(tables)}")
                    if tables:
                        df = pd.DataFrame(tables)
                        if "columns" in df.columns:
                            df["columns"] = df["columns"].apply(lambda cols: ", ".join(map(str, cols)))
                        st.dataframe(df, use_container_width=True)

                    # Diagnostics
                    diagnostics = payload.get("diagnostics") or {}
                    diag_items = diagnostics.get("items", []) or []
                    diag_summary = diagnostics.get("summary", {}) or {}

                    if diag_items or diag_summary:
                        st.divider()
                        st.subheader("Data Quality & Preprocessing Diagnostics")

                        s_info = int(diag_summary.get("info", 0))
                        s_warn = int(diag_summary.get("warning", 0))
                        s_err = int(diag_summary.get("error", 0))
                        s_handled = sum(1 for it in diag_items if it.get("handled"))

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Info", s_info)
                        c2.metric("Warnings", s_warn)
                        c3.metric("Errors", s_err)
                        c4.metric("Handled", s_handled)

                        with st.expander("Show diagnostic details", expanded=(s_err > 0 or s_warn > 0)):
                            if diag_items:
                                df_diag = pd.DataFrame(diag_items)
                                nice_cols = {
                                    "dataset": "Dataset",
                                    "sheet": "Sheet",
                                    "table_name": "Table",
                                    "severity": "Severity",
                                    "code": "Code",
                                    "message": "Message",
                                    "handled": "Handled",
                                    "suggestion": "Suggestion",
                                }
                                df_diag = df_diag.rename(
                                    columns={k: v for k, v in nice_cols.items() if k in df_diag.columns}
                                )
                                st.dataframe(df_diag, use_container_width=True)
                            else:
                                st.caption("No diagnostics reported.")

# =========================
# Chat section
# =========================
st.divider()
st.header("Chat with your dataset")

# Init chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

def current_dataset() -> str:
    return (st.session_state.get("dataset") or dataset_input or "").strip()

# Resolve nl_model to send (None = use same as NL→SQL)
resolved_nl_model = None if nl_model_choice == "(same as above)" else nl_model_choice

# Status line
status_dataset = current_dataset() or "— not set —"
st.caption(
    f"Active dataset: `{status_dataset}` • NL→SQL: `{model_choice}` • "
    f"SQL→NL: `{resolved_nl_model or model_choice}` • Tone: `{tone_choice}`"
)

# Render history
for m in st.session_state.chat_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def ask_backend(ds: str, question: str, model: str, nl_model: str | None, tone: str, allow_table: bool, max_rows: int = 50) -> dict:
    if not ds:
        return {"error": "No active dataset. Please ingest an Excel file or enter a dataset name."}
    try:
        payload = {
            "dataset": ds,
            "question": question,
            "max_rows": max_rows,
            "model": model,
        }
        # Only include optional fields if user changed them from defaults
        if nl_model is not None:
            payload["nl_model"] = nl_model
        if tone:
            payload["tone"] = tone
        if allow_table is not None:
            payload["allow_table"] = allow_table

        r = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=180)
        if r.status_code != 200:
            return {"error": f"Backend returned {r.status_code}: {r.text}"}
        return r.json() or {}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Chat input
prompt = st.chat_input("Ask a question (e.g., “show the metadata table”)")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    ds = current_dataset()

    with st.spinner("Thinking…"):
        out = ask_backend(
            ds=ds,
            question=prompt,
            model=model_choice,
            nl_model=resolved_nl_model,
            tone=tone_choice,
            allow_table=allow_table_choice,
            max_rows=50,
        )

    if out.get("error"):
        msg = f"⚠️ {out['error']}"
        st.session_state.chat_messages.append({"role": "assistant", "content": msg})
        with st.chat_message("assistant"):
            st.markdown(msg)
    else:
        # Natural-language summary
        answer_text = (out.get("answer") or "").strip()
        sql = (out.get("sql") or "").strip()
        rows = out.get("rows") or []

        parts: list[str] = []
        if answer_text:
            parts.append(answer_text)

        if sql:
            parts.append("")  # spacing
            with st.expander("Show SQL"):
                st.code(sql, language="sql")

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            parts.append(f"Returned **{len(rows)}** row(s).")
        else:
            parts.append("_No rows returned._")

        answer_md = "\n".join(parts).strip()
        st.session_state.chat_messages.append({"role": "assistant", "content": answer_md})
        with st.chat_message("assistant"):
            st.markdown(answer_md)
