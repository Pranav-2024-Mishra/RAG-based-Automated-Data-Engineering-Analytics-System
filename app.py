"""
Automated Data Engineering & Analytics System
Upload -> ETL Pipeline -> RAG System -> Conversational Analytics
"""
import io
import os
import streamlit as st
import pandas as pd

from etl import load_data, profile_data, clean_data
from rag import build_chunks, Retriever
from llm import get_client, ask_expert, suggest_chart
from charts import build_chart

st.set_page_config(
    page_title="Data Engineering & Analytics System",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system: typography, hero, stepper, tabs, stat row, chat panels
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Sora', sans-serif !important; letter-spacing: -0.01em; }
code, .mono { font-family: 'JetBrains Mono', monospace !important; }

#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* ---------- Hero panel ---------- */
.hero {
    background: #131826;
    border: 1px solid #262D3F;
    border-left: 4px solid #F2A93B;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.75rem;
}
.hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.6rem;
    font-weight: 700;
    color: #F3F5F8;
}
.hero p {
    margin: 0;
    color: #8891A6;
    font-size: 0.95rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #0A0D13;
    border-right: 1px solid #1D2333;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

.brand { margin-bottom: 1.5rem; }
.brand .mark {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #F2A93B;
    letter-spacing: 0.04em;
}
.brand .name {
    font-family: 'Sora', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #F3F5F8;
    margin-top: 0.15rem;
}

.stepper { list-style: none; margin: 0 0 1.5rem 0; padding: 0; }
.stepper li {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.45rem 0;
    font-size: 0.88rem;
}
.stepper .dot {
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    flex-shrink: 0;
    border: 1.5px solid #333B50;
    color: #5B6478;
}
.stepper .done .dot { background: #F2A93B; border-color: #F2A93B; color: #0B0E14; }
.stepper .current .dot { border-color: #F2A93B; color: #F2A93B; }
.stepper .done span.label, .stepper .current span.label { color: #E7E9EE; font-weight: 500; }
.stepper .pending span.label { color: #5B6478; }

.status-line {
    display: flex; align-items: center; gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #8891A6;
    margin-top: 0.75rem;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-on { background: #4ADE80; box-shadow: 0 0 6px #4ADE8090; }
.status-off { background: #F16060; }

/* ---------- Tabs: underline style ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.75rem;
    border-bottom: 1px solid #262D3F;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    padding: 0.6rem 0.1rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: #8891A6;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #F2A93B !important;
    border-bottom: 2px solid #F2A93B !important;
    background: transparent !important;
}

/* ---------- Stat row (replaces card-style metrics) ---------- */
.stat-row { display: flex; padding: 0.25rem 0 1.25rem 0; }
.stat { flex: 1; padding-right: 1.5rem; }
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    color: #F3F5F8;
}
.stat-label { font-size: 0.82rem; color: #8891A6; margin-top: 0.15rem; }
.stat-divider { width: 1px; background: #262D3F; margin: 0 1.5rem 0 0; }

/* ---------- File uploader ---------- */
div[data-testid="stFileUploaderDropzone"] {
    background: #131826;
    border: 1.5px dashed #333B50;
    border-radius: 10px;
    transition: border-color 0.15s ease;
}
div[data-testid="stFileUploaderDropzone"]:hover { border-color: #F2A93B; }

/* ---------- Dataframes ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid #262D3F;
    border-radius: 10px;
    overflow: hidden;
}

/* ---------- Chat ---------- */
div[data-testid="stChatMessage"] {
    background: #131826;
    border: 1px solid #262D3F;
    border-radius: 10px;
    padding: 0.4rem 0.2rem;
    margin-bottom: 0.6rem;
}

/* ---------- Alerts: quiet, left-accent style instead of solid blocks ---------- */
div[data-testid="stAlert"] {
    background: #131826;
    border: 1px solid #262D3F;
    border-radius: 8px;
}

/* ---------- Buttons ---------- */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}

/* ---------- Expander (Explainability) ---------- */
details {
    border: 1px solid #262D3F !important;
    border-radius: 8px !important;
    background: #0F131E;
}

/* ---------- Accessibility ---------- */
:focus-visible { outline: 2px solid #F2A93B !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "raw_df": None,
    "cleaned_df": None,
    "profile_report": None,
    "chunks": None,
    "retriever": None,
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# API key: read silently from secrets/env — never shown in the UI
# ---------------------------------------------------------------------------
API_KEY = None
if hasattr(st, "secrets") and st.secrets.get("GROQ_API_KEY"):
    API_KEY = st.secrets["GROQ_API_KEY"]
elif os.environ.get("GROQ_API_KEY"):
    API_KEY = os.environ["GROQ_API_KEY"]

# ---------------------------------------------------------------------------
# Sidebar: brand, pipeline stepper, model, connection status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="mark">SYSTEM</div>
        <div class="name">Data Engineering &amp; Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    step1 = "done" if st.session_state.raw_df is not None else "current"
    step2 = "done" if st.session_state.cleaned_df is not None else ("current" if st.session_state.raw_df is not None else "pending")
    step3 = "current" if st.session_state.cleaned_df is not None else "pending"

    def dot_content(state):
        return "✓" if state == "done" else ""

    st.markdown(f"""
    <ul class="stepper">
        <li class="{step1}"><span class="dot">{dot_content(step1) or '1'}</span><span class="label">Upload data</span></li>
        <li class="{step2}"><span class="dot">{dot_content(step2) or '2'}</span><span class="label">Clean &amp; profile</span></li>
        <li class="{step3}"><span class="dot">{dot_content(step3) or '3'}</span><span class="label">Ask questions</span></li>
    </ul>
    """, unsafe_allow_html=True)

    model = st.selectbox(
        "Model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        index=0,
    )

    status_class = "status-on" if API_KEY else "status-off"
    status_text = "groq connected" if API_KEY else "groq not configured"
    st.markdown(f"""
    <div class="status-line">
        <span class="status-dot {status_class}"></span>
        <span>{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    if not API_KEY:
        st.caption(
            "Add `GROQ_API_KEY` to `.streamlit/secrets.toml` (local) or your "
            "app's Secrets settings (deployed)."
        )

    st.markdown("---")
    st.caption("Your data stays in this session only.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Automated Data Engineering &amp; Analytics System</h1>
    <p>Upload data, clean and profile it, then ask questions in plain English and get grounded answers with charts.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["1. Upload & ETL", "2. Profiling & Download", "3. Chat with data"])

# ---------------------------------------------------------------------------
# TAB 1: Upload & ETL
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Upload your dataset")
    uploaded_file = st.file_uploader("CSV or Excel, up to 200MB", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state.raw_df = df
            st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.raw_df is not None:
        st.subheader("Clean & transform")
        col1, col2, col3 = st.columns(3)
        with col1:
            drop_dupes = st.checkbox("Drop duplicate rows", value=True)
        with col2:
            missing_strategy = st.selectbox(
                "Missing value strategy",
                ["drop", "fill_mean", "fill_median", "fill_mode", "none"],
                index=0,
            )
        with col3:
            cap_outliers = st.checkbox("Cap outliers (IQR method)", value=False)

        if st.button("Run ETL pipeline", type="primary"):
            with st.spinner("Cleaning and transforming data..."):
                cleaned = clean_data(
                    st.session_state.raw_df,
                    drop_duplicates=drop_dupes,
                    missing_strategy=missing_strategy,
                    cap_outliers=cap_outliers,
                )
                st.session_state.cleaned_df = cleaned
                st.session_state.profile_report = profile_data(st.session_state.raw_df)

                chunks = build_chunks(cleaned)
                st.session_state.chunks = chunks
                st.session_state.retriever = Retriever(chunks)
                st.session_state.chat_history = []

            st.success(
                f"ETL complete — {cleaned.shape[0]} rows × {cleaned.shape[1]} columns. "
                f"Open **3. Chat with data** to start asking questions."
            )
            st.dataframe(cleaned.head(20), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: Profiling + Download
# ---------------------------------------------------------------------------
with tab2:
    if st.session_state.profile_report is None:
        st.info("Run the ETL pipeline in the first tab to see a profiling report.")
    else:
        report = st.session_state.profile_report
        shape = report["shape"]

        st.subheader("Data profiling report")
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat"><div class="stat-value">{shape[0]}</div><div class="stat-label">Rows (original)</div></div>
            <div class="stat-divider"></div>
            <div class="stat"><div class="stat-value">{shape[1]}</div><div class="stat-label">Columns</div></div>
            <div class="stat-divider"></div>
            <div class="stat"><div class="stat-value">{report['duplicate_count']}</div><div class="stat-label">Duplicate rows</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Missing values**")
        if len(report["missing_values"]) == 0:
            st.caption("No missing values found.")
        else:
            st.dataframe(report["missing_values"], use_container_width=True)

        st.markdown("**Outliers (IQR method)**")
        if len(report["outliers"]) == 0:
            st.caption("No outliers detected.")
        else:
            st.dataframe(report["outliers"], use_container_width=True)

        st.markdown("---")
        st.subheader("Download processed data")
        cleaned = st.session_state.cleaned_df
        if cleaned is not None:
            dl1, dl2 = st.columns(2)
            with dl1:
                csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv_bytes,
                                    file_name="cleaned_dataset.csv", mime="text/csv",
                                    use_container_width=True)
            with dl2:
                excel_buffer = io.BytesIO()
                cleaned.to_excel(excel_buffer, index=False, engine="openpyxl")
                st.download_button("Download Excel", data=excel_buffer.getvalue(),
                                    file_name="cleaned_dataset.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Conversational Analytics
# ---------------------------------------------------------------------------
with tab3:
    if st.session_state.retriever is None:
        st.info("Run the ETL pipeline in the first tab before chatting with your data.")
    elif not API_KEY:
        st.warning(
            "Groq isn't configured yet. Add `GROQ_API_KEY` to `.streamlit/secrets.toml` "
            "(local) or your app's Secrets settings (Streamlit Cloud), then refresh."
        )
    else:
        st.subheader("Ask questions about your data")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("chart"):
                    st.plotly_chart(msg["chart"], use_container_width=True)
                if msg.get("sources"):
                    with st.expander("Explainability — sources used"):
                        for s in msg["sources"]:
                            st.markdown(f'<span class="mono">{s["id"]} · relevance {s["score"]:.2f}</span>', unsafe_allow_html=True)
                            st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))

        query = st.chat_input("Ask a question about your data...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant data and asking the expert..."):
                    client = get_client(API_KEY)
                    retrieved = st.session_state.retriever.retrieve(query, top_k=4)
                    answer = ask_expert(
                        client, query, retrieved,
                        [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[:-1]],
                        model=model,
                    )
                    st.markdown(answer)

                    fig = None
                    chart_spec = suggest_chart(
                        client, query,
                        st.session_state.cleaned_df.dtypes.astype(str).to_dict(),
                        model=model,
                    )
                    if chart_spec.get("needs_chart"):
                        fig = build_chart(st.session_state.cleaned_df, chart_spec)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Explainability — sources used"):
                        for s in retrieved:
                            st.markdown(f'<span class="mono">{s["id"]} · relevance {s["score"]:.2f}</span>', unsafe_allow_html=True)
                            st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))

            st.session_state.chat_history.append({
                "role": "assistant", "content": answer, "chart": fig, "sources": retrieved,
            })