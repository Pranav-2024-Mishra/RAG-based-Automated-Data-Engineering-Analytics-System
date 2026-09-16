# 🤖 Automated Data Engineering & Analytics System

Upload a dataset → clean it (ETL) → chat with it in natural language (RAG + Groq LLM) → get auto-generated charts.

## How it maps to the architecture

| Diagram box | File | What it does |
|---|---|---|
| Data Upload | `app.py` (Tab 1) | `st.file_uploader` for CSV/Excel |
| Extract | `etl.py: load_data()` | Reads CSV/Excel into a DataFrame |
| Transform | `etl.py: clean_data()` | Drops duplicates, handles missing values, caps outliers |
| Data Profiling Report | `etl.py: profile_data()` | Missing values, duplicates, outliers |
| Load | `rag.py: build_chunks()` | Turns cleaned data into retrievable text chunks |
| Retriever | `rag.py: Retriever` | TF-IDF similarity search over chunks |
| Expert (Groq LLM API) | `llm.py: ask_expert()` | Sends retrieved context + question to Groq |
| Natural Language Answers | `app.py` (Tab 3) | Chat UI |
| Auto-Generated Charts | `llm.py: suggest_chart()` + `charts.py` | LLM decides chart type/columns, Plotly renders it |
| Explainability Mode | `app.py` (Tab 3) | Expander showing which chunks were retrieved |
| Download Processed Data | `app.py` (Tab 2) | CSV/Excel download buttons |

Not included (left as extension points, noted at the bottom): Domain-specific plugins, continuous learning, pipeline export as a standalone service.

## 1. Run locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a **free Groq API key**: https://console.groq.com/keys

Option A — paste it into the sidebar text box when the app runs.
Option B — create `.streamlit/secrets.toml` (copy `secrets.toml.example`) and put your key there so it's pre-filled.

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## 2. Try it

1. **Upload & ETL tab** — upload a CSV/Excel file, choose cleaning options, click "Run ETL Pipeline."
2. **Profiling & Download tab** — review the missing-values/duplicates/outliers report, download the cleaned file.
3. **Chat with Data tab** — ask things like *"What's the average sales by region?"* or *"Show me the trend of sales over time"*. Expand "Explainability" under any answer to see exactly which data chunks were used.

## 3. Deploy on Streamlit Community Cloud

1. Push this folder to a **public (or private) GitHub repo**.
2. Go to https://share.streamlit.io → **"New app"** → pick your repo/branch → set main file to `app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
4. Click **Deploy**. First build takes a minute or two.

That's it — no GPU, no vector DB, no heavy embedding model downloads, so it fits comfortably in Streamlit Cloud's free tier.

## 4. Extending this into the full diagram

- **Domain-specific plugins** (Finance/Healthcare/Retail): add prompt templates or extra chunk-builders per domain in `rag.py`.
- **Continuous learning**: persist chat feedback (👍/👎 per answer) to a small SQLite/Supabase table and use it to refine retrieval later.
- **Pipeline export**: add a button that serializes the cleaning steps chosen in Tab 1 into a reusable Python/YAML pipeline config.
- **Scaling the retriever**: swap `rag.py`'s TF-IDF `Retriever` for a proper vector store (FAISS + `sentence-transformers`) if your datasets get very large or you need true semantic search — this repo intentionally starts lightweight so it deploys instantly.

## Project structure

```
data-analytics-app/
├── app.py                        # Streamlit UI, ties everything together
├── etl.py                        # Extract, Transform, Profile
├── rag.py                        # Chunking + TF-IDF Retriever (RAG)
├── llm.py                        # Groq API calls (Expert + chart suggestion)
├── charts.py                     # Plotly chart builder
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml.example
```
