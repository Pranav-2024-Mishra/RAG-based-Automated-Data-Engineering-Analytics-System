"""
RAG System using ChromaDB + FastEmbed.

Flow:

DataFrame
    ↓
LangChain Documents
    ↓
FastEmbed embeddings
    ↓
ChromaDB
    ↓
Semantic Retrieval
"""

import numpy as np
import pandas as pd
import streamlit as st

from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ============================================================
# 1. Build LangChain Documents
# ============================================================

def build_chunks(
    df: pd.DataFrame,
    rows_per_chunk: int = 50,
    max_row_chunks: int = 200
) -> list[Document]:

    chunks = []

    # --------------------------------------------------------
    # Schema information
    # --------------------------------------------------------

    schema_text = "Dataset schema:\n" + "\n".join(
        f"- {col} ({df[col].dtype})"
        for col in df.columns
    )

    schema_text += (
        f"\nTotal rows: {len(df)}"
        f"\nTotal columns: {len(df.columns)}"
    )

    chunks.append(
        Document(
            page_content=schema_text,
            metadata={"id": "schema"}
        )
    )

    # --------------------------------------------------------
    # Numeric statistics
    # --------------------------------------------------------

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_cols) > 0:

        desc = df[numeric_cols].describe().round(2)

        stats_text = "Numeric column statistics:\n"

        for col in numeric_cols:

            stats_text += (
                f"- {col}: "
                f"mean={desc.loc['mean', col]}, "
                f"min={desc.loc['min', col]}, "
                f"max={desc.loc['max', col]}, "
                f"std={desc.loc['std', col]}\n"
            )

        chunks.append(
            Document(
                page_content=stats_text,
                metadata={"id": "numeric_stats"}
            )
        )

    # --------------------------------------------------------
    # Categorical value counts
    # --------------------------------------------------------

    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in cat_cols[:10]:

        value_counts = (
            df[col]
            .value_counts()
            .head(10)
        )

        text = (
            f"Top values in column '{col}':\n"
        )

        text += "\n".join(
            f"- {value}: {count} occurrences"
            for value, count in value_counts.items()
        )

        chunks.append(
            Document(
                page_content=text,
                metadata={"id": f"cat_{col}"}
            )
        )

    # --------------------------------------------------------
    # Row-level chunks
    # --------------------------------------------------------

    max_rows_to_chunk = rows_per_chunk * max_row_chunks

    if len(df) > max_rows_to_chunk:

        row_sample = (
            df
            .sample(
                n=max_rows_to_chunk,
                random_state=42
            )
            .sort_index()
        )

        sample_note = (
            f" (sampled {max_rows_to_chunk} "
            f"of {len(df)} total rows)"
        )

    else:

        row_sample = df
        sample_note = ""

    # --------------------------------------------------------
    # Create row chunks
    # --------------------------------------------------------

    for i in range(
        0,
        len(row_sample),
        rows_per_chunk
    ):

        batch = row_sample.iloc[
            i:i + rows_per_chunk
        ]

        text = (
            f"Data rows{sample_note}, "
            f"batch starting at position {i}:\n"
        )

        text += batch.to_string(
            index=False
        )

        chunks.append(
            Document(
                page_content=text,
                metadata={
                    "id": f"rows_{i}"
                }
            )
        )

    return chunks


# ============================================================
# 2. FastEmbed → LangChain Embeddings
# ============================================================

class FastEmbedWrapper(Embeddings):
    """
    Small wrapper that allows FastEmbed to work with
    LangChain's embedding interface.
    """

    def __init__(self):

        self.model = TextEmbedding(
            model_name=EMBEDDING_MODEL
        )

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        embeddings = self.model.embed(texts)

        return [
            embedding.tolist()
            for embedding in embeddings
        ]

    def embed_query(
        self,
        text: str
    ) -> list[float]:

        embedding = next(
            self.model.embed([text])
        )

        return embedding.tolist()


# ============================================================
# 3. Load embedding model once
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embedding_model():

    return FastEmbedWrapper()


# ============================================================
# 4. ChromaDB Retriever
# ============================================================

class Retriever:
    """
    ChromaDB-based semantic retriever.

    Keeps the same interface as the old FAISS Retriever,
    so app.py does not need major changes.
    """

    def __init__(
        self,
        chunks: list[Document]
    ):

        self.chunks = chunks

        # Load embedding model
        self.model = get_embedding_model()

        # Create Chroma vector database
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.model,
            collection_name="data_analysis_rag"
        )

    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 4
    ) -> list:

        results = (
            self.vectorstore
            .similarity_search_with_relevance_scores(
                query,
                k=top_k
            )
        )

        retrieved = []

        for doc, score in results:

            retrieved.append({
                "id": doc.metadata.get(
                    "id",
                    "unknown"
                ),

                "text": doc.page_content,

                "score": float(score)
            })

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if not retrieved:

            retrieved = [
                {
                    "id": doc.metadata.get(
                        "id",
                        "unknown"
                    ),
                    "text": doc.page_content,
                    "score": 0.0
                }

                for doc in self.chunks

                if doc.metadata.get("id")
                in ["schema", "numeric_stats"]
            ]

        return retrieved