"""
Expert layer: uses LangChain's ChatGroq (langchain-groq, an actively
maintained LangChain partner package) for both the RAG answer and the
chart-suggestion step.
"""
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_client(api_key: str) -> str:
    """We don't build the ChatGroq instance here because the model can
    change per call (the sidebar model selector); this just passes the key
    through so app.py's existing call sites don't need to change."""
    return api_key


def _build_llm(api_key: str, model: str, temperature: float) -> ChatGroq:
    return ChatGroq(api_key=api_key, model=model, temperature=temperature)


def ask_expert(client: str, query: str, context_chunks: list, chat_history: list,
               model: str = DEFAULT_MODEL) -> str:
    """Answer the user's question using ONLY the retrieved context chunks."""
    context_text = "\n\n---\n\n".join(c["text"] for c in context_chunks)

    system_prompt = (
        "You are a data analyst assistant. Answer the user's question about "
        "their dataset using ONLY the context below. Be precise, reference "
        "actual numbers/values from the context, and clearly say if the "
        "context doesn't contain enough information to answer.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    messages = [SystemMessage(content=system_prompt)]
    for m in chat_history[-6:]:  # a little conversational memory
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=query))

    llm = _build_llm(client, model, temperature=0.2)
    chain = llm | StrOutputParser()
    return chain.invoke(messages)


def suggest_chart(client: str, query: str, dtypes: dict, model: str = DEFAULT_MODEL) -> dict:
    """Ask the LLM whether a chart would help answer this query, and if so, how."""
    system_prompt = (
        "Decide if a chart would help answer this data question. "
        "Respond ONLY with valid JSON (no markdown fences, no extra text) exactly like:\n"
        '{"needs_chart": true, "chart_type": "bar", "x_column": "col_name", '
        '"y_column": "col_name_or_null", "reason": "short reason"}\n\n'
        "chart_type must be one of: bar, line, scatter, histogram, pie.\n"
        f"Available columns and dtypes: {json.dumps(dtypes)}"
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]

    try:
        llm = _build_llm(client, model, temperature=0)
        response = llm.invoke(messages)
        content = response.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception:
        return {"needs_chart": False}