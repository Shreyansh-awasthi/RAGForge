import asyncio
import hashlib
import logging
import os
import shutil
import time
import uuid
from typing import Annotated, Any, List, Literal, Optional, TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from flashrank import Ranker, RerankRequest
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.graph import END, START, StateGraph
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
import nest_asyncio
import uvicorn
from pydantic import BaseModel, Field
import operator
from mcp.server.fastmcp import FastMCP

# LangSmith tracing
from langsmith import traceable

load_dotenv()
logging.getLogger("Rag.models.factories.base_factory").setLevel(logging.ERROR)

# --- Debug: confirm LangSmith env vars are actually loaded ---
# Remove these three lines once you've confirmed tracing works.
logging.info("LANGSMITH_TRACING=%s", os.getenv("LANGSMITH_TRACING"))
logging.info("LANGSMITH_PROJECT=%s", os.getenv("LANGSMITH_PROJECT"))
logging.info("LANGSMITH_API_KEY set=%s", bool(os.getenv("LANGSMITH_API_KEY")))


class AgentState(TypedDict):
    thread_id: Annotated[str, Field(..., description="Session/thread Identifier")]
    file_path: Annotated[str, Field(..., description="Path to Upload document")]
    question: Annotated[str, Field(..., description="Question of the User")]
    docs: Annotated[List[Any], Field(default_factory=list)]
    chunks: Annotated[List[Any], Field(default_factory=list)]
    docs_status: Annotated[Optional[str], Field(default=None)]
    retrieved_docs: Annotated[List[Any], Field(default_factory=list)]
    retrieval: Annotated[Optional[str], Field(default=None)]
    history: Annotated[List[dict], operator.add]


MAX_OUTPUT_TOKENS = 800
RETRY_TOKENS = 400

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=MAX_OUTPUT_TOKENS,
)

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
RERANKER = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
PERSIST_ROOT = "./chroma_store"

NOT_RELEVANT_MESSAGE = "This question does not appear to be related to the uploaded document. Please check your document or rephrase your question."
BUSY_MESSAGE = "The AI service is currently busy. Please wait a moment and try again."

from collections import OrderedDict

MAX_THREADS_CACHED = 20      # max distinct thread_ids kept in memory at once
MAX_ANSWERS_CACHED = 200     # max cached (thread_id, question) answers


def _lru_set(store: "OrderedDict", key, value, max_size: int):
    """Insert into an OrderedDict acting as a bounded LRU cache,
    evicting the oldest entry once max_size is exceeded."""
    if key in store:
        store.pop(key)
    store[key] = value
    while len(store) > max_size:
        oldest_key, _ = store.popitem(last=False)
        logging.info("Evicting cache entry for %s", oldest_key)


def _lru_get(store: "OrderedDict", key):
    """Get + mark as recently used (moves key to the end)."""
    if key not in store:
        return None
    store.move_to_end(key)
    return store[key]


DOC_CACHE: "OrderedDict" = OrderedDict()
CHUNK_CACHE: "OrderedDict" = OrderedDict()
RETRIEVER_STORE: "OrderedDict" = OrderedDict()
ANSWER_CACHE: "OrderedDict" = OrderedDict()


def _file_fingerprint(file_path: str) -> str:
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        mtime = 0
    return hashlib.sha256(f"{file_path}:{mtime}".encode()).hexdigest()


def process_pdf(file_path: str) -> list:
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_documents(state: AgentState) -> dict:
    thread_id = state.get("thread_id")
    file_path = state.get("file_path")

    if not file_path or not thread_id:
        logging.error("Missing file_path or thread_id in state.")
        return {"docs_status": "ERROR", "docs": []}

    cached_docs = _lru_get(DOC_CACHE, thread_id)
    if cached_docs is not None:
        logging.info("Using cached parsed documents for thread %s", thread_id)
        return {"docs_status": "READY", "docs": cached_docs}

    try:
        logging.info("Loading PDF: %s", file_path)
        parsed_docs = process_pdf(file_path)
        _lru_set(DOC_CACHE, thread_id, parsed_docs, MAX_THREADS_CACHED)
        return {"docs_status": "READY", "docs": parsed_docs}
    except Exception as e:
        logging.exception("Document processing failed: %s", e)
        return {"docs_status": "FAILED", "docs": []}


def text_split(state: AgentState) -> dict:
    thread_id = state.get("thread_id")

    cached_chunks = _lru_get(CHUNK_CACHE, thread_id)
    if cached_chunks is not None:
        return {"chunks": cached_chunks}

    documents = state.get("docs", [])
    if not documents:
        return {"chunks": []}

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        divided_chunks = text_splitter.split_documents(documents)
        _lru_set(CHUNK_CACHE, thread_id, divided_chunks, MAX_THREADS_CACHED)
        return {"chunks": divided_chunks}
    except Exception as e:
        print(f"Text split error: {e}")
        return {"chunks": []}


def Embeddings(state: AgentState) -> dict:
    thread_id = state["thread_id"]
    chunks_to_embed = state.get("chunks", [])

    if _lru_get(RETRIEVER_STORE, thread_id) is not None:
        return {}

    if not chunks_to_embed:
        return {}

    try:
        persist_dir = os.path.join(PERSIST_ROOT, thread_id)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        vector_store = Chroma.from_documents(
            documents=chunks_to_embed,
            embedding=EMBEDDING_MODEL,
            collection_name=f"session_{thread_id}",
            persist_directory=persist_dir,
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        bm25_retriever = BM25Retriever.from_documents(chunks_to_embed)
        bm25_retriever.k = 5

        hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

        # Evict oldest thread's on-disk Chroma folder too, not just the
        # in-memory reference, so ./chroma_store doesn't grow forever.
        if len(RETRIEVER_STORE) >= MAX_THREADS_CACHED:
            oldest_thread_id, _ = next(iter(RETRIEVER_STORE.items()))
            old_dir = os.path.join(PERSIST_ROOT, oldest_thread_id)
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir, ignore_errors=True)

        _lru_set(RETRIEVER_STORE, thread_id, hybrid_retriever, MAX_THREADS_CACHED)
        return {}

    except Exception as e:
        print(f"Embedding error: {e}")
        return {}


async def contextualize_question(state: AgentState) -> dict:
    history = state.get("history", [])
    question = state.get("question", "")

    if not history:
        return {}

    recent = history[-3:]
    history_text = "\n".join(
        f"Q: {h['question']}\nA: {h['answer']}" for h in recent
    )

    prompt = f"""Given this conversation history and a new question, rewrite the new question to be fully standalone and self-contained, resolving any pronouns or references (e.g. "that", "it", "this") using the history. If the question is already standalone, return it unchanged. Return ONLY the rewritten question, nothing else.

History:
{history_text}

New question: {question}

Standalone question:"""

    try:
        response = await llm.ainvoke(prompt, max_tokens=100)
        rewritten = str(response.content).strip()
        if rewritten:
            return {"question": rewritten}
    except Exception as e:
        logging.exception("Question reformulation failed: %s", e)

    return {}


def retriever(state: AgentState) -> dict:
    thread_id = state["thread_id"]
    question = state.get("question", "")

    actual_retriever = _lru_get(RETRIEVER_STORE, thread_id)
    if not actual_retriever:
        return {"retrieved_docs": state.get("chunks", [])[:3]}

    try:
        matched_documents = actual_retriever.invoke(question)
        if not matched_documents:
            return {"retrieved_docs": state.get("chunks", [])[:3]}
    except Exception as e:
        print(f"Retrieval error: {e}")
        return {"retrieved_docs": state.get("chunks", [])[:3]}

    try:
        passages = [
            {"id": i, "text": doc.page_content}
            for i, doc in enumerate(matched_documents)
        ]
        rerank_request = RerankRequest(query=question, passages=passages)
        reranked = RERANKER.rerank(rerank_request)

        id_to_doc = {i: doc for i, doc in enumerate(matched_documents)}
        top_k = reranked[:3]
        reordered_docs = [id_to_doc[r["id"]] for r in top_k]

        return {"retrieved_docs": reordered_docs}
    except Exception as e:
        print(f"Reranking error: {e}")
        return {"retrieved_docs": matched_documents[:3]}


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "rate_limit_exceeded" in msg


async def _call_llm_with_retry(prompt: str, primary_tokens: int, retry_tokens: int, max_retries: int = 2):
    last_error = None
    for attempt in range(max_retries + 1):
        tokens_for_this_attempt = primary_tokens if attempt == 0 else retry_tokens
        try:
            response = await llm.ainvoke(prompt, max_tokens=tokens_for_this_attempt)
            return response
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e):
                wait_time = 2 * (attempt + 1)
                logging.warning(
                    "Rate limited (attempt %s/%s), retrying in %ss with max_tokens=%s",
                    attempt + 1, max_retries + 1, wait_time, retry_tokens,
                )
                await asyncio.sleep(wait_time)
                continue
            raise
    raise last_error


async def retrieval(state: AgentState) -> dict:
    question = state.get("question", "")
    thread_id = state.get("thread_id")
    documents = state.get("retrieved_docs", [])

    cache_key = (thread_id, question)
    cached_answer = _lru_get(ANSWER_CACHE, cache_key)
    if cached_answer is not None:
        return {"retrieval": cached_answer}

    if not documents:
        answer = NOT_RELEVANT_MESSAGE
        return {"retrieval": answer, "history": [{"question": question, "answer": answer}]}

    MAX_CONTEXT_CHARS = 4000
    content = " ".join(doc.page_content for doc in documents)[:MAX_CONTEXT_CHARS]

    prompt = f"""You are an expert AI research assistant. Answer the question using STRICTLY and ONLY the information present in the context below.

CRITICAL INSTRUCTIONS:
- Broad or general questions like "what is this document about", "summarize this", "what does the document say", or similar are ALWAYS considered relevant as long as the context below has any content — treat these as a request to summarize the context provided.
- Only respond with "{NOT_RELEVANT_MESSAGE}" if the question is truly unrelated to the context (e.g. random text, greetings like "hello" or "sorry", or asking about a topic that does not appear anywhere in the context at all).
- If the context is relevant, only use facts explicitly present in it. Do NOT use outside knowledge, do NOT infer beyond what is written, and do NOT fabricate details.
- If the context only partially answers the question, answer only the supported part and explicitly state what is not covered.
- Match the length and detail of your answer to what the question actually asks for. If the user specifies a word or length limit, respect it exactly. If the question is broad, a few clear paragraphs is enough. Do not pad the answer.
- Write the answer as clean, natural plain-text prose in well-formed paragraphs.
- Do NOT use any markdown formatting at all: no asterisks, no bold, no italics, no bullet points, no numbered lists, no headings, no hashtags, no underscores.
- Do not output any thinking steps, scratchpad text, or <think></think> tags. Output ONLY the final answer text.

Context:
{content}

Question: {question}
Answer:"""

    try:
        response = await _call_llm_with_retry(prompt, MAX_OUTPUT_TOKENS, RETRY_TOKENS)
        final_text = str(response.content).strip()
        if "</think>" in final_text:
            final_text = final_text.split("</think>")[-1].strip()

        final_text = (
            final_text.replace("**", "")
            .replace("__", "")
            .replace("###", "")
            .replace("##", "")
            .replace("#", "")
        )

        if not final_text:
            final_text = NOT_RELEVANT_MESSAGE

        _lru_set(ANSWER_CACHE, cache_key, final_text, MAX_ANSWERS_CACHED)
        return {
            "retrieval": final_text,
            "history": [{"question": question, "answer": final_text}],
        }
    except Exception as e:
        logging.exception("LLM call failed: %s", e)
        if _is_rate_limit_error(e):
            fallback = BUSY_MESSAGE
        else:
            fallback = "Something went wrong while generating the answer. Please try again."
        return {"retrieval": fallback, "history": [{"question": question, "answer": fallback}]}


graph = StateGraph(AgentState)

graph.add_node("load_docs", load_documents)
graph.add_node("text_split", text_split)
graph.add_node("embeddings", Embeddings)
graph.add_node("contextualize", contextualize_question)
graph.add_node("retriever", retriever)
graph.add_node("retrieval", retrieval)

graph.add_edge(START, "load_docs")
graph.add_edge("load_docs", "text_split")
graph.add_edge("text_split", "embeddings")
graph.add_edge("embeddings", "contextualize")
graph.add_edge("contextualize", "retriever")
graph.add_edge("retriever", "retrieval")
graph.add_edge("retrieval", END)

memory = InMemorySaver()
model = graph.compile(checkpointer=memory)


@traceable(name="run_query")
async def run_query(file_path: str, question: str, thread_id: str | None = None) -> dict:
    thread_id = thread_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["rag-api"],
        "metadata": {"thread_id": thread_id, "file_path": file_path},
    }
    initial_state = {
        "thread_id": thread_id,
        "file_path": file_path,
        "question": question,
    }
    t0 = time.perf_counter()
    result = await model.ainvoke(initial_state, config=config)
    elapsed = time.perf_counter() - t0
    print(f"[thread {thread_id}] answered in {elapsed:.2f}s")
    return result