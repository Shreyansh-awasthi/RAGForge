# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from fastapi.responses import StreamingResponse
# from typing import Optional
# import uvicorn
# import logging
# import time
# import traceback

# from Rag import model, load_documents, text_split, Embeddings, retriever, llm

# app = FastAPI(title="RAG API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     file_path: str
#     question: str
#     thread_id: Optional[str] = None


# class QueryResponse(BaseModel):
#     thread_id: str
#     answer: str
#     elapsed_seconds: float


# @app.post("/query", response_model=QueryResponse)
# async def query_endpoint(payload: QueryRequest):

#     if not payload.thread_id:
#         raise HTTPException(
#             status_code=400,
#             detail="thread_id is required — generate one per session on the frontend and reuse it for follow-up questions.",
#         )

#     config = {"configurable": {"thread_id": payload.thread_id}}
#     initial_state = {
#         "thread_id": payload.thread_id,
#         "file_path": payload.file_path,
#         "question": payload.question,
#     }

#     t0 = time.perf_counter()
#     try:
#         result = await model.ainvoke(initial_state, config=config)
#     except Exception as e:
#         logging.exception("Pipeline failed: %s", e)
#         raise HTTPException(status_code=500, detail=str(e))
#     elapsed = time.perf_counter() - t0

#     return QueryResponse(
#         thread_id=payload.thread_id,
#         answer=result.get("retrieval", ""),
#         elapsed_seconds=round(elapsed, 2),
#     )


# @app.post("/query/stream")
# async def query_stream(payload: QueryRequest):
#     if not payload.thread_id:
#         raise HTTPException(status_code=400, detail="thread_id is required")

#     state = {
#         "thread_id": payload.thread_id,
#         "file_path": payload.file_path,
#         "question": payload.question,
#     }

#     try:
#         state.update(load_documents(state))
#         state.update(text_split(state))
#         state.update(Embeddings(state))
#         state.update(retriever(state))
#     except Exception as e:
#         traceback.print_exc()  # prints full error to your terminal
#         raise HTTPException(status_code=500, detail=str(e))

#     documents = state.get("retrieved_docs", [])
#     content = " ".join(doc.page_content for doc in documents)[:6000]
#     prompt = f"Answer the question using only this context:\n{content}\n\nQuestion: {payload.question}\nAnswer:"

#     async def generate():
#         try:
#             async for chunk in llm.astream(prompt):
#                 if chunk.content:
#                     yield chunk.content
#         except Exception as e:
#             traceback.print_exc()
#             yield f"\n[stream error: {e}]"

#     return StreamingResponse(generate(), media_type="text/plain")


# @app.get("/health")
# async def health():
#     return {"status": "ok"}

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from fastapi.responses import StreamingResponse
# from typing import Optional
# import uvicorn
# import traceback
# import os

# try:
#     from .Rag import load_documents, text_split, Embeddings, retriever, llm
# except (ModuleNotFoundError, ImportError):
#     from Rag import load_documents, text_split, Embeddings, retriever, llm

# app = FastAPI(title="RAG API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class QueryRequest(BaseModel):
#     file_path: str
#     question: str
#     thread_id: Optional[str] = None


# @app.post("/query/stream")
# async def query_stream(payload: QueryRequest):
#     if not payload.thread_id:
#         raise HTTPException(status_code=400, detail="thread_id is required")

#     file_path = payload.file_path.strip().strip('"').strip("'")

#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

#     state = {
#         "thread_id": payload.thread_id,
#         "file_path": file_path,
#         "question": payload.question,
#     }

#     try:
#         state.update(load_documents(state))
#         state.update(text_split(state))
#         state.update(Embeddings(state))
#         state.update(retriever(state))
#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

#     documents = state.get("retrieved_docs", [])
#     if not documents:
#         async def empty():
#             yield "No relevant content found in the document."
#         return StreamingResponse(empty(), media_type="text/plain")

#     content = " ".join(doc.page_content for doc in documents)[:6000]
#     prompt = f"Answer the question using only this context:\n{content}\n\nQuestion: {payload.question}\nAnswer:"

#     async def generate():
#         try:
#             async for chunk in llm.astream(prompt):
#                 if chunk.content:
#                     yield chunk.content
#         except Exception as e:
#             traceback.print_exc()
#             yield f"\n[stream error: {e}]"

#     return StreamingResponse(generate(), media_type="text/plain")


# @app.get("/health")
# async def health():
#     return {"status": "ok"}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# LangSmith env vars MUST load before any LangChain-based imports
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

print("TRACING:", os.getenv("LANGSMITH_TRACING"))
print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))
print("KEY SET:", bool(os.getenv("LANGSMITH_API_KEY")))

import asyncio
import logging
import traceback
import inspect

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

try:
    from .Rag import run_query
except (ModuleNotFoundError, ImportError):
    from Rag import run_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_TIMEOUT_SECONDS = 90


class QueryRequest(BaseModel):
    file_path: str
    question: str
    thread_id: Optional[str] = None


@app.post("/query/stream")
async def query_stream(payload: QueryRequest):
    if not payload.thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    file_path = payload.file_path.strip().strip('"').strip("'")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    try:
        if inspect.iscoroutinefunction(run_query):
            coro = run_query(file_path, payload.question, payload.thread_id)
        else:
            loop = asyncio.get_event_loop()
            coro = loop.run_in_executor(
                None, run_query, file_path, payload.question, payload.thread_id
            )

        try:
            result = await asyncio.wait_for(coro, timeout=REQUEST_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.error(
                "run_query timed out after %ss for thread_id=%s",
                REQUEST_TIMEOUT_SECONDS,
                payload.thread_id,
            )
            raise HTTPException(
                status_code=504,
                detail=f"Query timed out after {REQUEST_TIMEOUT_SECONDS}s. Try again or check Groq API status.",
            )

        if not isinstance(result, dict):
            raise ValueError(
                f"run_query returned {type(result).__name__}, expected dict. Got: {result!r}"
            )

        return {"answer": result.get("retrieval", "")}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["chroma_store/*", "*.sqlite3"],
    )