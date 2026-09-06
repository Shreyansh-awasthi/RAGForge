from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import asyncio
import logging
import traceback
import inspect

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

try:
    from .Rag import run_query
except (ModuleNotFoundError, ImportError):
    from Rag import run_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag-api")

logger.info("LANGSMITH_TRACING=%s", os.getenv("LANGSMITH_TRACING"))
logger.info("LANGSMITH_PROJECT=%s", os.getenv("LANGSMITH_PROJECT"))
logger.info("LANGSMITH_API_KEY set=%s", bool(os.getenv("LANGSMITH_API_KEY")))

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_TIMEOUT_SECONDS = 90
MAX_FILE_SIZE_MB = 20
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_CONCURRENT_QUERIES = 3
query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc)
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {type(exc).__name__}: {exc}"},
    )


@app.post("/query/stream")
async def query_stream(
    file: UploadFile = File(...),
    question: str = Form(...),
    thread_id: str = Form(...),
):
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed size is {MAX_FILE_SIZE_MB}MB.",
        )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    safe_filename = f"{thread_id}_{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        with open(saved_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        await asyncio.wait_for(query_semaphore.acquire(), timeout=30)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Server is busy handling other requests. Please try again shortly.",
        )

    try:
        try:
            if inspect.iscoroutinefunction(run_query):
                coro = run_query(saved_path, question, thread_id)
            else:
                loop = asyncio.get_event_loop()
                coro = loop.run_in_executor(
                    None, run_query, saved_path, question, thread_id
                )

            try:
                result = await asyncio.wait_for(coro, timeout=REQUEST_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.error(
                    "run_query timed out after %ss for thread_id=%s",
                    REQUEST_TIMEOUT_SECONDS,
                    thread_id,
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
            logger.exception("run_query failed for thread_id=%s", thread_id)
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    finally:
        query_semaphore.release()


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)