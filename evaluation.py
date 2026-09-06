import asyncio
import os
from langchain_groq import ChatGroq
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
import ragas

from Rag import model, EMBEDDING_MODEL

print(f"Detected ragas version: {ragas.__version__}")

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
    max_tokens=4000,
    max_retries=6,
)
ragas_llm = LangchainLLMWrapper(llm, bypass_n=True)
ragas_embeddings = LangchainEmbeddingsWrapper(EMBEDDING_MODEL)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

REQUIRED_COLUMNS = set()
for metric in METRICS:
    cols = getattr(metric, "required_columns", None) or getattr(metric, "_required_columns", None)
    if cols:
        if isinstance(cols, dict):
            for v in cols.values():
                REQUIRED_COLUMNS.update(v if isinstance(v, (list, set)) else [v])
        else:
            REQUIRED_COLUMNS.update(cols)

print(f"Metrics require these columns: {REQUIRED_COLUMNS}")

USE_NEW_SCHEMA = "user_input" in REQUIRED_COLUMNS or "reference" in REQUIRED_COLUMNS


EVAL_SET = [
    {
        "question": "What does rich mean according to the book?",
        "ground_truth": "Being rich means having assets that generate income without active work, as opposed to earning a paycheck.",
        "file_path": r"C:\Users\ayush\OneDrive\Desktop\Rich-Dad-Poor-Dad.pdf",
    },
    {
        "question": "Who is Rich Dad?",
        "ground_truth": "Rich Dad is the author's friend's father, who taught him about money and financial independence.",
        "file_path": r"C:\Users\ayush\OneDrive\Desktop\Rich-Dad-Poor-Dad.pdf",
    },
]


async def run_pipeline_for_eval(item, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "thread_id": thread_id,
        "file_path": item["file_path"],
        "question": item["question"],
    }
    result = await model.ainvoke(initial_state, config=config)

    retrieved_docs = result.get("retrieved_docs", [])
    contexts = [doc.page_content for doc in retrieved_docs] or [""]
    answer_text = result.get("retrieval", "") or ""

    print(f"\n--- Question: {item['question']} ---")
    print(f"Ground truth: {item['ground_truth']}")
    print(f"Answer: {answer_text[:200]}")
    for i, c in enumerate(contexts):
        print(f"Context[{i}]: {c[:200]}")

    if USE_NEW_SCHEMA:
        return {
            "user_input": item["question"],
            "response": answer_text,
            "retrieved_contexts": contexts,
            "reference": item["ground_truth"],
        }
    else:
        return {
            "question": item["question"],
            "answer": answer_text,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        }


async def build_eval_dataset():
    rows = []
    for i, item in enumerate(EVAL_SET):
        row = await run_pipeline_for_eval(item, thread_id=f"eval-{i}")
        rows.append(row)
        print(f"[{i+1}/{len(EVAL_SET)}] done: {item['question'][:50]}...")
    return Dataset.from_list(rows)


def run_evaluation():
    print(f"Using {'NEW' if USE_NEW_SCHEMA else 'OLD'} ragas schema for this dataset.")
    dataset = asyncio.run(build_eval_dataset())

    results = evaluate(
        dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            timeout=180,
            max_retries=8,
            max_wait=90,
            max_workers=1,
        ),
        raise_exceptions=True,
    )

    df = results.to_pandas()
    print(df.to_string())
    df.to_csv("ragas_eval_results.csv", index=False)
    print("\nSaved to ragas_eval_results.csv")

    print("\n--- Average scores ---")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in df.columns:
            print(f"{metric}: {df[metric].mean():.3f}")


if __name__ == "__main__":
    run_evaluation()