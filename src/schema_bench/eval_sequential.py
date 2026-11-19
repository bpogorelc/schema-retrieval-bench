from __future__ import annotations

import time
from typing import Any, Sequence

from schema_bench.catalog import SchemaDoc, load_catalog
from schema_bench.labels import Question, QuestionBank, load_questions
from schema_bench.metrics import ndcg_at_k, recall_at_k
from schema_bench.retrievers.bm25 import BM25Retriever
from schema_bench.retrievers.dense import DenseRetriever
from schema_bench.retrievers.hybrid import HybridRetriever, tune_hybrid_on_dev
from schema_bench.threading_utils import cap_cpu_threads


def build_retrievers(
    docs: Sequence[SchemaDoc],
    dev_questions: Sequence[Question] | None = None,
    tune_hybrid: bool = True,
) -> dict[str, Any]:
    cap_cpu_threads(1)
    bm25 = BM25Retriever(docs)
    dense = DenseRetriever(docs)
    if tune_hybrid and dev_questions:
        hybrid = tune_hybrid_on_dev(bm25, dense, dev_questions)
    else:
        hybrid = HybridRetriever(bm25, dense)
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}


def evaluate_questions(
    questions: Sequence[Question],
    retrievers: dict[str, Any],
    top_k: int = 5,
    workload: str = "quality",
    topology: str = "local_sequential",
    run_id: str = "",
    repeat_index: int = 0,
    worker_id: str = "main",
    node_id: str = "local",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in questions:
        for method, retriever in retrievers.items():
            t0 = time.perf_counter()
            ranked = retriever.retrieve(q.text, top_k=top_k)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            ranked_ids = [doc_id for doc_id, _ in ranked]
            scores = [score for _, score in ranked]
            rows.append(
                {
                    "run_id": run_id,
                    "question_id": q.id,
                    "intent_family": q.intent_family,
                    "split": q.split,
                    "method": method,
                    "workload": workload,
                    "topology": topology,
                    "repeat_index": repeat_index,
                    "worker_id": worker_id,
                    "node_id": node_id,
                    "latency_ms": latency_ms,
                    "ndcg_at_5": ndcg_at_k(ranked_ids, q.relevant, k=top_k),
                    "recall_at_5": recall_at_k(ranked_ids, q.relevant, k=top_k),
                    "ranked_ids": ranked_ids,
                    "ranked_scores": scores,
                    "query_text": q.text,
                }
            )
    return rows


def run_sequential_eval(
    question_ids: list[str] | None = None,
    split: str | None = None,
    top_k: int = 5,
    tune_hybrid: bool = True,
    run_id: str = "sequential",
    workload: str = "quality",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    docs = load_catalog()
    bank = load_questions()
    if question_ids:
        questions = bank.by_ids(question_ids)
    elif split:
        questions = bank.by_split(split)  # type: ignore[arg-type]
    else:
        questions = bank.questions

    startup_t0 = time.perf_counter()
    retrievers = build_retrievers(
        docs,
        dev_questions=bank.by_split("dev"),
        tune_hybrid=tune_hybrid,
    )
    # Warm one query so model/index path is hot before timed loop.
    sample = questions[0].text if questions else "revenue"
    for retriever in retrievers.values():
        retriever.retrieve(sample, top_k=top_k)
    startup_s = time.perf_counter() - startup_t0

    warm_t0 = time.perf_counter()
    rows = evaluate_questions(
        questions,
        retrievers,
        top_k=top_k,
        workload=workload,
        topology="local_sequential",
        run_id=run_id,
    )
    warm_s = time.perf_counter() - warm_t0

    meta = {
        "run_id": run_id,
        "engine": "sequential",
        "topology": "local_sequential",
        "n_questions": len(questions),
        "startup_s": startup_s,
        "warm_eval_s": warm_s,
        "workers": 1,
        "batch_size": len(questions),
        "node_ids": ["local"],
        "label_status": bank.label_status,
        "hybrid_rrf_k": retrievers["hybrid"].rrf_k,
        "hybrid_bm25_weight": retrievers["hybrid"].bm25_weight,
        "hybrid_dense_weight": retrievers["hybrid"].dense_weight,
    }
    return rows, meta


def smoke_question_ids(bank: QuestionBank | None = None) -> list[str]:
    bank = bank or load_questions()
    # One exact, one paraphrase, one multi-entry.
    preferred = ["q06", "q02", "q22"]
    available = {q.id for q in bank.questions}
    return [qid for qid in preferred if qid in available]
