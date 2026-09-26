"""Run end-to-end evaluation on 200 RAG test cases and export for LLM-as-a-judge.

Evaluates the exact 200 test cases sampled in the retrieval evaluation phase:
- 50 cases from each category: body-part, disease, drug, medicine (seed 42).
- Retrieves top-K chunks via the retrieval engine (Hybrid / Dense / Sparse).
- Generates answer using the configured LLM (e.g., Qwen on Cloudflare/vLLM, Groq, OpenRouter).
- Saves results to a structured .jsonl file for downstream LLM-as-a-judge evaluation
  (measuring Faithfulness, Answer Relevance, Context Relevance, Groundedness).
"""
import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
import urllib.request
import re
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from medical_rag.generation import SYSTEM_PROMPT, build_messages
from medical_rag.llm import chat
from medical_rag.retrieval import Hit
from validate_chunking import slug_from_url

def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s or ""))
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def retrieve_via_api(api_url: str, question: str, mode: str = "hybrid", k: int = 5) -> list[dict]:
    """Retrieve top-K chunks using the running FastAPI service."""
    url = f"{api_url.rstrip('/')}/api/retrieve"
    payload = json.dumps({"question": question, "mode": mode, "k": k}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        data = json.loads(res.read().decode("utf-8"))
        return data.get("hits", [])


def process_case(item: dict, api_url: str, mode: str, k: int, model_name: str | None, base_url: str | None = None) -> dict:
    """Process a single QA case: retrieve chunks, generate answer, and score retrieval hit."""
    question = item["question"]
    ground_truth_context = clean_text(item.get("context", ""))
    gt_slug = slug_from_url(item.get("article_url", ""))
    article_type = item.get("topic", "")

    t0 = time.time()
    raw_hits = retrieve_via_api(api_url, question, mode=mode, k=k)
    retrieve_time = time.time() - t0

    # Format chunks & check ground truth hit
    formatted_chunks = []
    hits_for_prompt = []
    first_hit_rank = None
    hit_rate = 0

    for idx, h in enumerate(raw_hits, 1):
        norm_text = clean_text(h.get("text", ""))
        is_gt = bool(ground_truth_context and ground_truth_context in norm_text)
        if is_gt and first_hit_rank is None:
            first_hit_rank = idx
            hit_rate = 1

        formatted_chunks.append({
            "rank": idx,
            "id": h.get("id"),
            "title": h.get("article_title"),
            "url": h.get("article_url"),
            "section_path": h.get("section_path"),
            "type": h.get("type"),
            "retrieval_score": h.get("retrieval_score"),
            "rerank_score": h.get("rerank_score"),
            "is_ground_truth": is_gt,
            "text": h.get("text", "")
        })

        hits_for_prompt.append(Hit(
            id=str(h.get("id")),
            text=h.get("text", ""),
            type=h.get("type", ""),
            article_title=h.get("article_title", ""),
            article_url=h.get("article_url", ""),
            section_path=h.get("section_path", ""),
            retrieval_score=h.get("retrieval_score", 0.0),
            score=h.get("rerank_score")
        ))

    # Generate answer with LLM
    t_gen_start = time.time()
    messages = build_messages(question, hits_for_prompt)
    try:
        llm_answer = chat(messages, model=model_name, base_url=base_url)
    except Exception as e:
        log.error("LLM generation error for question '%s': %s", question[:50], e)
        llm_answer = f"ERROR: {e}"
    gen_time = time.time() - t_gen_start

    return {
        "id": f"{item['topic']}_{item['sample_idx']}",
        "sample_idx": item["sample_idx"],
        "category": item["topic"],
        "question_idx": item.get("question_idx"),
        "question": question,
        "ground_truth_context": item.get("context", ""),
        "ground_truth_answer": item.get("answer", ""),
        "reference_article_title": item.get("title", ""),
        "reference_article_url": item.get("article_url", ""),
        "retrieval_mode": mode,
        "k": k,
        "hit_rate_at_k": hit_rate,
        "first_hit_rank": first_hit_rank,
        "retrieved_chunks": formatted_chunks,
        "llm_model": model_name or os.environ.get("LLM_MODEL", "default"),
        "llm_answer": llm_answer,
        "timing": {
            "retrieval_sec": round(retrieve_time, 3),
            "generation_sec": round(gen_time, 3),
            "total_sec": round(retrieve_time + gen_time, 3)
        }
    }


def load_dataset(test_case_dir: str, per_type: int = 50, seed: int = 42) -> list[dict]:
    """Load and sample 50 cases from each category CSV."""
    dataset = []
    global_idx = 0
    for path in sorted(Path(test_case_dir).glob("*.csv")):
        frame = pd.read_csv(path, encoding="utf-8")
        if frame.empty:
            continue
        sample = frame.sample(n=min(per_type, len(frame)), random_state=seed)
        topic = path.stem
        for _, row in sample.iterrows():
            record = row.to_dict()
            record["topic"] = topic
            record["sample_idx"] = global_idx
            dataset.append(record)
            global_idx += 1
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end RAG eval and export for LLM judge.")
    parser.add_argument("--test-case-dir", default="test_case", help="Directory containing test CSVs")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI backend URL")
    parser.add_argument("--output", default="eval_results/rag_e2e_200_cases.jsonl", help="Output JSONL path")
    parser.add_argument("--per-type", type=int, default=50, help="Samples per category")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "sparse"])
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel worker threads")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for dry-run testing")
    parser.add_argument("--llm-model", default=None, help="Override LLM model name")
    parser.add_argument("--llm-base-url", default=None, help="Override LLM base URL")

    args = parser.parse_args()

    # Verify backend is running
    try:
        with urllib.request.urlopen(f"{args.api_url}/api/health", timeout=5) as res:
            health_data = json.loads(res.read().decode())
            log.info("Backend health OK: %s chunks indexed", health_data.get("points"))
    except Exception as e:
        log.error("Failed to connect to backend at %s: %s", args.api_url, e)
        print(f"Error: Backend is not reachable at {args.api_url}. Please ensure FastAPI is running.")
        return 1

    # Load 200 cases
    dataset = load_dataset(args.test_case_dir, per_type=args.per_type, seed=args.seed)
    if args.limit:
        dataset = dataset[:args.limit]

    total = len(dataset)
    log.info("Loaded %d test cases for evaluation (per_type=%d, seed=%d)", total, args.per_type, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = args.llm_base_url or os.environ.get("LLM_BASE_URL")
    model_name = args.llm_model or os.environ.get("LLM_MODEL", "default")
    log.info("Starting evaluation with LLM model: %s | Base URL: %s | Mode: %s | K: %d | Concurrency: %d",
             model_name, base_url, args.mode, args.k, args.concurrency)

    results = []
    completed = 0
    t_start = time.time()

    # Open output file and write progressively
    with open(output_path, "w", encoding="utf-8") as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(process_case, item, args.api_url, args.mode, args.k, model_name, base_url): item
                for item in dataset
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f_out.flush()
                    completed += 1
                    
                    elapsed = time.time() - t_start
                    eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
                    sys.stdout.write(
                        f"\r[{completed}/{total}] ({completed*100/total:5.1f}%) "
                        f"Hit: {'YES' if res['hit_rate_at_k'] else ' NO'} | "
                        f"Total time: {elapsed:.0f}s (ETA: {eta:.0f}s) | "
                        f"Cat: {res['category']:10s} Q: {res['question'][:30]}..."
                    )
                    sys.stdout.flush()
                except Exception as e:
                    completed += 1
                    log.error("Error processing case: %s", e)

    print("\n" + "=" * 80)
    total_time = time.time() - t_start
    hits = sum(r["hit_rate_at_k"] for r in results)
    avg_gen_time = sum(r["timing"]["generation_sec"] for r in results) / len(results) if results else 0
    print(f"EVALUATION COMPLETE")
    print(f"- Total cases: {len(results)}/{total}")
    print(f"- HitRate@{args.k}: {hits}/{len(results)} ({hits*100/len(results):.1f}%)")
    print(f"- Total runtime: {total_time:.2f}s (avg {avg_gen_time:.2f}s / LLM answer)")
    print(f"- Output JSONL: {output_path.resolve()}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
