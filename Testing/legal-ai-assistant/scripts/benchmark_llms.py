"""
Benchmark our RAG system vs. OpenAI GPT-4o-mini and Anthropic Claude Sonnet 4.5.

All three models are routed through OpenRouter — no extra API keys needed beyond
OPENROUTER_API_KEY. Model IDs come from config.BENCHMARK_MODELS (configurable via env).

Two modes:
  --mode rag (default):  All three models receive the same retrieved context our
                         pipeline produces. Isolates "given equal evidence, which
                         model writes the best answer?"
  --mode raw:            Each model answers from its own knowledge — no retrieval.
                         Tests "how much do general-purpose models know about
                         Egyptian Criminal Law without RAG?"

Usage:
    python scripts/benchmark_llms.py
    python scripts/benchmark_llms.py --mode raw
    python scripts/benchmark_llms.py --questions-file my_qs.txt --output out.csv
"""
import os
import sys
import csv
import time
import argparse
import logging
from typing import List, Tuple

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from app.services.llm import get_client
from app.services.retrieval import retrieval_service
from app.services.reranker import reranker_service
from app.services.preprocessing import extract_article_references
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_QUESTIONS = [
    "ما هي عقوبة السرقة بالإكراه في القانون المصري؟",
    "ما هي حالات الدفاع الشرعي في قانون العقوبات المصري؟",
    "ما هي شروط التوقيف الاحتياطي في القانون المصري؟",
    "ما هي أركان جريمة القتل العمد في القانون الجنائي المصري؟",
    "ما هي عقوبة التزوير في المحررات الرسمية؟",
]


def query_model(model: str, question: str, context: str = None) -> Tuple[str, float]:
    """Call a model via OpenRouter. Returns (answer, latency_seconds)."""
    client = get_client()
    if context:
        prompt = PROMPTS["qa_restrictive"].format(context=context, question=question)
    else:
        prompt = question
    system = SYSTEM_MESSAGES["qa"]

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.001,
            max_tokens=1024,
        )
        text = (response.choices[0].message.content or "").strip()
        return text, time.time() - t0
    except Exception as e:
        logger.warning(f"{model} failed: {e}")
        return f"[ERROR: {e}]", time.time() - t0


def benchmark(questions: List[str], mode: str) -> List[dict]:
    """Run questions through ours/openai/claude. Returns list of result dicts."""
    if mode == "rag" and not retrieval_service.is_loaded:
        logger.info("Loading retrieval service for benchmark...")
        retrieval_service.load()
        reranker_service.load()

    results = []
    for i, q in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {q[:60]}...")
        row = {"question": q, "mode": mode}

        context = None
        if mode == "rag":
            contexts, sources, _ = retrieval_service.retrieve(q)
            context = "\n---\n".join(contexts)[:config.MAX_CONTEXT_CHARS]
            row["context_sources"] = len(sources)
        else:
            row["context_sources"] = 0

        for label, model in config.BENCHMARK_MODELS.items():
            ans, lat = query_model(model, q, context=context)
            row[f"{label}_answer"] = ans
            row[f"{label}_latency_s"] = round(lat, 2)
            row[f"{label}_length"] = len(ans)
            row[f"{label}_articles"] = ", ".join(extract_article_references(ans))

        results.append(row)
    return results


def write_csv(results: List[dict], path: str) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: List[dict]) -> None:
    """Print a quick numeric summary to stdout."""
    if not results:
        return

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Questions:       {len(results)}")
    print(f"  Mode:            {results[0].get('mode')}")
    for label in config.BENCHMARK_MODELS:
        lats = [r.get(f"{label}_latency_s", 0) for r in results]
        lens = [r.get(f"{label}_length", 0) for r in results]
        n_arts = [len((r.get(f"{label}_articles") or "").split(",")) if r.get(f"{label}_articles") else 0 for r in results]
        errs = sum(1 for r in results if (r.get(f"{label}_answer") or "").startswith("[ERROR"))
        print(f"  {label:<8} model={config.BENCHMARK_MODELS[label]:<35} "
              f"avg_lat={mean(lats):.2f}s  avg_len={mean(lens):.0f}  avg_articles={mean(n_arts):.1f}  errors={errs}")


def main():
    parser = argparse.ArgumentParser(description="Compare our RAG vs OpenAI vs Claude (via OpenRouter).")
    parser.add_argument("--mode", choices=["rag", "raw"], default="rag",
                        help="rag: shared retrieved context for all 3. raw: each model answers from its own knowledge.")
    parser.add_argument("--questions-file", help="Optional file with one question per line.")
    parser.add_argument("--output", default="benchmark_results.csv", help="Output CSV path.")
    args = parser.parse_args()

    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    print("=" * 60)
    print(f"🔬 LLM Benchmark — mode={args.mode}, {len(questions)} questions")
    print("=" * 60)
    for label, model in config.BENCHMARK_MODELS.items():
        print(f"  • {label:<8} {model}")
    print()

    results = benchmark(questions, args.mode)
    write_csv(results, args.output)
    print(f"\n✅ {len(results)} rows → {args.output}")
    print_summary(results)


if __name__ == "__main__":
    main()
