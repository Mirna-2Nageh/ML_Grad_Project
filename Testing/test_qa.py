"""Ad-hoc CLI test runner for the Legal AI Assistant.

Quick usage:
  # Single question — pass any Arabic legal question as argv:
  python test_qa.py "ما هي عقوبة السرقة بالإكراه في القانون المصري؟"

  # Full 28-question benchmark (uses evaluation_results(3).csv as source):
  python test_qa.py --all

  # 5-question quick smoke test (one from each category):
  python test_qa.py --quick

Exit code is 0 if all tested questions pass article-validation, else 1.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

API_BASE = "http://localhost:8000/api/v1"
TIMEOUT_S = 180
HERE = Path(__file__).parent

# Five representative questions covering the main failure modes the eval(v3-v8)
# exposed — picked one from each category so a 5-question run gives broad
# coverage in under 3 minutes.
QUICK_QUESTIONS = [
    # Substantive law, was failing across v3/v4 — fixed by v6 rescue.
    "ما هي حالات الدفاع الشرعي في قانون العقوبات المصري؟",
    # Procedural law, was citing wrong code in v3 — fixed by prompt rules.
    "ما هي شروط التوقيف الاحتياطي في القانون المصري؟",
    # Substantive — historically passes, sanity baseline.
    "ما هي عقوبة السرقة بالإكراه في القانون المصري؟",
    # Definitional — tests article-level grounding.
    "ما هي أركان جريمة القتل العمد في القانون الجنائي المصري؟",
    # Input gate — should be rejected before LLM call.
    "## هذا ليس سؤالاً",
]


def ask(question: str, k: int = 7) -> dict:
    """Send a single question to /api/v1/qa and return the parsed response."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{API_BASE}/qa", json={"question": question, "k": k}, timeout=TIMEOUT_S
        )
        data = r.json()
        data["_total_ms"] = round((time.time() - t0) * 1000)
        data["_http"] = r.status_code
        return data
    except requests.exceptions.ConnectionError:
        return {"_error": "Backend not reachable at " + API_BASE + " — start it via ./start_project.sh"}
    except Exception as e:
        return {"_error": f"Request failed: {e}"}


def render(question: str, data: dict, idx: int = None, total: int = None) -> bool:
    """Print one question's result. Returns True if it passed validation."""
    header = f"[{idx}/{total}] " if idx is not None else ""
    print()
    print("─" * 80)
    print(f"{header}{question[:120]}")
    print("─" * 80)

    if "_error" in data:
        print(f"  ❌ {data['_error']}")
        return False

    conf = int(round((data.get("confidence_score", 0) or 0) * 100))
    factors = data.get("confidence_factors") or {}
    art_val = factors.get("article_validation", "—")
    model = data.get("model", "?")
    n_sources = len(data.get("sources") or [])

    if art_val == "passed":
        verdict = "✅ PASS"
    elif art_val == "failed":
        verdict = "❌ HALLUCINATION"
    else:
        verdict = "·  (no articles cited)"

    print(f"  {verdict}  |  confidence={conf}%  |  model={model}")
    print(f"  sources={n_sources}  |  latency={data['_total_ms']}ms")
    print()

    # Answer (first 300 chars)
    ans = (data.get("answer") or "").strip()
    if ans:
        print("  ANSWER:")
        for line in ans[:400].splitlines()[:6]:
            print(f"    {line[:110]}")
        if len(ans) > 400:
            print("    ...")

    # Warnings (these are diagnostic gold — show all)
    warnings = data.get("warnings") or []
    if warnings:
        print()
        print("  WARNINGS:")
        for w in warnings:
            print(f"    ⚠ {w[:120]}")

    # Rescue marker — show explicitly when article-lookup rescue fired
    rescue_used = any(
        (s.get("source") or "") == "[article-lookup rescue]"
        for s in (data.get("sources") or [])
    )
    if rescue_used:
        print()
        print("  🔍 Article-lookup rescue triggered for this question.")

    return art_val != "failed"


def load_benchmark_questions() -> list:
    csv_path = HERE / "evaluation_results(3).csv"
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row[reader.fieldnames[0]] for row in reader if row[reader.fieldnames[0]].strip()]


def main():
    parser = argparse.ArgumentParser(description="Quick test runner for the Legal AI Assistant.")
    parser.add_argument("question", nargs="?", help="Arabic legal question (omit to use --all or --quick)")
    parser.add_argument("--all", action="store_true", help="Run the full 28-question benchmark")
    parser.add_argument("--quick", action="store_true", help="Run 5 representative questions")
    parser.add_argument("--k", type=int, default=7, help="Number of retrieval results (default 7)")
    args = parser.parse_args()

    # Choose what to test
    if args.all:
        questions = load_benchmark_questions()
        if not questions:
            print("ERROR: evaluation_results(3).csv not found — provide questions manually instead.")
            return 1
    elif args.quick:
        questions = QUICK_QUESTIONS
    elif args.question:
        questions = [args.question]
    else:
        parser.print_help()
        print()
        print("EXAMPLES:")
        print('  python test_qa.py "ما هي عقوبة السرقة؟"')
        print("  python test_qa.py --quick")
        print("  python test_qa.py --all")
        return 0

    # Health check first
    try:
        h = requests.get(f"{API_BASE}/health", timeout=5).json()
        print(f"✅ Backend healthy — {h.get('vectors', '?')} vectors loaded, model={h.get('model', '?')}")
    except Exception as e:
        print(f"❌ Backend not reachable at {API_BASE} ({e})")
        print("   Start it with: cd /home/marno000onaaa/Desktop/ML_Grad_Project/Testing && ./start_project.sh")
        return 1

    # Run questions
    n_total = len(questions)
    n_pass = n_fail = n_neutral = 0
    for i, q in enumerate(questions, 1):
        data = ask(q, k=args.k)
        passed = render(q, data, idx=i, total=n_total)

        # Tally — distinguish "passed validation" from "no citation"
        factors = (data.get("confidence_factors") or {}) if "_error" not in data else {}
        art_val = factors.get("article_validation", "—")
        if art_val == "passed":
            n_pass += 1
        elif art_val == "failed":
            n_fail += 1
        else:
            n_neutral += 1

        # Polite pacing for Groq free-tier TPM (6000/min)
        if n_total > 1 and i < n_total:
            time.sleep(4)

    # Summary
    print()
    print("=" * 80)
    print(f"SUMMARY: {n_total} questions")
    print(f"  ✅ Passed validation: {n_pass}  ({100*n_pass/n_total:.1f}%)")
    print(f"  ❌ Hallucinations:    {n_fail}  ({100*n_fail/n_total:.1f}%)")
    if n_neutral:
        print(f"  ·  No citations:      {n_neutral}  (e.g. input-gated or pure refusals)")
    print("=" * 80)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
