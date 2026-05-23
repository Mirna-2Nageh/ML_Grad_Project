"""CLI re-run of the Streamlit tab5 evaluation.

Loads the questions from evaluation_results(3).csv (col 0) and POSTs each to
/api/v1/qa with the same payload + timeout as the Streamlit panel, then writes
the new results to evaluation_results_v4.csv with identical column shape so
direct row-by-row comparison is possible.
"""
import csv
import json
import sys
import time
from pathlib import Path

import requests

API_BASE = "http://localhost:8000/api/v1"
TIMEOUT_S = 180
# Groq free tier has a 6000 token/minute limit and each retry-on-hallucination
# burns ~4000 tokens; 4s between requests keeps the TPM bucket refilled.
SLEEP_BETWEEN_S = 4.0
HERE = Path(__file__).parent

IN_CSV = HERE / "evaluation_results(3).csv"
OUT_CSV = HERE / "evaluation_results_v8.csv"

COLS = [
    "السؤال", "الإجابة", "الثقة %", "تحقق المواد", "تنبيهات",
    "المصادر", "زمن API (ms)", "زمن إجمالي (ms)", "زمن استرجاع (ms)",
]


def load_questions(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row[reader.fieldnames[0]] for row in reader if row[reader.fieldnames[0]].strip()]


def query(q: str):
    t0 = time.time()
    try:
        r = requests.post(f"{API_BASE}/qa", json={"question": q, "k": 7}, timeout=TIMEOUT_S)
        total_t = (time.time() - t0) * 1000
        data = r.json()
        factors = data.get("confidence_factors") or {}
        return {
            "السؤال": q[:80],
            "الإجابة": (data.get("answer", "") or "")[:150],
            "الثقة %": int(round((data.get("confidence_score", 0) or 0) * 100)),
            "تحقق المواد": factors.get("article_validation", "—"),
            "تنبيهات": len(data.get("warnings", []) or []),
            "المصادر": len(data.get("sources", []) or []),
            "زمن API (ms)": round(data.get("latency_ms", 0)),
            "زمن إجمالي (ms)": round(total_t),
            "زمن استرجاع (ms)": round(data.get("retrieval_ms", 0)),
            "_status_code": r.status_code,
            "_warnings_full": data.get("warnings", []),
            "_model": data.get("model", ""),
        }
    except Exception as e:
        return {
            "السؤال": q[:80], "الإجابة": f"خطأ: {e}",
            "الثقة %": 0, "تحقق المواد": "—", "تنبيهات": 0,
            "المصادر": 0, "زمن API (ms)": 0,
            "زمن إجمالي (ms)": round((time.time() - t0) * 1000),
            "زمن استرجاع (ms)": 0,
            "_status_code": -1, "_warnings_full": [], "_model": "",
        }


def main():
    questions = load_questions(IN_CSV)
    print(f"Loaded {len(questions)} questions from {IN_CSV.name}", flush=True)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i:2d}/{len(questions)}] {q[:90]}", flush=True)
        res = query(q)
        results.append(res)
        marker = {
            "passed": "OK",
            "failed": "HALLUCINATION",
            "—": "(no-context)",
        }.get(res["تحقق المواد"], res["تحقق المواد"])
        gated = "[INPUT_GATE]" if res["_model"] == "input_gate" else ""
        print(
            f"         conf={res['الثقة %']}% | art_val={marker} | "
            f"warnings={res['تنبيهات']} | code={res['_status_code']} {gated}",
            flush=True,
        )
        if res["_warnings_full"]:
            for w in res["_warnings_full"]:
                print(f"           ⚠ {w[:150]}", flush=True)
        time.sleep(SLEEP_BETWEEN_S)

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in COLS})

    # Stats
    confs = [r["الثقة %"] for r in results]
    n_pass = sum(1 for r in results if r["تحقق المواد"] == "passed")
    n_fail = sum(1 for r in results if r["تحقق المواد"] == "failed")
    n_gated = sum(1 for r in results if r["_model"] == "input_gate")
    n_err = sum(1 for r in results if r["_status_code"] not in (200, -1) or r["الإجابة"].startswith("خطأ"))
    print()
    print("=" * 60)
    print(f"Wrote {OUT_CSV.name}")
    print(f"Total:           {len(results)}")
    print(f"Passed art_val:  {n_pass} ({100*n_pass/len(results):.1f}%)")
    print(f"Failed art_val:  {n_fail} ({100*n_fail/len(results):.1f}%)")
    print(f"Input-gated:     {n_gated}")
    print(f"Errors:          {n_err}")
    print(f"Avg confidence:  {sum(confs)/len(confs):.1f}%")
    print(f"Min/Max conf:    {min(confs)}% / {max(confs)}%")


if __name__ == "__main__":
    sys.exit(main())
