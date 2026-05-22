"""
Ad-hoc evaluation battery for the /qa endpoint.

Runs a fixed set of representative Arabic legal questions chosen to stress the
four optimization dimensions:
  - citation grounding   (warnings + article_validation factor)
  - legal accuracy        (judged by reading dump.json)
  - answer style/format   (length, structure — judged by reading dump.json)
  - latency               (latency_ms / model used)

Usage:
  python scripts/eval_battery.py                       # default: restrictive, k=7
  PROMPT_STYLE=standard K=7 python scripts/eval_battery.py
  python scripts/eval_battery.py --tag baseline        # label the run

Outputs a scorecard table to stdout and writes the full answers to
scripts/eval_runs/<tag>.json so legal quality / style can be read afterward.
"""
import json
import os
import sys
import time
import argparse
from datetime import datetime

import requests

API = os.getenv("API_BASE", "http://localhost:8000/api/v1")
PROMPT_STYLE = os.getenv("PROMPT_STYLE", "restrictive")
K = int(os.getenv("K", "7"))
# Gemini free tier is RPM-limited; pace requests so the battery doesn't trip 429/503.
SLEEP_BETWEEN_S = float(os.getenv("SLEEP_BETWEEN_S", "18"))

# id, scenario, question.  Scenario tells me what to check when reading answers.
BATTERY = [
    ("specific_theft",
     "specific factual — expects concrete articles + penalties",
     "ما هي عقوبة السرقة بالإكراه ليلاً مع حمل سلاح في القانون المصري؟"),
    ("ambiguous_forgery",
     "broad/ambiguous — should trigger AMBIGUITY PROTOCOL (overview + ask to clarify actor/doc type)",
     "ما عقوبة التزوير؟"),
    ("actor_distinction",
     "actor distinction — public official vs private person, official vs customary document",
     "ما الفرق في عقوبة تزوير محرر رسمي بين الموظف العام والفرد العادي؟"),
    ("cross_reference",
     "cross-reference synthesis — penalty articles that refer to other articles must be stitched",
     "ما عقوبة القتل العمد مع سبق الإصرار والترصد، وما الظروف المشددة؟"),
    ("out_of_scope",
     "out-of-scope — should honestly use the FALLBACK message, not invent",
     "ما هي قيمة ضريبة القيمة المضافة على السيارات المستوردة في مصر لعام 2024؟"),
    ("procedural",
     "procedural law — criminal procedure, not penal code",
     "ما هي مدة الحبس الاحتياطي المسموح بها قانوناً في الجنايات؟"),
    ("murder_penalty",
     "homicide expert rule — expects الإعدام (م230) for premeditated, distinguishes degrees",
     "ما عقوبة القتل العمد بدون سبق إصرار، وما الفرق عن القتل الخطأ؟"),
    ("assault_penalty",
     "assault expert rule — penalty graded by injury severity (م240-244)",
     "ما عقوبة الضرب الذي ينتج عنه عاهة مستديمة؟"),
]


def run_one(q):
    body = {"question": q, "k": K, "prompt_style": PROMPT_STYLE}
    t0 = time.time()
    r = requests.post(f"{API}/qa", json=body, timeout=180)
    wall = (time.time() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    d["_wall_ms"] = wall
    return d


def cited_articles(text):
    import re
    # crude: count "المادة <num>" mentions after assuming digits already western
    return sorted(set(re.findall(r"الماد[ةه]\s*(\d+)", text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="label for this run")
    args = ap.parse_args()
    tag = args.tag or f"{PROMPT_STYLE}_k{K}_{datetime.now():%H%M%S}"

    results = []
    print(f"\n=== eval battery | style={PROMPT_STYLE} k={K} | tag={tag} ===\n")
    hdr = f"{'id':<18} {'conf':>5} {'art_val':>9} {'topic':>5} {'src':>3} {'lat_s':>6} {'len':>5} {'warn':>4} {'model':<18}"
    print(hdr)
    print("-" * len(hdr))

    for i, (qid, scenario, q) in enumerate(BATTERY):
        if i > 0 and SLEEP_BETWEEN_S > 0:
            time.sleep(SLEEP_BETWEEN_S)
        try:
            d = run_one(q)
        except Exception as e:
            print(f"{qid:<18} ERROR: {e}")
            results.append({"id": qid, "scenario": scenario, "question": q, "error": str(e)})
            continue
        ans = d.get("answer", "")
        cf = d.get("confidence_factors", {}) or {}
        row = {
            "id": qid,
            "scenario": scenario,
            "question": q,
            "answer": ans,
            "confidence_score": d.get("confidence_score"),
            "confidence_factors": cf,
            "warnings": d.get("warnings", []),
            "sources": [
                {"filename": s.get("filename"), "article": s.get("article"),
                 "legal_topic": s.get("legal_topic"), "rerank_score": s.get("rerank_score")}
                for s in d.get("sources", [])
            ],
            "cited_articles_in_answer": cited_articles(ans),
            "latency_ms": d.get("latency_ms"),
            "model": d.get("model"),
            "answer_len": len(ans),
        }
        results.append(row)
        print(f"{qid:<18} {row['confidence_score']!s:>5} "
              f"{cf.get('article_validation','-'):>9} "
              f"{str(cf.get('topic_match','-')):>5} "
              f"{len(row['sources']):>3} "
              f"{(row['latency_ms'] or 0)/1000:>6.1f} "
              f"{row['answer_len']:>5} "
              f"{len(row['warnings']):>4} "
              f"{(row['model'] or '')[:18]:<18}")

    out_dir = os.path.join(os.path.dirname(__file__), "eval_runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "style": PROMPT_STYLE, "k": K, "ts": datetime.now().isoformat(),
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nfull answers → {out_path}")


if __name__ == "__main__":
    main()
