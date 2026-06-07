"""
Scored evaluation harness — the measurement backbone for tuning the system.

Unlike eval_battery.py (which dumps answers for a human to read) and
eval_retrieval.py (which only checks article presence), this harness produces a
single scored *scorecard* over a labeled gold set so any change — embeddings,
chunking, reranker-k, LLM — can be compared before/after with hard numbers.

Two modes (run either or both):

  retrieval  — LLM-free. Runs the production retrieval pipeline per gold query and
               scores whether the authoritative article(s) reached the context.
               Measures the *ceiling* on answer quality independent of the LLM and
               of free-tier rate limits. Always runnable, no server needed.

  e2e        — Hits the running API (POST /qa). Scores citation correctness,
               hallucination rate, out-of-scope abstention, confidence calibration,
               and latency. Needs the server up (uvicorn) + a working LLM provider.

Metrics (retrieval):
  recall@k            fraction of queries whose context contained ≥1 expected article
  article_recall      mean fraction of a query's expected articles found in context
  primary_hit@k       fraction of queries where a retrieved source *is* an expected
                      article (source.article match — tests article-aware chunking)
  mrr                 mean reciprocal rank of the first source carrying an expected article

Metrics (e2e):
  cite_recall         fraction of in-scope queries whose answer cited ≥1 expected article
  hallucination_rate  fraction of answers citing an article absent from the context
  abstention          fraction of OUT-of-scope queries the model correctly refused
  conf_correct/wrong  mean confidence on correct vs wrong answers (calibration gap)
  p50/p90 latency_ms

Usage:
  python scripts/eval_harness.py                      # retrieval mode (offline)
  python scripts/eval_harness.py --mode e2e --tag bge_m3
  python scripts/eval_harness.py --mode both --tag bge_m3
  python scripts/eval_harness.py --compare baseline_minilm bge_m3   # diff two scorecards
"""
import os
import sys
import json
import time
import argparse
import statistics
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.preprocessing import extract_article_references

OUT_DIR = os.path.join(os.path.dirname(__file__), "eval_runs")
API = os.getenv("API_BASE", "http://localhost:8000/api/v1")
# Pace e2e requests so the free-tier LLM provider doesn't trip 429/503.
SLEEP_BETWEEN_S = float(os.getenv("SLEEP_BETWEEN_S", "8"))

# Fallback / refusal phrasing the prompts emit for out-of-scope questions.
_REFUSAL_MARKERS = (
    "غير متوفر", "لا تتوفر", "لا يمكنني", "خارج نطاق", "لم أعثر",
    "لا توجد معلومات", "النصوص المقدمة لا",
)

# ── Gold set ──────────────────────────────────────────────────────────────────
# (id, query, expected_articles, in_scope, domain). Mirrors eval_retrieval.py's
# crimes + adds explicit out-of-scope probes so abstention is measurable.
GOLD = [
    # crimes WITH an expert rule
    ("theft_armed_night", "ما عقوبة السرقة بالإكراه ليلاً مع حمل سلاح؟", ["314", "315", "316"], True, "substantive"),
    ("forgery_official_official", "ما عقوبة تزوير محرر رسمي من موظف عام؟", ["211", "212", "213"], True, "substantive"),
    ("forgery_official_private", "ما عقوبة تزوير محرر رسمي من شخص عادي؟", ["214"], True, "substantive"),
    ("murder_premeditated", "ما عقوبة القتل العمد مع سبق الإصرار والترصد؟", ["230"], True, "substantive"),
    ("murder_negligent", "ما عقوبة القتل الخطأ؟", ["238"], True, "substantive"),
    ("assault_permanent", "ما عقوبة الضرب الذي ينتج عنه عاهة مستديمة؟", ["240"], True, "substantive"),
    ("assault_simple", "ما عقوبة الضرب البسيط الذي لا يسبب عجزاً؟", ["242"], True, "substantive"),
    # crimes WITHOUT an expert rule (rely on raw retrieval)
    ("fraud", "ما عقوبة جريمة النصب والاحتيال؟", ["336"], True, "substantive"),
    ("breach_trust", "ما عقوبة خيانة الأمانة؟", ["341", "340"], True, "substantive"),
    ("bribery", "ما عقوبة الرشوة للموظف العام؟", ["103", "104", "106", "108"], True, "substantive"),
    ("indecent_assault", "ما عقوبة هتك العرض بالقوة؟", ["267", "268", "269"], True, "substantive"),
    ("defamation", "ما عقوبة القذف في حق شخص؟", ["302", "303", "306"], True, "substantive"),
    # procedure
    ("pretrial_detention", "ما مدة الحبس الاحتياطي في الجنايات؟", ["143"], True, "procedural"),
    # out-of-scope — the model should abstain, not invent
    ("oos_vat", "ما هي قيمة ضريبة القيمة المضافة على السيارات المستوردة في مصر لعام 2024؟", [], False, "out_of_scope"),
    ("oos_visa", "ما هي إجراءات استخراج تأشيرة سياحية لزيارة فرنسا؟", [], False, "out_of_scope"),
]


def _pct(n, d):
    return round(100.0 * n / d, 1) if d else 0.0


# ── Retrieval mode (LLM-free) ───────────────────────────────────────────────────
def eval_retrieval(k=None):
    from app.services.retrieval import retrieval_service
    from app.services.reranker import reranker_service
    retrieval_service.load()
    reranker_service.load()

    rows = []
    in_scope = [g for g in GOLD if g[3]]
    print(f"\n=== retrieval mode | {len(in_scope)} in-scope gold queries | k={k or 'default'} ===\n")
    print(f"{'hit':>3} {'art_rec':>7} {'prim':>4} {'rr':>5}  {'expected':<20} {'found':<20} query")
    print("-" * 110)

    for gid, q, expected, scope, domain in GOLD:
        if not scope:
            continue
        contexts, sources, _ = retrieval_service.retrieve(q, k=k) if k else retrieval_service.retrieve(q)
        ctx_articles = set(extract_article_references(" ".join(contexts)))
        found = [a for a in expected if a in ctx_articles]
        hit = len(found) > 0
        art_recall = len(found) / len(expected) if expected else 0.0

        # primary-article hit: a retrieved source whose own article is an expected one
        src_articles = [str(s.get("article")) for s in sources if s.get("article")]
        primary_hit = any(a in expected for a in src_articles)

        # reciprocal rank of the first source carrying an expected article
        rr = 0.0
        for rank, s in enumerate(sources, start=1):
            s_arts = set([str(s.get("article"))] + [str(x) for x in (s.get("referenced_articles") or [])])
            if s_arts & set(expected):
                rr = 1.0 / rank
                break

        rows.append({"id": gid, "domain": domain, "expected": expected, "found": found,
                     "hit": hit, "article_recall": round(art_recall, 3),
                     "primary_hit": primary_hit, "rr": round(rr, 3),
                     "n_sources": len(sources)})
        print(f"{'✓' if hit else '✗':>3} {art_recall:>7.2f} {'Y' if primary_hit else '·':>4} "
              f"{rr:>5.2f}  {','.join(expected):<20} {','.join(found) or '—':<20} {q}")

    n = len(rows)
    summary = {
        "n_queries": n,
        "recall_at_k": _pct(sum(r["hit"] for r in rows), n),
        "article_recall": round(statistics.mean(r["article_recall"] for r in rows), 3) if n else 0,
        "primary_hit_at_k": _pct(sum(r["primary_hit"] for r in rows), n),
        "mrr": round(statistics.mean(r["rr"] for r in rows), 3) if n else 0,
    }
    print("-" * 110)
    print(f"recall@k={summary['recall_at_k']}%  article_recall={summary['article_recall']}  "
          f"primary_hit@k={summary['primary_hit_at_k']}%  MRR={summary['mrr']}")
    return {"mode": "retrieval", "summary": summary, "rows": rows}


# ── End-to-end mode (hits the API) ──────────────────────────────────────────────
def eval_e2e(k):
    import requests

    rows = []
    print(f"\n=== e2e mode | {len(GOLD)} gold queries | API={API} | k={k} ===\n")
    print(f"{'id':<22} {'scope':>5} {'cite':>4} {'halluc':>6} {'conf':>5} {'lat_s':>6}")
    print("-" * 60)

    for i, (gid, q, expected, scope, domain) in enumerate(GOLD):
        if i > 0 and SLEEP_BETWEEN_S > 0:
            time.sleep(SLEEP_BETWEEN_S)
        try:
            t0 = time.time()
            r = requests.post(f"{API}/qa", json={"question": q, "k": k, "prompt_style": "restrictive"}, timeout=1200)
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            print(f"{gid:<22} ERROR: {e}")
            rows.append({"id": gid, "error": str(e), "in_scope": scope})
            continue
        wall = (time.time() - t0) * 1000
        answer = d.get("answer", "") or ""
        conf = d.get("confidence_score")
        ctx_articles = set()
        for s in d.get("sources", []):
            if s.get("article"):
                ctx_articles.add(str(s["article"]))
            for a in (s.get("referenced_articles") or []):
                ctx_articles.add(str(a))
        cited = set(extract_article_references(answer))
        cite_correct = bool(cited & set(expected)) if expected else None
        # hallucination: an article cited in the answer that the context never supplied
        hallucinated = sorted(cited - ctx_articles)
        refused = any(m in answer for m in _REFUSAL_MARKERS)

        rows.append({
            "id": gid, "domain": domain, "in_scope": scope, "expected": expected,
            "cited": sorted(cited), "cite_correct": cite_correct,
            "hallucinated": hallucinated, "refused": refused,
            "confidence_score": conf, "latency_ms": d.get("latency_ms") or round(wall, 1),
            "warnings": d.get("warnings", []), "model": d.get("model"),
        })
        print(f"{gid:<22} {('in' if scope else 'OUT'):>5} "
              f"{('✓' if cite_correct else ('·' if cite_correct is None else '✗')):>4} "
              f"{('!' if hallucinated else '·'):>6} "
              f"{str(conf):>5} {(rows[-1]['latency_ms'] or 0)/1000:>6.1f}")

    ok = [r for r in rows if "error" not in r]
    in_scope = [r for r in ok if r["in_scope"]]
    out_scope = [r for r in ok if not r["in_scope"]]
    correct = [r for r in in_scope if r["cite_correct"]]
    wrong = [r for r in in_scope if not r["cite_correct"]]
    confs_ok = [r["confidence_score"] for r in correct if r["confidence_score"] is not None]
    confs_bad = [r["confidence_score"] for r in wrong if r["confidence_score"] is not None]
    lats = sorted(r["latency_ms"] for r in ok if r.get("latency_ms"))

    summary = {
        "n_queries": len(ok),
        "cite_recall": _pct(len(correct), len(in_scope)),
        "hallucination_rate": _pct(sum(1 for r in ok if r["hallucinated"]), len(ok)),
        "abstention_on_oos": _pct(sum(1 for r in out_scope if r["refused"]), len(out_scope)),
        "conf_correct": round(statistics.mean(confs_ok), 3) if confs_ok else None,
        "conf_wrong": round(statistics.mean(confs_bad), 3) if confs_bad else None,
        "latency_p50_ms": lats[len(lats) // 2] if lats else None,
        "latency_p90_ms": lats[int(len(lats) * 0.9)] if lats else None,
    }
    print("-" * 60)
    print(f"cite_recall={summary['cite_recall']}%  halluc={summary['hallucination_rate']}%  "
          f"abstention={summary['abstention_on_oos']}%  conf(ok/bad)={summary['conf_correct']}/{summary['conf_wrong']}  "
          f"lat_p50={summary['latency_p50_ms']}ms")
    return {"mode": "e2e", "summary": summary, "rows": rows}


def _save(tag, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"scorecard_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nscorecard → {path}")


def _compare(tag_a, tag_b):
    def load(t):
        with open(os.path.join(OUT_DIR, f"scorecard_{t}.json"), encoding="utf-8") as f:
            return json.load(f)
    a, b = load(tag_a), load(tag_b)
    print(f"\n=== {tag_a}  →  {tag_b} ===")
    for mode in ("retrieval", "e2e"):
        sa, sb = a.get(mode, {}).get("summary"), b.get(mode, {}).get("summary")
        if not (sa and sb):
            continue
        print(f"\n[{mode}]")
        for kpi in sa:
            va, vb = sa.get(kpi), sb.get(kpi)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                arrow = "↑" if vb > va else ("↓" if vb < va else "=")
                print(f"  {kpi:<22} {va:>8}  →  {vb:>8}  {arrow} {round(vb - va, 3):+}")
            else:
                print(f"  {kpi:<22} {va!s:>8}  →  {vb!s:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "e2e", "both"], default="retrieval")
    ap.add_argument("--tag", default=None, help="label for the scorecard file")
    ap.add_argument("--k", type=int, default=int(os.getenv("K", "10")))
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"),
                    help="diff two saved scorecards and exit")
    args = ap.parse_args()

    if args.compare:
        _compare(*args.compare)
        return

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"tag": tag, "ts": datetime.now().isoformat(), "k": args.k}
    if args.mode in ("retrieval", "both"):
        payload["retrieval"] = eval_retrieval(k=args.k)
    if args.mode in ("e2e", "both"):
        payload["e2e"] = eval_e2e(k=args.k)
    _save(tag, payload)


if __name__ == "__main__":
    main()
