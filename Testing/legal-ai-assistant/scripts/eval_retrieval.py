"""
LLM-free retrieval-quality evaluation.

For each labeled query, runs the production retrieval pipeline (FAISS+BM25+RRF+rerank
+ expert-rule prepend + context expansion) and checks whether the *authoritative*
article number(s) for that crime actually appear in the retrieved context.

This measures the real ceiling on answer quality independently of the LLM (which is
free-tier rate-limited): if the right article never reaches the context, no model can
cite it. "covered" = crime has an expert rule; "uncovered" = relies on raw retrieval.

Run:  python scripts/eval_retrieval.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.retrieval import retrieval_service
from app.services.reranker import reranker_service
from app.services.preprocessing import extract_article_references

# (query, expected_article_numbers, has_expert_rule)
CASES = [
    # --- crimes WITH an expert rule (should pass via prepended rule) ---
    ("ما عقوبة السرقة بالإكراه ليلاً مع حمل سلاح؟", ["314", "315", "316"], True),
    ("ما عقوبة تزوير محرر رسمي من موظف عام؟", ["211", "212", "213"], True),
    ("ما عقوبة تزوير محرر رسمي من شخص عادي؟", ["214"], True),
    ("ما عقوبة القتل العمد مع سبق الإصرار والترصد؟", ["230"], True),
    ("ما عقوبة القتل الخطأ؟", ["238"], True),
    ("ما عقوبة الضرب الذي ينتج عنه عاهة مستديمة؟", ["240"], True),
    ("ما عقوبة الضرب البسيط الذي لا يسبب عجزاً؟", ["242"], True),
    # --- crimes WITHOUT an expert rule (rely on raw retrieval) ---
    ("ما عقوبة جريمة النصب والاحتيال؟", ["336"], False),
    ("ما عقوبة خيانة الأمانة؟", ["341", "340"], False),
    ("ما عقوبة الرشوة للموظف العام؟", ["103", "104", "106", "108"], False),
    ("ما عقوبة هتك العرض بالقوة؟", ["267", "268", "269"], False),
    ("ما عقوبة القذف في حق شخص؟", ["302", "303", "306"], False),
    # --- procedure ---
    ("ما مدة الحبس الاحتياطي في الجنايات؟", ["143"], False),
]


def main():
    retrieval_service.load()
    reranker_service.load()

    hdr = f"{'cov':>3} {'hit':>3}  {'expected':<22} {'found_in_ctx':<22} query"
    print(hdr); print("-" * 100)
    covered_hits = uncovered_hits = covered_n = uncovered_n = 0
    for q, expected, has_rule in CASES:
        contexts, sources, _ = retrieval_service.retrieve(q)
        ctx_articles = set(extract_article_references(" ".join(contexts)))
        present = [a for a in expected if a in ctx_articles]
        hit = len(present) > 0
        if has_rule:
            covered_n += 1; covered_hits += hit
        else:
            uncovered_n += 1; uncovered_hits += hit
        print(f"{'Y' if has_rule else 'N':>3} {'✓' if hit else '✗':>3}  "
              f"{','.join(expected):<22} {','.join(present) or '—':<22} {q}")

    print("-" * 100)
    print(f"WITH expert rule:    {covered_hits}/{covered_n} retrieved the authoritative article")
    print(f"WITHOUT expert rule: {uncovered_hits}/{uncovered_n} retrieved the authoritative article")
    print(f"OVERALL recall:      {covered_hits+uncovered_hits}/{len(CASES)}")


if __name__ == "__main__":
    main()
