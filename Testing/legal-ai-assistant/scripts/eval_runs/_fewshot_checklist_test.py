"""Ad-hoc validation: confirm the PROCEDURAL_DEFENSE_CHECKLIST + WORKED_EXAMPLE few-shot
makes all five defense-reasoning points fire on the reviewer's drug case.

Run against a live backend on :8000:
    ./LaW/bin/python scripts/eval_runs/_fewshot_checklist_test.py

Prints per-point coverage + the raw analysis. Exit code 0 = LLM answered (regardless of
coverage), 2 = LLM unavailable (503/quota), 1 = transport error.
"""
import json
import sys
import urllib.request

REQ = {
    "case_facts": (
        "ضبط المتهم (يعمل دليفري بشركة ذا فستر) بتهمة إحراز مواد مخدرة بقصد الاتجار. "
        "صدر إذن النيابة العامة يوم 15/8/2023 الساعة 11:00 مساءً لضبط وتفتيش المتهم ودراجته "
        "البخارية رقم (ط ج ى 21126) حال تردده على دائرة قسم ثان شرم الشيخ. أقر المتهم: الضبط "
        "حصل الصبح 15/8/2023 حوالي الساعة 12:00 صباحاً. تم الضبط خلف بورتو شرم الواقعة داخل "
        "دائرة قسم ثان شرم الشيخ."
    ),
    "evidence": (
        "محضر الضبط محرر بمعرفة النقيب/ كريم جمعة، ويذكر أن الدراجة المضبوطة رقمها (ط ج ى 2836). "
        "عُثر مع المتهم على 6100 جنيه. ملحوظة النيابة: تم تبصيم الأحراز بخاتم أمين شرطة/ نادر وهبة."
    ),
    "defendant_statement": (
        "أنا دليفري، والفلوس حوالي 3855 جنيه فلوسي غير عهدة الشغل. الضبط حصل قبل ما يطلع الإذن."
    ),
}


def main():
    body = json.dumps(REQ).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/api/v1/weakness",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            d = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if e.code == 503:
            print(f"LLM UNAVAILABLE (503) — quota/busy. Detail: {detail[:200]}")
            return 2
        print(f"HTTP {e.code}: {detail[:300]}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"TRANSPORT ERROR: {e}")
        return 1

    a = d.get("analysis", "")
    print(a)
    checks = {
        "A timeline-before-warrant": any(x in a for x in ["قبل صدور", "سابق", "بطلان القبض"]),
        "B plate mismatch": any(x in a for x in ["2836", "21126", "اختلاف", "لم يشمله"]),
        "C jurisdiction (no over-claim)": ("بورتو" not in a) or ("داخل دائرة" in a),
        "D chain-of-custody seal": any(x in a for x in ["تحريز", "الأحراز", "نادر", "بصم", "اختلاط"]),
        "E trafficking intent": any(x in a for x in ["قصد الاتجار", "عهدة", "دليفري", "إيراد"]),
    }
    print("\n================ CHECKLIST COVERAGE ================")
    for k, v in checks.items():
        print(("PASS" if v else "MISS"), k)
    print(f"\ncovered {sum(checks.values())}/5 | confidence={d.get('confidence_score')} "
          f"| model={d.get('model')} | warnings={d.get('warnings')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
