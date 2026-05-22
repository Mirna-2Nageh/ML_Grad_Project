# Gold case — Karim Adel (assault + theft)

A documented expert critique of a real system output, captured as a regression test
case. Used to verify the fixes to:
1. **Element matching** (article selection vs facts) — was citing م240 for an 8-day
   injury when م242 fits.
2. **Multi-charge handling** — was addressing only assault, silently dropping theft.
3. **Evidence + defendant-narrative analysis** — was ignoring cameras, witness
   ambiguity, prior financial dispute, and the defendant's mutual-fight claim.

## Inputs (reconstructed from the expert critique)

### `case_facts` (Arabic)
> حُرر محضر اتهام ضد المتهم/كريم عادل بتهمتين: (1) ضرب أحدث جروحاً للمجني عليه،
> (2) سرقة هاتف من نوع iPhone 15 Pro Max. وقعت الحادثة بعد مشاجرة بين الطرفين.
> يُثبت التقرير الطبي الشرعي إصابة المجني عليه بجرح قطعي وكدمة، ومدة العجز عن
> الأشغال الشخصية المقدَّرة طبياً ثمانية أيام.

### `evidence` (Arabic)
> كاميرات المراقبة بالمكان لم تُظهر بوضوح واقعة أخذ الهاتف من حوزة المجني عليه.
> شاهد الواقعة الوحيد لم يتمكن من تحديد طبيعة الشيء الذي كان يحمله المتهم وقت
> الواقعة. توجد خلافات مالية سابقة موثقة بين المتهم والمجني عليه. لم يُذكر ضبط
> الهاتف بحوزة المتهم بعد الواقعة.

### `defendant_statement` (Arabic)
> المتهم يقرر أن المجني عليه هو الذي بدأ بالاعتداء عليه، وأن الواقعة لم تكن سوى
> مشاجرة عرضية، وأنه لم يأخذ أي هاتف، وأن الخلاف المالي السابق هو دافع الاتهام
> الكيدي.

## Expert critique of the pre-fix output (2026-05)

The original memo:
- Cited **م240** (عاهة مستديمة) for an injury with **8 days incapacity** — wrong
  article (correct is **م242**); over-charging the client.
- **Ignored the theft charge entirely** — never discussed whether الاختلاس was proven,
  whether نية التملك existed, or how to attack each element.
- **Analyzed no evidence** — the unclear cameras, the ambiguous witness, and the prior
  financial dispute were never engaged with.
- **Did not address the defendant's own narrative** — mutual fight, possible
  self-defense / تعدٍ متبادل, mala fide accusation.

## Expected correct analysis (post-fix)

A correct memo must contain ALL of the following:

| Dimension | Expectation |
|---|---|
| Article selection | Cite **م242** for assault (8-day incapacity → جنحة ضرب), NOT م240 |
| Charge enumeration | Both **assault AND theft** explicitly listed in الإطار القانوني |
| Theft — material element | Discuss الاختلاس; note cameras unclear, witness ambiguous, no recovery → not established |
| Theft — mental element | Discuss نية التملك; mutual-fight + prior financial dispute → reasonable doubt |
| Evidence analysis | Each item (cameras, witness, prior dispute) explicitly analyzed |
| Defendant's narrative | Engaged: mutual fight, self-defense angle, mala fide motive |
| Requests | Separate request per charge: acquittal on theft + re-characterization of assault to م242 |

## Anti-patterns the system MUST avoid

- ❌ Citing م240 when injury < 20 days
- ❌ Citing only one charge when multiple are present
- ❌ Ignoring evidence items
- ❌ Ignoring the defendant's statement
- ❌ Adding generic disclaimers ("laws change over time", "consult a lawyer")
- ❌ Fabricating cassation rulings or articles not in retrieved context

## How to regenerate

```bash
curl -s -X POST http://localhost:8000/api/v1/defense \
  -H 'Content-Type: application/json' -d @- <<'JSON'
{
  "case_facts":  "<see case_facts above>",
  "evidence":    "<see evidence above>",
  "defendant_statement": "<see defendant_statement above>"
}
JSON
```

Then grade the response against the "Expected correct analysis" table above.
