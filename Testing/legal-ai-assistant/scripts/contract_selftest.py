#!/usr/bin/env python3
"""
Contract self-test — hits every endpoint and asserts the RESPONSE SHAPE (field names +
types), not the values. Detects accidental wire-contract drift (a renamed/removed/retyped
field) before it breaks the .NET frontend.

Why it's standalone: the expected schemas below are declared HERE, independently of
app/models.py. If someone renames a field in the Pydantic models, the live response changes
but these expectations don't — so the test fails and the drift is caught. (Importing the
models would be circular and would never catch a rename.) Only `httpx` is required, so the
backend team can run it against any deployment too.

Usage:
    python scripts/contract_selftest.py                 # quick: free/no-LLM checks only
    python scripts/contract_selftest.py --full          # also exercises LLM endpoints (~6 LLM calls)
    python scripts/contract_selftest.py --base-url http://host:8000/api/v1
    python scripts/contract_selftest.py --full --base-url https://staging.example.com/api/v1

Exit code 0 = no contract failures (skips are OK), 1 = at least one shape mismatch.
A 503 from an LLM endpoint is reported as SKIP, not FAIL — it means the free-tier budget is
momentarily exhausted, which is a documented response, not contract drift.
"""
import sys
import json
import time
import uuid
import argparse

import httpx

# ── Shape spec language ───────────────────────────────────────────────────────
NUM = (int, float)               # JSON makes no int/float distinction — accept either


def NULLABLE(spec):
    return ("nullable", spec)


def LIST(item_spec):             # item_spec=None → only assert it's a list
    return ("list", item_spec)


def OBJ(schema):
    return ("obj", schema)


# ── Reusable sub-schemas ───────────────────────────────────────────────────────
CONF_FACTORS = {
    "rerank_signal": NUM,
    "source_count": NUM,
    "article_validation": str,
    "topic_match": bool,
}

SOURCE_INFO = {
    "filename": str,
    "source": str,
    "doc_type": str,
    "legal_category": str,
    "legal_topic": str,
    "article": NULLABLE(str),
    "referenced_articles": LIST(None),
    "page": NULLABLE(NUM),
    "retrieval_score": NUM,
    "rerank_score": NUM,
}

ATTACHED_DOC = {
    "doc_id": str,
    "filename": str,
    "content_type": str,
    "char_count": NUM,
    "attached_at": NUM,
}

# ── Envelope schemas (the wire contract) ───────────────────────────────────────
QA_ENV = {
    "answer": str,
    "confidence_score": NUM,
    "confidence_factors": OBJ(CONF_FACTORS),
    "sources": LIST(SOURCE_INFO),
    "warnings": LIST(str),
    "conflicts_detected": bool,
    "latency_ms": NUM,
    "retrieval_ms": NUM,
    "model": str,
}

CHAT_ENV = {
    "answer": str,
    "session_id": str,
    "confidence_score": NUM,
    "confidence_factors": OBJ(CONF_FACTORS),
    "sources": LIST(SOURCE_INFO),
    "warnings": LIST(str),
    "conflicts_detected": bool,
    "latency_ms": NUM,
    "turn_count": NUM,
    "was_compacted": bool,
    "model": str,
}

WEAKNESS_ENV = {
    "analysis": str,
    "confidence_score": NUM,
    "confidence_factors": OBJ(CONF_FACTORS),
    "sources": LIST(SOURCE_INFO),
    "warnings": LIST(str),
    "conflicts_detected": bool,
    "latency_ms": NUM,
    "model": str,
}

DEFENSE_ENV = dict(WEAKNESS_ENV, memorandum=str, self_check_revisions=NUM)
DEFENSE_ENV.pop("analysis")

FORENSIC_ENV = {
    "analysis": str,
    "conflicts_detected": bool,
    "confidence_score": NUM,
    "confidence_factors": OBJ(CONF_FACTORS),
    "sources": LIST(SOURCE_INFO),
    "warnings": LIST(str),
    "latency_ms": NUM,
    "model": str,
}

SUMMARIZE_ENV = {"summary": str, "input_length": NUM, "latency_ms": NUM, "model": str}

HEALTH_ENV = {
    "status": str, "vectors": NUM, "chunks": NUM, "model": str,
    "embedding_model": str, "reranker_loaded": bool, "reranker_model": str,
    "load_errors": LIST(str),
}

PARSE_ENV = {"text": str, "filename": str, "content_type": str, "char_count": NUM, "warnings": LIST(str)}

INGEST_ENV = {
    "status": str, "files_processed": NUM, "chunks_created": NUM,
    "vectors_added": NUM, "errors": LIST(str), "duration_s": NUM,
}

ATTACH_ENV = {
    "session_id": str, "attached": OBJ(ATTACHED_DOC),
    "attachments_total": NUM, "total_chars": NUM, "warnings": LIST(str),
}

ATTACH_LIST_ENV = {"session_id": str, "attachments": LIST(ATTACHED_DOC), "total_chars": NUM}

SESSION_INFO_ENV = {
    "session_id": str, "turn_count": NUM, "message_count": NUM,
    "has_summary": bool, "created_at": NUM, "last_active": NUM,
}

STREAM_FINAL_ENV = {
    "done": bool, "session_id": str, "confidence_score": NUM,
    "confidence_factors": OBJ(CONF_FACTORS), "sources": LIST(SOURCE_INFO),
    "warnings": LIST(str), "conflicts_detected": bool, "turn_count": NUM,
    "latency_ms": NUM, "model": str,
}


# ── Validator ──────────────────────────────────────────────────────────────────
def _check_value(label, value, spec, errors, path):
    if isinstance(spec, dict):
        # A bare dict spec is a nested object schema (e.g. a LIST item or sub-object).
        _validate(label, value, spec, errors, path)
        return
    if isinstance(spec, tuple) and spec and spec[0] in ("nullable", "list", "obj"):
        kind = spec[0]
        if kind == "nullable":
            if value is None:
                return
            _check_value(label, value, spec[1], errors, path)
        elif kind == "list":
            if not isinstance(value, list):
                errors.append(f"{label}: {path} expected list, got {type(value).__name__}")
                return
            item_spec = spec[1]
            if item_spec is not None:
                for i, item in enumerate(value):
                    _check_value(label, item, item_spec, errors, f"{path}[{i}]")
        elif kind == "obj":
            _validate(label, value, spec[1], errors, path)
    else:
        types = spec if isinstance(spec, tuple) else (spec,)
        if not isinstance(value, types):
            names = "/".join(t.__name__ for t in types)
            errors.append(f"{label}: {path} expected {names}, got {type(value).__name__} ({value!r:.40})")


def _validate(label, obj, schema, errors, path=""):
    if not isinstance(obj, dict):
        errors.append(f"{label}: {path or '<root>'} expected object, got {type(obj).__name__}")
        return
    for field, spec in schema.items():
        p = f"{path}.{field}" if path else field
        if field not in obj:
            errors.append(f"{label}: MISSING field '{p}'")
            continue
        _check_value(label, obj[field], spec, errors, p)
    extra = set(obj) - set(schema)
    if extra:
        # Extra fields are allowed (contract is add-only) — report as info, not failure.
        errors.append(f"INFO {label}: extra field(s) not in expected schema: {sorted(extra)}")


# ── Test harness ───────────────────────────────────────────────────────────────
class Runner:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=180.0)
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def _log(self, status, name, extra=""):
        mark = {"PASS": "\033[32mPASS\033[0m", "FAIL": "\033[31mFAIL\033[0m",
                "SKIP": "\033[33mSKIP\033[0m", "INFO": "\033[36mINFO\033[0m"}[status]
        print(f"  [{mark}] {name}" + (f" — {extra}" if extra else ""))

    def shape(self, name, body, schema):
        """Validate body against schema. Returns True on pass."""
        errors = []
        _validate(name, body, schema, errors)
        infos = [e for e in errors if e.startswith("INFO ")]
        hard = [e for e in errors if not e.startswith("INFO ")]
        for i in infos:
            self._log("INFO", name, i[5:])
        if hard:
            self.failed += 1
            self._log("FAIL", name, hard[0] + (f" (+{len(hard)-1} more)" if len(hard) > 1 else ""))
            for e in hard[1:6]:
                print(f"        {e}")
            return False
        self.passed += 1
        self._log("PASS", name)
        return True

    def expect_status(self, name, resp, want):
        if resp.status_code == want:
            self.passed += 1
            self._log("PASS", name, f"HTTP {want}")
            return True
        self.failed += 1
        self._log("FAIL", name, f"expected HTTP {want}, got {resp.status_code}: {resp.text[:120]}")
        return False

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            self._log("PASS", name)
        else:
            self.failed += 1
            self._log("FAIL", name, detail)
        return cond

    def llm_shape(self, name, resp, schema):
        """For LLM endpoints: 503 -> SKIP (budget), 200 -> validate, else FAIL."""
        if resp.status_code == 503:
            self.skipped += 1
            self._log("SKIP", name, "503 — LLM budget exhausted (documented, not drift)")
            return
        if resp.status_code != 200:
            self.failed += 1
            self._log("FAIL", name, f"unexpected HTTP {resp.status_code}: {resp.text[:120]}")
            return
        self.shape(name, resp.json(), schema)

    def post(self, path, **kw):
        return self.client.post(self.base + path, **kw)

    def get(self, path, **kw):
        return self.client.get(self.base + path, **kw)


def run(base_url, full):
    r = Runner(base_url)
    print(f"\nContract self-test → {r.base}   (mode: {'FULL' if full else 'quick'})\n")

    # ── Free / no-LLM checks ──────────────────────────────────────────────────
    print("Health & no-LLM endpoints:")
    try:
        resp = r.get("/health")
    except Exception as e:
        print(f"  \033[31mCANNOT REACH SERVER\033[0m at {r.base}: {e}")
        return 1
    if r.expect_status("GET /health status", resp, 200):
        r.shape("GET /health shape", resp.json(), HEALTH_ENV)

    # /parse (text) — no LLM
    resp = r.post("/parse", data={"text": "المادة ٢٣٠ تنص على عقوبة القتل العمد مع سبق الإصرار والترصد."})
    if r.expect_status("POST /parse (text) status", resp, 200):
        r.shape("POST /parse shape", resp.json(), PARSE_ENV)

    # /parse rejects .doc — negative contract
    resp = r.post("/parse", files={"file": ("x.doc", b"\xd0\xcf binary", "application/msword")})
    r.expect_status("POST /parse (.doc) -> 400", resp, 400)

    # /ingest/scan — non-destructive
    resp = r.post("/ingest/scan")
    if r.expect_status("POST /ingest/scan status", resp, 200):
        r.shape("POST /ingest/scan shape", resp.json(), INGEST_ENV)

    # Input gate envelope (no LLM): /qa with a fragment
    resp = r.post("/qa", json={"question": "## المستوى الأول — أساسيات فقط"})
    if r.expect_status("POST /qa (gated fragment) status", resp, 200):
        body = resp.json()
        r.shape("POST /qa (gated) envelope", body, QA_ENV)
        r.check("POST /qa (gated) model == input_gate", body.get("model") == "input_gate",
                f"got model={body.get('model')!r}")

    # /qa validation: too-short question -> 422
    resp = r.post("/qa", json={"question": "hi"})
    r.expect_status("POST /qa (too short) -> 422", resp, 422)

    # Chat envelope via gate (no LLM) + anonymous-session isolation
    print("\nChat / session contract:")
    s1 = r.post("/chat", json={"message": "## fragment one"})
    s2 = r.post("/chat", json={"message": "## fragment two"})
    if r.expect_status("POST /chat (gated) status", s1, 200):
        b1 = s1.json()
        r.shape("POST /chat (gated) envelope", b1, CHAT_ENV)
        sid1 = b1.get("session_id")
        sid2 = s2.json().get("session_id") if s2.status_code == 200 else None
        r.check("anonymous sessions get distinct, non-'default' ids",
                bool(sid1) and sid1 != "default" and sid1 != sid2,
                f"sid1={sid1!r} sid2={sid2!r}")
        # GET session info shape
        gi = r.get(f"/chat/{sid1}")
        if r.expect_status("GET /chat/{id} status", gi, 200):
            r.shape("GET /chat/{id} shape", gi.json(), SESSION_INFO_ENV)

    # Attach / list / detach (no LLM)
    print("\nDocument-upload contract (no-LLM paths):")
    sid = "selftest-" + uuid.uuid4().hex[:8]
    _attach_text = (
        "عقد إيجار بين الطرف الأول المؤجر أحمد محمود والطرف الثاني المستأجر سمير علي. "
        "اتفق الطرفان على إيجار الشقة الكائنة في القاهرة لمدة سنة بقيمة شهرية قدرها خمسة آلاف جنيه. "
        "نشب نزاع بين الطرفين حول اتهام المستأجر بإتلاف ممتلكات العقار عمداً."
    )
    resp = r.post("/chat/attach", data={"session_id": sid}, files={"file": ("note.txt", _attach_text.encode(), "text/plain")})
    attached_ok = False
    if r.expect_status("POST /chat/attach status", resp, 200):
        attached_ok = r.shape("POST /chat/attach shape", resp.json(), ATTACH_ENV)
    if attached_ok:
        gl = r.get(f"/chat/{sid}/attachments")
        if r.expect_status("GET /chat/{id}/attachments status", gl, 200):
            r.shape("GET /chat/{id}/attachments shape", gl.json(), ATTACH_LIST_ENV)
        r.client.delete(r.base + f"/chat/{sid}")  # cleanup

    # ── LLM-backed checks (only in --full) ─────────────────────────────────────
    if not full:
        print("\n(LLM endpoints skipped — run with --full to exercise /qa real, /chat, "
              "/weakness, /defense, /forensic, /qa/upload, /chat/stream. ~6 LLM calls.)")
    else:
        print("\nLLM-backed endpoints (503 => SKIP, not drift):")
        # /qa real — exercises sources[] / SourceInfo. (Cached questions cost 0 tokens.)
        resp = r.post("/qa", json={"question": "ما هي عقوبة خيانة الأمانة في القانون المصري؟"})
        r.llm_shape("POST /qa (real) shape", resp, QA_ENV)

        resp = r.post("/chat", json={"message": "ما هي عقوبة الرشوة في القانون المصري؟"})
        r.llm_shape("POST /chat (real) shape", resp, CHAT_ENV)

        resp = r.post("/summarize", json={"text": "المادة 230: كل من قتل نفساً عمداً مع سبق الإصرار والترصد يعاقب بالإعدام، ويشترط توافر نية إزهاق الروح وسبق الإصرار."})
        r.llm_shape("POST /summarize shape", resp, SUMMARIZE_ENV)

        resp = r.post("/weakness", json={"case_facts": "اتهم المتهم بالسرقة بناء على شاهد واحد ولا توجد كاميرات مراقبة."})
        r.llm_shape("POST /weakness shape", resp, WEAKNESS_ENV)

        resp = r.post("/defense", json={"case_facts": "اتهم المتهم بإتلاف ممتلكات عمداً، والدليل الوحيد صورة غير واضحة."})
        r.llm_shape("POST /defense shape", resp, DEFENSE_ENV)

        resp = r.post("/forensic", json={"case_facts": "عُثر على القتيل مصاباً بطلق ناري.", "evidence": "تقرير الطب الشرعي يشير إلى الوفاة بطلق ناري، لكن لم يُعثر على سلاح."})
        r.llm_shape("POST /forensic shape", resp, FORENSIC_ENV)

        resp = r.post("/qa/upload",
                      data={"question": "ما موضوع هذا المستند؟"},
                      files={"file": ("contract.txt", "عقد إيجار بين المؤجر والمستأجر بشأن شقة في القاهرة.".encode(), "text/plain")})
        r.llm_shape("POST /qa/upload shape", resp, QA_ENV)

        # /chat/stream (SSE)
        _stream_check(r)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nRESULT: {r.passed} passed, {r.failed} failed, {r.skipped} skipped\n{'='*60}")
    return 1 if r.failed else 0


def _stream_check(r):
    name = "POST /chat/stream (SSE)"
    try:
        with r.client.stream("POST", r.base + "/chat/stream",
                             json={"message": "ما تعريف القتل الخطأ في القانون المصري؟"}) as resp:
            if resp.status_code == 503:
                r.skipped += 1
                r._log("SKIP", name, "503 — LLM budget exhausted")
                return
            if resp.status_code != 200:
                r.failed += 1
                r._log("FAIL", name, f"HTTP {resp.status_code}")
                return
            token_events = 0
            final = None
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                evt = json.loads(line[len("data:"):].strip())
                if evt.get("done") is True:
                    final = evt
                elif "chunk" in evt:
                    token_events += 1
            if final is None:
                r.failed += 1
                r._log("FAIL", name, "no final done:true event")
                return
            # Error final events are a valid documented shape too.
            if "error" in final:
                r.skipped += 1
                r._log("SKIP", name, "stream returned a clean error event (providers down)")
                return
            r.check("stream produced token events", token_events > 0, "no {chunk,done:false} events")
            r.shape("stream final-event shape", final, STREAM_FINAL_ENV)
    except Exception as e:
        r.failed += 1
        r._log("FAIL", name, f"exception: {e}")


def main():
    ap = argparse.ArgumentParser(description="Conan API contract self-test (shape drift detector).")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1", help="API base URL")
    ap.add_argument("--full", action="store_true", help="also exercise LLM-backed endpoints (~6 LLM calls)")
    args = ap.parse_args()
    sys.exit(run(args.base_url, args.full))


if __name__ == "__main__":
    main()
