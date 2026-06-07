"""
🏛️ Legal AI Assistant — Streamlit Evaluation & Testing UI
Run: streamlit run streamlit_app.py
"""
import time
import requests
import streamlit as st
import pandas as pd

# ─── Configuration ───
API_BASE = "http://localhost:8000/api/v1"
# Max time to wait for a response from an LLM-backed endpoint (qa/chat/summarize/
# weakness/defense). Set to 20 min so slow concurrent/heavy requests aren't cut off
# by the client before the backend finishes.
RESPONSE_TIMEOUT = 1200  # seconds (20 min)

st.set_page_config(
    page_title="⚖️ المساعد القانوني الذكي",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main { direction: rtl; }
    .stTextArea textarea { direction: rtl; font-family: 'Noto Kufi Arabic', 'Arial', sans-serif; font-size: 16px; }
    .stTextInput input { direction: rtl; font-family: 'Noto Kufi Arabic', 'Arial', sans-serif; }
    .answer-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #0f3460;
        direction: rtl;
        font-size: 16px;
        line-height: 1.8;
        margin: 10px 0;
    }
    .warning-box {
        background: #3d2914;
        color: #ffd966;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #f1c40f;
        direction: rtl;
        font-size: 14px;
        margin: 8px 0;
    }
    .source-chip {
        display: inline-block;
        background: #0f3460;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 12px;
    }
    .source-score {
        background: #1abc9c;
        color: #082c25;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        margin-left: 6px;
    }
    .metric-card {
        background: #16213e;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}


def _render_warnings(warnings):
    """Render a list of warning strings as RTL warning boxes."""
    for w in warnings or []:
        st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)


def _friendly_error(r):
    """Turn a non-200 response into a readable Arabic message.

    Handles FastAPI's two shapes: a plain `{"detail": "..."}` string (e.g. the 503
    'try again' message) and Pydantic's validation list `[{loc, msg, ctx}, ...]`
    (e.g. min_length too short) — the latter would otherwise render as empty output."""
    try:
        detail = r.json().get("detail", "")
    except Exception:
        return f"خطأ {r.status_code}: {(r.text or '')[:200]}"
    if isinstance(detail, list):  # Pydantic validation errors
        parts = []
        for e in detail:
            ctx = e.get("ctx") or {}
            if ctx.get("min_length"):
                parts.append(f"النص قصير جدًا — الحد الأدنى {ctx['min_length']} حرف.")
            elif ctx.get("max_length"):
                parts.append(f"النص طويل جدًا — الحد الأقصى {ctx['max_length']} حرف.")
            else:
                parts.append(e.get("msg", "قيمة غير صالحة"))
        return " ".join(parts) or "طلب غير صالح."
    return str(detail) or f"خطأ {r.status_code}"


def _render_metrics(data, with_retrieval=True):
    """Render a row of metrics: latency, retrieval, sources, confidence."""
    cols = st.columns(4 if with_retrieval else 3)
    cols[0].metric("⏱️ زمن الاستجابة", f"{data.get('latency_ms', 0):.0f} ms")
    idx = 1
    if with_retrieval:
        cols[idx].metric("🔍 زمن الاسترجاع", f"{data.get('retrieval_ms', 0):.0f} ms")
        idx += 1
    cols[idx].metric("📚 المراجع", f"{len(data.get('sources', []))}")
    idx += 1
    conf = data.get("confidence_score", 0) or 0
    conf_pct = int(round(conf * 100))
    conf_label = "✓ عالية" if conf >= 0.7 else ("متوسطة" if conf >= 0.4 else "منخفضة")
    cols[idx].metric(f"🎯 الثقة ({conf_label})", f"{conf_pct}%")


def _render_sources(data):
    sources = data.get("sources", [])
    if not sources:
        return
    with st.expander("📚 المصادر المستخدمة"):
        for s in sources:
            chips = []
            chips.append(s.get("filename", "N/A"))
            if s.get("doc_type"):
                chips.append(s["doc_type"])
            if s.get("legal_topic") and s["legal_topic"] != "general":
                chips.append(f"📂 {s['legal_topic']}")
            if s.get("article"):
                chips.append(f"المادة {s['article']}")
            score = s.get("rerank_score", 0) or 0
            chip_html = (
                f'<span class="source-chip">{" | ".join(chips)}'
                f'<span class="source-score">{score:.2f}</span></span>'
            )
            st.markdown(chip_html, unsafe_allow_html=True)


def _parse_via_api(uploaded_file):
    """Send an UploadedFile to /api/v1/parse and return its extracted text.

    Returns (text, warnings_list) or (None, [error_msg]) on failure. Used by
    the weakness / summarize / defense tabs to prefill their text areas from
    .txt / .pdf / .docx uploads without leaving Streamlit.
    """
    try:
        files_data = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        }
        r = requests.post(f"{API_BASE}/parse", files=files_data, timeout=120)
        if r.status_code >= 400:
            try:
                err = r.json().get("detail", "")
            except Exception:
                err = r.text
            return None, [f"خطأ {r.status_code}: {err}"]
        data = r.json()
        return data.get("text", ""), data.get("warnings", [])
    except Exception as e:
        return None, [f"فشل قراءة الملف: {e}"]


def _render_confidence_breakdown(data):
    """Show the per-component confidence factors in an expander."""
    factors = data.get("confidence_factors") or {}
    if not factors:
        return
    with st.expander("🔬 تفاصيل حساب الثقة"):
        cols = st.columns(4)
        cols[0].metric("Rerank", f"{factors.get('rerank_signal', 0):.2f}")
        cols[1].metric("عدد المصادر", factors.get("source_count", 0))
        cols[2].metric("تحقق المواد", factors.get("article_validation", "N/A"))
        cols[3].metric("مطابقة الموضوع", "✓" if factors.get("topic_match") else "✗")


# ─── Sidebar ───
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scales.png", width=80)
    st.title("⚖️ المساعد القانوني")
    st.markdown("**مساعد القانون الجنائي المصري**")
    st.divider()

    healthy, health_data = check_api_health()
    if healthy:
        st.success("✅ API متصل")
        st.caption(f"Vectors: {health_data.get('vectors', 0):,}")
        st.caption(f"Chunks: {health_data.get('chunks', 0):,}")
        st.caption(f"Model: {health_data.get('model', 'N/A')}")
        if health_data.get("reranker_loaded"):
            st.caption(f"🎯 Reranker: {health_data.get('reranker_model', 'on')}")
        else:
            st.caption("⚠️ Reranker: off")
    else:
        st.error("❌ API غير متصل")
        st.caption("تأكد من تشغيل: `uvicorn app.main:app`")

    st.divider()

    # Permanent-index ingestion (mode #3 of the upload feature). Files added
    # here become searchable for ALL future questions, unlike the per-question
    # attachment in Tab 1 which is one-shot.
    with st.expander("📥 إضافة مستندات للفهرس الدائم"):
        st.caption(
            "ارفع ملفات (.txt / .pdf / .docx) لإضافتها لقاعدة المعرفة الدائمة. "
            "سيتم تجزئتها وفهرستها، وستظهر في نتائج البحث لجميع الأسئلة اللاحقة."
        )
        ingest_files = st.file_uploader(
            "اختر ملفًا أو أكثر",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            key="ingest_files",
        )
        if st.button("🚀 إضافة للفهرس", key="ingest_btn", use_container_width=True):
            if not ingest_files:
                st.warning("اختر ملفًا واحدًا على الأقل")
            else:
                with st.spinner(f"جاري معالجة {len(ingest_files)} ملف..."):
                    try:
                        files_data = [
                            ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                            for f in ingest_files
                        ]
                        r = requests.post(
                            f"{API_BASE}/ingest", files=files_data, timeout=600,
                        )
                        if r.status_code >= 400:
                            st.error(f"خطأ {r.status_code}: {r.text[:200]}")
                        else:
                            data = r.json()
                            status = data.get("status", "?")
                            if status == "ok":
                                st.success(
                                    f"✅ تمت معالجة {data.get('files_processed', 0)} ملف، "
                                    f"إضافة {data.get('chunks_created', 0)} مقطع "
                                    f"({data.get('duration_s', 0):.1f}s)"
                                )
                            else:
                                st.warning(f"الحالة: {status}")
                                st.write(data)
                            for err in (data.get("errors") or [])[:5]:
                                st.warning(f"⚠️ {err}")
                    except Exception as e:
                        st.error(f"فشل الرفع: {e}")

    st.divider()
    st.caption("v2.0.0 | Phase 1: rerank + confidence + grounding")


# ─── Main Tabs ───
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "❓ سؤال وجواب",
    "📝 تلخيص",
    "🔍 نقاط الضعف",
    "📋 مذكرة دفاع",
    "📊 التقييم"
])

# ═══════════ TAB 1: Q&A ═══════════
with tab1:
    st.header("❓ اسأل سؤالاً قانونياً")
    question = st.text_input(
        "اكتب سؤالك هنا",
        placeholder="ما هي عقوبة السرقة بالإكراه في القانون المصري؟",
        key="qa_input"
    )
    col1, col2 = st.columns([3, 1])
    with col2:
        k_val = st.slider("عدد المراجع", 3, 30, 10, key="qa_k")

    # Optional per-question attachment: upload a file OR paste text. When
    # either is given, the request goes to /qa/upload instead of /qa and the
    # parsed text is prepended to the retrieved context for THIS question only.
    with st.expander("📎 إرفاق مستند (اختياري) — .txt / .pdf / .docx أو نص ملصق"):
        uploaded_file = st.file_uploader(
            "ارفع ملفًا للسؤال عنه",
            type=["txt", "pdf", "docx"],
            key="qa_upload_file",
            help="سيُضاف محتوى الملف إلى السياق لهذا السؤال فقط — لن يُضاف للفهرس الدائم.",
        )
        pasted_text = st.text_area(
            "أو الصق النص هنا",
            height=120,
            key="qa_upload_text",
            placeholder="مثال: المادة 240 من قانون العقوبات...",
        )

    if st.button("🔍 ابحث", key="qa_btn", type="primary", use_container_width=True):
        if question:
            has_upload = bool(uploaded_file or (pasted_text and pasted_text.strip()))
            spinner_text = (
                "جاري البحث والتحليل مع المستند المرفق..."
                if has_upload else "جاري البحث والتحليل..."
            )
            with st.spinner(spinner_text):
                try:
                    if has_upload:
                        # Multipart upload to /qa/upload — file wins if both given
                        form_data = {
                            "question": (None, question),
                            "k": (None, str(k_val)),
                            "prompt_style": (None, "restrictive"),
                        }
                        if uploaded_file is not None:
                            form_data["file"] = (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        elif pasted_text and pasted_text.strip():
                            form_data["text"] = (None, pasted_text)
                        r = requests.post(f"{API_BASE}/qa/upload", files=form_data, timeout=RESPONSE_TIMEOUT)
                    else:
                        r = requests.post(f"{API_BASE}/qa", json={
                            "question": question, "k": k_val
                        }, timeout=RESPONSE_TIMEOUT)

                    if r.status_code >= 400:
                        st.error(_friendly_error(r))
                    else:
                        data = r.json()
                        st.markdown(f'<div class="answer-box">{data.get("answer", "")}</div>', unsafe_allow_html=True)
                        _render_warnings(data.get("warnings"))
                        _render_metrics(data, with_retrieval=True)
                        _render_sources(data)
                        _render_confidence_breakdown(data)
                except Exception as e:
                    st.error(f"خطأ: {e}")

# ═══════════ TAB 2: Summarization ═══════════
with tab2:
    st.header("📝 تلخيص النصوص القانونية")
    st.caption("ارفع ملفًا (.txt / .pdf / .docx) للحصول على تلخيص آلي — أو الصق النص يدويًا.")

    sum_file = st.file_uploader(
        "📎 ارفع ملفًا",
        type=["txt", "pdf", "docx"],
        key="sum_upload_file",
    )
    # When the user uploads a new file, parse it via /parse and prefill the
    # text area. The parsed text lands in st.session_state so the user can
    # edit before submitting; we key by filename+size to detect "new" uploads
    # without re-parsing on every rerun.
    if sum_file is not None:
        file_sig = f"{sum_file.name}::{sum_file.size}"
        if st.session_state.get("sum_last_sig") != file_sig:
            with st.spinner(f"جاري قراءة {sum_file.name}..."):
                parsed, warns = _parse_via_api(sum_file)
            if parsed is not None:
                st.session_state["sum_input"] = parsed
                st.session_state["sum_last_sig"] = file_sig
                st.success(f"✅ تم استخراج {len(parsed):,} حرف من {sum_file.name}")
                for w in warns:
                    st.info(f"ℹ️ {w}")
            else:
                for w in warns:
                    st.error(f"⚠️ {w}")

    text = st.text_area(
        "النص القانوني (يمكنك التعديل قبل الإرسال)",
        height=250,
        placeholder="المادة الأولى: ... — أو ارفع ملفًا أعلاه ليُعبَّأ تلقائيًا",
        key="sum_input"
    )
    st.caption("ℹ️ الحد الأدنى للنص: 50 حرفًا.")
    if st.button("📝 لخّص", key="sum_btn", type="primary", use_container_width=True):
        if not text or len(text.strip()) < 50:
            st.warning(f"النص قصير جدًا — اكتب 50 حرفًا على الأقل (الحالي: {len((text or '').strip())}).")
        else:
            with st.spinner("جاري التلخيص..."):
                try:
                    r = requests.post(f"{API_BASE}/summarize", json={"text": text}, timeout=RESPONSE_TIMEOUT)
                    if r.status_code >= 400:
                        st.error(_friendly_error(r))
                    else:
                        data = r.json()
                        st.markdown(f'<div class="answer-box">{data.get("summary", "")}</div>', unsafe_allow_html=True)
                        st.metric("⏱️ زمن الاستجابة", f"{data.get('latency_ms', 0):.0f} ms")
                except Exception as e:
                    st.error(f"خطأ: {e}")

# ═══════════ TAB 3: Weakness Detection ═══════════
with tab3:
    st.header("🔍 تحليل نقاط الضعف")
    st.caption("ارفع ملف القضية (.txt / .pdf / .docx) ليُعبَّأ تلقائيًا في وقائع القضية — أو املأ الحقول يدويًا.")

    weak_file = st.file_uploader(
        "📎 ارفع ملف القضية",
        type=["txt", "pdf", "docx"],
        key="weak_upload_file",
    )
    if weak_file is not None:
        file_sig = f"{weak_file.name}::{weak_file.size}"
        if st.session_state.get("weak_last_sig") != file_sig:
            with st.spinner(f"جاري قراءة {weak_file.name}..."):
                parsed, warns = _parse_via_api(weak_file)
            if parsed is not None:
                st.session_state["weak_input"] = parsed
                st.session_state["weak_last_sig"] = file_sig
                st.success(f"✅ تم استخراج {len(parsed):,} حرف من {weak_file.name}")
                for w in warns:
                    st.info(f"ℹ️ {w}")
            else:
                for w in warns:
                    st.error(f"⚠️ {w}")

    case_facts = st.text_area(
        "وقائع القضية (يمكنك التعديل قبل التحليل)",
        height=250,
        placeholder="المتهم متهم بارتكاب جريمة... وتم القبض عليه بتاريخ... — أو ارفع ملفًا أعلاه",
        key="weak_input"
    )
    weak_evidence = st.text_area(
        "الأدلة (اختياري)",
        height=120,
        placeholder="التقرير الطبي، الشهود، كاميرات المراقبة، السوابق...",
        key="weak_evidence",
    )
    weak_defendant = st.text_area(
        "أقوال المتهم / دفوعه (اختياري)",
        height=100,
        placeholder="الدفاع الشرعي، انتفاء القصد، تفسير بديل للوقائع...",
        key="weak_defendant",
    )

    st.caption("ℹ️ وقائع القضية: 20 حرفًا على الأقل.")
    if st.button("🔍 حلل", key="weak_btn", type="primary", use_container_width=True):
        if not case_facts or len(case_facts.strip()) < 20:
            st.warning(f"وقائع القضية قصيرة جدًا — اكتب 20 حرفًا على الأقل لتحليل ذي معنى (الحالي: {len((case_facts or '').strip())}).")
        else:
            with st.spinner("جاري تحليل نقاط الضعف..."):
                try:
                    payload = {"case_facts": case_facts}
                    if weak_evidence and weak_evidence.strip():
                        payload["evidence"] = weak_evidence
                    if weak_defendant and weak_defendant.strip():
                        payload["defendant_statement"] = weak_defendant
                    r = requests.post(f"{API_BASE}/weakness", json=payload, timeout=RESPONSE_TIMEOUT)
                    if r.status_code >= 400:
                        st.error(_friendly_error(r))
                    else:
                        data = r.json()
                        st.markdown(f'<div class="answer-box">{data.get("analysis", "")}</div>', unsafe_allow_html=True)
                        _render_warnings(data.get("warnings"))
                        _render_metrics(data, with_retrieval=False)
                        _render_sources(data)
                        _render_confidence_breakdown(data)
                except Exception as e:
                    st.error(f"خطأ: {e}")

# ═══════════ TAB 4: Defense Memo ═══════════
with tab4:
    st.header("📋 إنشاء مذكرة الدفاع")
    st.caption("ارفع ملف القضية ليُعبَّأ تلقائيًا في وقائع القضية — أو املأ الحقول يدويًا.")

    def_file = st.file_uploader(
        "📎 ارفع ملف القضية",
        type=["txt", "pdf", "docx"],
        key="def_upload_file",
    )
    if def_file is not None:
        file_sig = f"{def_file.name}::{def_file.size}"
        if st.session_state.get("def_last_sig") != file_sig:
            with st.spinner(f"جاري قراءة {def_file.name}..."):
                parsed, warns = _parse_via_api(def_file)
            if parsed is not None:
                st.session_state["def_facts"] = parsed
                st.session_state["def_last_sig"] = file_sig
                st.success(f"✅ تم استخراج {len(parsed):,} حرف من {def_file.name}")
                for w in warns:
                    st.info(f"ℹ️ {w}")
            else:
                for w in warns:
                    st.error(f"⚠️ {w}")

    def_facts = st.text_area(
        "وقائع القضية (يمكنك التعديل قبل الإرسال)",
        height=200,
        placeholder="المتهم متهم بـ... وقعت الجريمة بتاريخ... — أو ارفع ملفًا أعلاه",
        key="def_facts",
    )
    def_weak = st.text_area("نقاط الضعف المحددة (اختياري)", height=100, key="def_weak")
    def_evidence = st.text_area(
        "الأدلة (اختياري)",
        height=120,
        placeholder="التقرير الطبي، الشهود، كاميرات المراقبة، السوابق...",
        key="def_evidence",
    )
    def_defendant = st.text_area(
        "أقوال المتهم / دفوعه (اختياري)",
        height=100,
        placeholder="الدفاع الشرعي، انتفاء القصد، تفسير بديل للوقائع...",
        key="def_defendant",
    )

    st.caption("ℹ️ وقائع القضية: 20 حرفًا على الأقل.")
    if st.button("📋 أنشئ المذكرة", key="def_btn", type="primary", use_container_width=True):
        if not def_facts or len(def_facts.strip()) < 20:
            st.warning(f"وقائع القضية قصيرة جدًا — اكتب 20 حرفًا على الأقل لإنشاء مذكرة (الحالي: {len((def_facts or '').strip())}).")
        else:
            with st.spinner("جاري إنشاء مذكرة الدفاع..."):
                try:
                    payload = {"case_facts": def_facts}
                    if def_weak and def_weak.strip():
                        payload["weaknesses"] = def_weak
                    if def_evidence and def_evidence.strip():
                        payload["evidence"] = def_evidence
                    if def_defendant and def_defendant.strip():
                        payload["defendant_statement"] = def_defendant
                    r = requests.post(f"{API_BASE}/defense", json=payload, timeout=RESPONSE_TIMEOUT)
                    if r.status_code >= 400:
                        st.error(_friendly_error(r))
                    else:
                        data = r.json()
                        st.markdown(f'<div class="answer-box">{data.get("memorandum", "")}</div>', unsafe_allow_html=True)
                        _render_warnings(data.get("warnings"))
                        _render_metrics(data, with_retrieval=False)
                        _render_sources(data)
                        _render_confidence_breakdown(data)
                except Exception as e:
                    st.error(f"خطأ: {e}")

# ═══════════ TAB 5: Evaluation ═══════════
with tab5:
    st.header("📊 تقييم الأداء")
    st.markdown("اختبر النظام مع مجموعة أسئلة وقيّم الإجابات (مع درجة الثقة وتحقق المواد)")

    default_questions = [
        "ما هي عقوبة السرقة بالإكراه في القانون المصري؟",
        "ما هي حالات الدفاع الشرعي في قانون العقوبات المصري؟",
        "ما هي شروط التوقيف الاحتياطي في القانون المصري؟",
        "ما هي أركان جريمة القتل العمد في القانون الجنائي المصري؟",
        "ما هي عقوبة التزوير في المحررات الرسمية؟",
    ]

    st.subheader("أسئلة الاختبار")
    custom_q = st.text_area(
        "أضف أسئلة (سؤال في كل سطر)",
        value="\n".join(default_questions),
        height=200,
        key="eval_qs"
    )

    if st.button("▶️ تشغيل التقييم", key="eval_btn", type="primary"):
        questions = [q.strip() for q in custom_q.strip().split("\n") if q.strip()]
        if not questions:
            st.warning("أضف سؤالاً واحداً على الأقل")
        else:
            results = []
            progress = st.progress(0)

            for i, q in enumerate(questions):
                try:
                    t0 = time.time()
                    r = requests.post(f"{API_BASE}/qa", json={"question": q, "k": 7}, timeout=RESPONSE_TIMEOUT)
                    total_t = (time.time() - t0) * 1000
                    data = r.json()
                    factors = data.get("confidence_factors") or {}
                    results.append({
                        "السؤال": q[:80],
                        "الإجابة": (data.get("answer", "") or "")[:150],
                        "الثقة %": int(round((data.get("confidence_score", 0) or 0) * 100)),
                        "تحقق المواد": factors.get("article_validation", "—"),
                        "تنبيهات": len(data.get("warnings", []) or []),
                        "المصادر": len(data.get("sources", []) or []),
                        "زمن API (ms)": round(data.get("latency_ms", 0)),
                        "زمن إجمالي (ms)": round(total_t),
                        "زمن استرجاع (ms)": round(data.get("retrieval_ms", 0)),
                    })
                except Exception as e:
                    results.append({
                        "السؤال": q[:80], "الإجابة": f"خطأ: {e}",
                        "الثقة %": 0, "تحقق المواد": "—", "تنبيهات": 0,
                        "المصادر": 0, "زمن API (ms)": 0, "زمن إجمالي (ms)": 0, "زمن استرجاع (ms)": 0,
                    })

                progress.progress((i + 1) / len(questions))
                time.sleep(1)

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("📈 ملخص الأداء")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("عدد الأسئلة", len(df))
            m2.metric("متوسط الثقة", f"{df['الثقة %'].mean():.0f}%")
            m3.metric("متوسط زمن API", f"{df['زمن API (ms)'].mean():.0f} ms")
            m4.metric("متوسط الاسترجاع", f"{df['زمن استرجاع (ms)'].mean():.0f} ms")
            m5.metric("نجاح", f"{sum(1 for r in results if 'خطأ' not in r['الإجابة'])}/{len(results)}")

            csv_data = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 تحميل النتائج CSV", csv_data, "evaluation_results.csv", "text/csv")
