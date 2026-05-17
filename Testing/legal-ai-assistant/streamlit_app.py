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
        k_val = st.slider("عدد المراجع", 3, 15, 7, key="qa_k")

    if st.button("🔍 ابحث", key="qa_btn", type="primary", use_container_width=True):
        if question:
            with st.spinner("جاري البحث والتحليل..."):
                try:
                    r = requests.post(f"{API_BASE}/qa", json={
                        "question": question, "k": k_val
                    }, timeout=180)
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
    text = st.text_area(
        "الصق النص القانوني هنا",
        height=250,
        placeholder="المادة الأولى: ...",
        key="sum_input"
    )
    if st.button("📝 لخّص", key="sum_btn", type="primary", use_container_width=True):
        if text and len(text) >= 50:
            with st.spinner("جاري التلخيص..."):
                try:
                    r = requests.post(f"{API_BASE}/summarize", json={"text": text}, timeout=180)
                    data = r.json()
                    st.markdown(f'<div class="answer-box">{data.get("summary", "")}</div>', unsafe_allow_html=True)
                    st.metric("⏱️ زمن الاستجابة", f"{data.get('latency_ms', 0):.0f} ms")
                except Exception as e:
                    st.error(f"خطأ: {e}")
        else:
            st.warning("النص قصير جداً (الحد الأدنى ٥٠ حرف)")

# ═══════════ TAB 3: Weakness Detection ═══════════
with tab3:
    st.header("🔍 تحليل نقاط الضعف")
    case_facts = st.text_area(
        "وقائع القضية",
        height=250,
        placeholder="المتهم متهم بارتكاب جريمة... وتم القبض عليه بتاريخ...",
        key="weak_input"
    )
    if st.button("🔍 حلل", key="weak_btn", type="primary", use_container_width=True):
        if case_facts:
            with st.spinner("جاري تحليل نقاط الضعف..."):
                try:
                    r = requests.post(f"{API_BASE}/weakness", json={"case_facts": case_facts}, timeout=180)
                    data = r.json()
                    # The weakness response uses 'analysis' instead of 'answer'.
                    data_for_render = {**data, "answer": data.get("analysis", "")}
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
    def_facts = st.text_area("وقائع القضية", height=200, key="def_facts")
    def_weak = st.text_area("نقاط الضعف المحددة (اختياري)", height=100, key="def_weak")

    if st.button("📋 أنشئ المذكرة", key="def_btn", type="primary", use_container_width=True):
        if def_facts:
            with st.spinner("جاري إنشاء مذكرة الدفاع..."):
                try:
                    r = requests.post(f"{API_BASE}/defense", json={
                        "case_facts": def_facts,
                        "weaknesses": def_weak or ""
                    }, timeout=180)
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
                    r = requests.post(f"{API_BASE}/qa", json={"question": q, "k": 7}, timeout=180)
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
