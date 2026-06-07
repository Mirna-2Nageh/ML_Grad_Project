# Conan: A Hallucination-Resistant, Agentic Retrieval-Augmented Generation Framework for Arabic Legal Reasoning in Egyptian Criminal Law

**Authors:** Mirna Nageh, Wageh Mostafa, Seif Sherif, Mariam Mohamed, Mayar Mohammed

**Affiliation:** Faculty of Computers & Artificial Intelligence, Capital (Helwan) University, Egypt

---

## Abstract

Retrieval-Augmented Generation (RAG) has become the de-facto architecture for grounding large language models (LLMs) on domain-specific knowledge, yet its application to Arabic legal information retrieval continues to suffer from two well-documented failure modes: hallucinated statutory citations and the LLM's tendency to confabulate when retrieval surfaces topically-related but legally-inadequate context. In this work we present **Conan**, a production-grade Arabic legal assistant for the Egyptian Criminal Code (قانون العقوبات) and the Code of Criminal Procedure (قانون الإجراءات الجنائية). Starting from a strong hybrid-retrieval baseline (FAISS dense search + BM25 sparse search fused via Reciprocal Rank Fusion, with cross-encoder reranking), we introduce a multi-layered grounding pipeline that drives the rate of hallucinated article-number citations from a baseline of 28.6 % down to 0.0 % on a 28-question Arabic legal benchmark. Our contributions span seven interlocking grounding mechanisms: (i) an input-gating filter that rejects fragmentary user inputs before retrieval, (ii) a strict citation-grounding prompt regime separating substantive from procedural law, (iii) post-generation **evidence validation** that machine-checks every cited article against the retrieved context, (iv) a corrective LLM retry with an explicit block-list of forbidden article numbers, (v) an **article-lookup rescue** that indexes 1,494 distinct Egyptian-law articles from chunk metadata and uses them as a second-chance grounding source, (vi) **iterative retrieval** that auto-expands the k-window when validation fails — without requiring an additional LLM call, and (vii) deterministic answer post-processing that rewrites awkward template-leakage phrases and strips persona-breaking openers. We further introduce a document-upload subsystem supporting three retention semantics — per-question attachment, session-attached document, and permanent corpus ingestion — covering `.txt`, `.pdf`, and `.docx` formats. The complete system, evaluated on a 28-question benchmark covering substantive and procedural Egyptian criminal law, achieves a **100 % evidence-validation pass rate with zero hallucinations**, an 8-fold improvement over the baseline. As a further contribution we release the underlying corpus — 933 legacy-format source files manually OCR-cleaned into 1,057 UTF-8 documents (47,028 indexed chunks) — publicly on HuggingFace. Beyond single-question Q&A, the system escalates open-ended case work to a **six-stage multi-agent agentic pipeline** — document analysis, weakness detection, per-weakness legal-research and precedent retrieval, defence-strategy ranking, and grounded synthesis — in which a deterministic, code-computed timing verdict is injected as a hard fact to eliminate a recurring chronological-reasoning error, every argument is anchored to retrieved authority rather than parametric memory, and a draft→verify→revise self-check closes the argument-level hallucination gap; the pipeline degrades gracefully to a single-shot path on any agent failure. These case-analysis endpoints (weakness analysis, defence-memorandum drafting, forensic-consistency checking) are further governed by an expert-derived procedural-defence reasoning checklist paired with a grounding-safe few-shot exemplar. The system is deployed as a production FastAPI service with a five-provider, multi-key failover chain, big-context routing, a zero-token answer cache, an in-process rate gate, a wall-clock cascade budget, persistent multi-turn sessions with compaction, token streaming, automated watch-folder ingestion with hot-reload, and a documented .NET integration contract. Finally, we report a retrieval-quality optimisation phase targeting the retrieval ceiling that bounds answer correctness — article-aware chunking that keeps each statute article intact in one retrievable unit (live), a widened cross-encoder candidate pool (live), and a BGE-M3 embedding upgrade prepared behind configuration — together with a scored evaluation harness reporting recall@k, article-recall, primary-hit@k, MRR, hallucination rate, abstention, and confidence calibration over a labeled gold set, so every change is measurable. We also report a practical systems finding: on CPU-only hardware the highest-quality embedder can be infeasible to *build* without accelerator access, making the embedder a joint quality/operability choice. We release the system, the evaluation harness, and a multi-version trail documenting the incremental improvements.

**Keywords:** Arabic Natural Language Processing; Retrieval-Augmented Generation; Agentic AI; Multi-Agent Systems; Legal Information Retrieval; Hallucination Mitigation; Egyptian Criminal Law; Article-Lookup Rescue; Iterative Retrieval; Article-Aware Chunking; BGE-M3; Neuro-Symbolic Reasoning

---

## 1. Introduction

The Arabic language is the official tongue of more than twenty sovereign states and the first language of an estimated 422 million speakers, yet it remains comparatively underserved by modern natural language processing (NLP) systems [1]. This under-representation is especially acute in *Arabic legal* NLP, where the combination of Modern Standard Arabic (MSA), domain-specific terminology, dialectal variation, and the morphological richness of the language compound the standard challenges of grounded language modelling.

Retrieval-Augmented Generation (RAG) has emerged as the dominant architecture for closing the gap between a frozen LLM's parametric memory and the dynamic, citation-heavy demands of legal practice. By retrieving relevant passages from a curated corpus and conditioning the generator on those passages, RAG systems can in principle produce answers that are both fluent and traceable [2,6]. However, recent surveys of Arabic legal RAG implementations — including the Moroccan family-code study of Hrimech et al. [2] and the morphologically-aware retrieval study of Aboasal et al. [5] — identify a recurring failure pattern: the LLM remains liable to **hallucinate statutory citations**, even when grounded retrieval is available, particularly when the user's question maps to articles that were not surfaced in the retrieval top-k.

This paper presents **Conan** ("كونان"), an Arabic legal-AI assistant for Egyptian Criminal Law. Conan is built on a hybrid-retrieval foundation (FAISS dense embeddings + BM25 sparse search, fused via Reciprocal Rank Fusion, with cross-encoder reranking) and extends that baseline with a *grounding-defence pipeline* designed specifically to suppress the citation-hallucination failure mode documented in the Arabic-legal-RAG literature. We argue that no single technique is sufficient — instead, a **layered** approach combining input filtering, prompt engineering, post-hoc validation, article-level metadata lookup, and answer rewriting is required.

### 1.1 Contributions

The contributions of this work are as follows:

1. **A multi-stage grounding pipeline** for Arabic legal RAG that reduces the rate of hallucinated article-number citations from **28.6 % to 0.0 %** on a 28-question Egyptian criminal-law benchmark, an absolute reduction of 28.6 percentage points across six measurement points (v3 → v8).
2. **An input-gating mechanism** (Section 5.1) that detects and rejects fragmentary, markdown-formatted, or otherwise incomplete user inputs before they incur an LLM call — a class of input we observed accounted for approximately one in five fabricated-citation cases in baseline measurements.
3. **An article-lookup rescue subsystem** (Section 5.4) that indexes 1,494 distinct Egyptian-law article numbers from chunk metadata at startup and uses them as a deterministic second-chance grounding source when post-generation evidence validation flags a missing citation. The mechanism converts validated hallucinations into grounded answers *without* re-running the costly retrieval stage.
4. **Iterative retrieval with adaptive k-expansion** (Section 5.6) that automatically widens the retrieval window from k=7 to k=14, then k=21, when evidence validation fails — re-using cached chunks where possible and avoiding additional LLM calls. This mechanism captures legally-relevant articles that the initial reranker's top-k missed.
5. **A document-upload pipeline** (Section 6) supporting three distinct retention semantics — per-question, session-attached, and permanent-corpus — across the three most common Arabic document formats (`.txt`, `.pdf`, `.docx`) and pasted-text strings. The pipeline shares a single parser and Arabic-cleaning module to guarantee consistent behaviour across modes.
6. **A five-version evaluation trail** (Section 7) on a 28-question Arabic legal benchmark, with the per-version CSV outputs released alongside the system, allowing reproducible auditing of every architectural change.
7. **A procedural-defence reasoning checklist** (Section 5.8) that encodes expert-lawyer case-analysis heuristics — warrant-timeline nullity, identifier mismatch, chain-of-custody, and intent-from-profession — into the case-analysis prompts, paired with a *grounding-safe* few-shot exemplar that raises reasoning coverage without reintroducing hallucinated citations.
8. **A six-stage multi-agent agentic case-analysis pipeline** (Section 5.10) that decomposes open-ended legal work (`/weakness`, `/defense`) into specialised agents — document analysis, weakness detection, per-weakness legal-research and precedent retrieval, defence-strategy ranking, and grounded synthesis — orchestrated with bounded-concurrency `asyncio` fan-out, a deterministic *neuro-symbolic* timing verdict computed in code and injected as a hard fact, retrieval-per-weakness grounding so every argument is backed by retrieved authority, and an agentic draft→verify→revise self-check, with automatic fallback to a single-shot path on any agent failure.
8. **An open Egyptian-criminal-law corpus** (Section 3) manually OCR-processed from 933 legacy `.doc`/`.pdf` source files into 1,057 cleaned UTF-8 documents (47,028 indexed chunks) and released publicly on HuggingFace — to our knowledge the first open-licensed corpus focused on Egyptian criminal law at this scope.
9. **A production-grade operational architecture** (Sections 4, 8): a multi-provider, multi-key failover chain with big-context routing for large case files, persistent compacting multi-turn sessions, token streaming, automated watch-folder ingestion with in-place index hot-reload, an agentic self-check for defence memoranda (Section 5.9), and a frozen, documented wire contract for an external .NET frontend.
10. **A retrieval-quality optimisation phase with a scored evaluation harness** (Section 7.6): article-aware chunking that keeps each statute article intact in one retrievable unit and a widened cross-encoder candidate pool (both live), plus a BGE-M3 embedding upgrade prepared behind configuration with build/swap tooling — together targeting the retrieval ceiling that bounds answer correctness — accompanied by a labeled-gold-set harness reporting recall@k, article-recall, primary-hit@k, MRR, hallucination rate, abstention, and confidence calibration, so every retrieval/LLM/chunking change is measurable and reproducible. We additionally report the CPU-embedding-cost constraint that defers the embedding rollout to GPU-equipped hardware.

The remainder of the paper is organised as follows. Section 2 surveys related work on Arabic legal RAG, multilingual retrieval, and hallucination mitigation. Section 3 describes the construction and public release of the Egyptian-criminal-law corpus. Section 4 describes the base retrieval and operational architecture. Section 5 presents the grounding-defence mechanisms and the multi-agent agentic case-analysis pipeline in detail. Section 6 describes the document-upload and ingestion subsystem. Section 7 reports the evaluation methodology and results. Section 8 covers implementation, deployment, and frontend integration. Section 9 discusses strengths, limitations, challenges, lessons learned, and future work, and Section 10 concludes.

### 1.2 Development Methodology, Design Philosophy, and Lifecycle

The system was developed in an **incremental, evaluation-driven** manner: every architectural change was motivated by a concrete failure observed in the preceding evaluation round, implemented behind a configuration flag, and kept only if the next round's scorecard improved (or, in one instructive case, reverted when it regressed — Section 9). The version trail is itself a contribution: it isolates the marginal effect of each mechanism on the hallucination rate.

Three principles guided every decision. **(P1) Grounding over fluency.** In a legal tool, a wrong citation delivered fluently is more dangerous than an honest refusal; we therefore optimise for *traceability of every claim to retrieved text*, treating an ungrounded-but-plausible answer as a hard failure. **(P2) Defence in depth.** No single mechanism removes hallucination; heterogeneous failure modes (fragmentary input, prompt ambiguity, retrieval gaps, parametric-memory citations, template leakage) each demand a dedicated, independently-toggleable layer. **(P3) Graceful degradation.** Every advanced component — reranker, agentic pipeline, rescue, self-check, each LLM provider — has a defined fallback so that a single sub-system failure narrows capability rather than breaking the request.

Table 0 summarises the seven cumulative development phases from initial design to the current system; each phase builds on the indices, services, and configuration of its predecessors and was gated on the evaluation scorecard of the previous one.

**Table 0: Development lifecycle from initial design to the current system.**

| Phase | Focus | Key deliverables | Outcome |
|-------|-------|------------------|---------|
| 0 — Design | Requirements, corpus scoping, architecture choice | RAG over hybrid retrieval selected; grounding-fidelity adopted as primary metric; wire contract sketched with the .NET team | Target architecture fixed; metric defined |
| 1 — Corpus | Data acquisition + cleaning | 933 legacy files manually OCR-cleaned → 1,057 UTF-8 docs; HuggingFace release | 47,028-chunk index; open corpus |
| 2 — Baseline RAG | Hybrid retrieval + generation | FAISS + BM25 + RRF + cross-encoder rerank; confidence scoring; FastAPI service | Functional assistant; 28.6 % hallucination (v3) |
| 3 — Grounding defence | Hallucination suppression | Input gate, citation-grounding prompts, evidence validation, block-list retry | 28.6 → 7.1 % (v4) |
| 4 — Rescue + iteration | Recall-side recovery | Article-lookup rescue, iterative *k*-expansion, answer post-processing | 7.1 → 0.0 % (v6–v8) |
| 5 — Agentic case work | Open-ended reasoning | Six-agent weakness/defence pipeline, procedural-defence checklist, memo self-check, charge-fidelity guard | Reasoning ~70 % → expert-aligned |
| 6 — Retrieval ceiling | Recall optimisation + measurement | Article-aware chunking, widened rerank pool, BGE-M3 prepared, scored eval harness | 79 % article-anchored chunks; auditable tuning |
| 7 — Productionisation | Reliability + integration | Five-provider failover, answer cache, rate gate, cascade budget, streaming, watch-ingest, Docker, tunnel, .NET contract | Deployable, free-tier-survivable service |

The principal engineering decisions, the alternatives considered, and the rationale for each are recorded in Table 0b. Several invert the defaults one would adopt with unlimited compute or a paid LLM tier — a direct consequence of the CPU-only, free-tier deployment target.

**Table 0b: Major design decisions and their rationale.**

| Decision | Alternative considered | Rationale for the choice |
|----------|------------------------|--------------------------|
| Hybrid retrieval (dense + sparse + RRF) | Pure dense (FAISS only) | Lexical BM25 recovers exact article-number and rare-term matches dense embeddings miss; RRF needs no score calibration between the two. |
| Post-generation defence pipeline | Better retriever alone | Even perfect retrieval cannot stop the LLM citing from parametric memory; detecting and rescuing this is orthogonal to recall. |
| Deterministic timing verdict in code | Let the LLM reason over the timeline | The model repeatedly inverted arrest-vs-warrant chronology; a code-computed verdict injected as a hard fact removes the error class entirely (neuro-symbolic). |
| Multi-agent decomposition for case work | Single-shot long prompt | Per-weakness retrieval grounds each argument in its own authority; specialised agents emit checkable JSON, enabling strategy-level pruning of weak arguments. |
| OpenRouter (Qwen-2.5-72B) primary | Groq Llama-3.3-70B primary | A funded, usage-billed key removes the 6,000-TPM free-tier ceiling that capped multi-call (retry/rescue/agent) requests; Groq/Cerebras/xAI/Gemini remain free fallbacks. |
| Exact-match answer cache | Semantic/fuzzy cache | In law, serving a cached answer to a *different* question is unacceptable; exact normalised-match guarantees correctness while still saving tokens. |
| CPU embeddings, BGE-M3 deferred | Build BGE-M3 index now | On the CC-5.0 GPU (below PyTorch's floor) BGE-M3 builds at ~5.5 s/chunk → multiple days; the embedder becomes a joint quality/operability choice (Section 7.6). |

---

## 2. Related Work

### 2.1 Arabic RAG: foundations and challenges

El-Beltagy and Abdallah [1] present a foundational case study of Arabic RAG, evaluating multiple semantic-embedding models and LLMs in the retrieval and generation stages respectively, and explicitly investigating the impact of dialectal variation between document language and query language. Their work establishes the baseline architectural template that the present paper inherits — a semantic retriever feeding a generator LLM — while also flagging the central challenge of selecting an embedding model that captures the semantic nuances of Arabic without over-relying on the English-dominated pre-training data of most multilingual embedding models.

Alghamdi et al. [4] focus specifically on the retriever component, evaluating multiple Arabic information-retrieval techniques in the context of question answering. They demonstrate that retrieval quality is the dominant factor in downstream answer correctness — a finding our own iterative-retrieval mechanism (Section 5.6) extends by treating retrieval k as an adaptively-tunable parameter rather than a fixed hyperparameter.

### 2.2 Arabic legal RAG

Hrimech et al. [2] present the most direct prior work to ours: a RAG system for the *Moroccan* family code, built on the BGE-m3 multilingual embedding model with a custom dataset of 2,500 Arabic question–answer pairs. Their evaluation, using Mean Reciprocal Rank, Recall@k, F1, and a panel of semantic-fidelity metrics, finds that BGE-m3-driven RAG substantially outperforms standalone LLMs on legal-Q&A — but they explicitly call out the challenges of *legal-terminology adherence*, *content-validity of reproduced clauses*, and the *scarcity of annotated Arabic legal corpora*. The grounding-defence mechanisms introduced in the present work (Sections 4.1–4.5) target these specific concerns in the context of Egyptian criminal law.

Aboasal et al. [5] approach the same domain from a *morphological-and-semantic* angle, introducing a synthetic benchmark of 500 legal articles and 1,000 Q&A pairs and evaluating the impact of Farasa-based morphological segmentation on BM25, Ada v3, BGE-M3, GTE, and Mistral-embed retrievers. Their hybrid Farasa-BM25 + Ada v3 configuration reaches a Mean Average Precision of 0.8304 and nDCG@10 of 0.8626 — figures that informed our choice to keep BM25 as a complementary signal alongside dense FAISS retrieval (Section 4) rather than abandoning it in favour of pure dense retrieval.

### 2.3 Benchmarking Arabic legal reasoning

Abu Shairah et al. [3] introduce **ALARB**, an Arabic Legal Argument Reasoning Benchmark comprising over 13,000 commercial court cases from Saudi Arabia, with each case annotated with the facts, the court's step-by-step reasoning, the verdict, and the cited regulatory clauses. ALARB sets the methodological precedent for our own evaluation harness: a benchmark whose primary axis is not the linguistic *fluency* of the generated text but its *legal-grounding fidelity* — specifically, whether cited statutory articles can be traced to retrieved or otherwise-available source text. While our 28-question benchmark is far smaller in scale, it inherits the philosophy that *what an LLM cites matters more than how it phrases the citation*.

### 2.4 RAG fundamentals and tooling

A pragmatic introduction to RAG architecture is provided by the Mini-RAG documentation [6], which decomposes the pipeline into three steps — *retrieval* (semantic search over a knowledge base), *augmentation* (passage injection into the prompt), and *generation* (LLM completion conditioned on the augmented prompt) — and walks through chunking, indexing, and query workflows. While not a peer-reviewed publication, the document is representative of the implementation-oriented tutorial literature that informed many practical design decisions in our system (chunking strategy, vector-database choice, prompt scaffolding).

### 2.5 Positioning of the present work

Relative to [1] and [2], our work shifts the locus of grounding from the retrieval stage alone to a *post-generation defence pipeline*: even with state-of-the-art retrieval, an LLM may emit a memory-based citation that retrieval did not provide, and our evaluation shows that detecting and rescuing this case is at least as important as improving the retriever. Relative to [4] and [5], we share the focus on hybrid retrieval but extend the operational regime to include *iterative k-expansion* and *article-number-keyed rescue*. Relative to [3], we adopt the grounding-fidelity evaluation philosophy but apply it to an Egyptian-criminal-law corpus rather than a Saudi-commercial-law one.

---

## 3. Dataset Construction

A central practical obstacle to Arabic legal RAG is the *scarcity of clean, machine-readable Egyptian legal text*. Authoritative Egyptian criminal-law material — particularly cassation-court rulings and the historical foundations of the Penal Code — exists almost exclusively as scanned PDFs and as legacy Microsoft Word 97/2003 binaries (`.doc`), both essentially unreadable to standard NLP pipelines without manual intervention. The corpus we release was constructed from scratch to address exactly this gap.

**Source collection.** We assembled an initial set of **933 source files** (818 `.doc`, 112 `.pdf`, 3 `.docx`) covering ten distinct collections: the Penal Code with its explanatory memorandum, the Code of Criminal Procedure with commentaries, the four-volume *Rules Established by the Court of Cassation*, the four-volume *Most Recent Principles Issued by the Criminal Chambers*, the *Encyclopedia of Cassation Criminal Rulings* (793 `.doc` files — the bulk of the manual-conversion effort), the *Quarter-Century Collection (1931–1955)*, the *Forensic Medicine Series*, separate *Felonies* (`جنايات`) and *Misdemeanours* (`جنح`) collections, and criminal-law textbooks. The dominance of `.doc` legacy binaries (87.3 %) is characteristic of Egyptian government and university legal archives dating to the 1990s–2000s.

**Manual OCR and cleaning pipeline.** Each legacy `.doc` was opened, converted, and manually cleaned; scanned PDFs were processed with `pdftotext -layout -enc UTF-8` (poppler-utils) and a second OCR pass where no text layer existed. Each output was reviewed for: paragraph-break preservation across page boundaries; removal of running headers/footers/page numbers; normalisation of the four alef forms (`أ`, `إ`, `آ`, `ا`) to a canonical alef; harmonisation of Arabic-Indic (`٠`–`٩`) and Eastern Arabic-Indic digits to Western 0–9; removal of boilerplate (`بسم الله الرحمن الرحيم`, `باسم الشعب`); and preservation of article-number markers (`المادة`, `المواد`, `مادة`) so the downstream `extract_article_references` extractor (Section 5.3) can find them.

**Result and public release.** The cleaned corpus (`final_total_dataset/`) contains **1,057 documents** (818 cleaned `.txt`, 238 PDFs retained where layout was load-bearing, 1 `.docx`), ~200 MB of UTF-8 Arabic, organised into nine top-level subdirectories preserving the original taxonomy. It is released publicly on HuggingFace as *Egyptian_Criminal_Legal_Assistant_RAG_V2* — to our knowledge the first open-licensed Arabic-legal corpus focused on Egyptian criminal law at this scope.

**Indexing.** The corpus is processed by doc-type-aware chunking (`CHUNKING_CONFIGS`: 2,000 chars / 200 overlap for penal-code and cassation material, 3,000/400 for case files, 2,048/300 for procedural texts and rule collections), embedded with a multilingual sentence-transformer, and indexed in FAISS alongside a parallel BM25 index. The resulting index contains **47,028 chunks**, of which 14,882 (31.6 %) carry `referenced_articles` metadata covering 1,494 distinct article numbers — the property exploited by the article-lookup rescue (Section 5.5).

---

## 4. System Architecture

Conan is implemented as a FastAPI service backed by a retrieval index of 47,028 chunks built from the Egyptian Penal Code, the Code of Criminal Procedure, a curated subset of cassation rulings, the Cassation Encyclopedia, and supplementary criminal-law references. The base architecture follows the pattern established by [1], [2], and [4]:

**Retrieval.** A user query is encoded with a multilingual sentence-transformer (`paraphrase-multilingual-MiniLM-L12-v2` in the live deployment) and used to query a FAISS index of pre-computed chunk embeddings (dense branch, top-60 candidates). In parallel, the same query is BM25-tokenised (with Arabic-Indic digit normalisation and alef-form unification) and scored against the corpus via a `rank_bm25` BM25Okapi index (sparse branch, top-60). The two ranked lists are fused using **Reciprocal Rank Fusion** with the canonical RRF constant `k = 60`. An optional **domain boost** (`USE_DOMAIN_BOOST`) adds a small weight (`DOMAIN_BOOST_WEIGHT = 0.05`) to the fused score of chunks whose `doc_type` matches the query's inferred domain (procedural vs. substantive) before reranking. The fused top-60 candidates (`RETRIEVAL_K_RERANK`) are passed to a **cross-encoder reranker** (BGE-Reranker-v2-M3, sigmoid-bounded), which produces the final top-`k` set (default `k = 10`). Because the cross-encoder is CPU-bound, concurrent reranks are capped by a semaphore (`RERANK_MAX_CONCURRENCY = 2`) so excess requests queue briefly rather than thrashing the CPU and breaching the client timeout.

**Context expansion.** Each surviving chunk is grown by one *same-source* neighbour on each side, providing roughly 2× the textual context per chunk while preventing cross-document contamination.

**Generation.** The final context, expert-rule entries (matched by keyword from a curated `expert_rules.json`), and the user's question are formatted into a prompt template enforcing strict textual adherence ("Use the provided texts ONLY") and dispatched to the LLM provider chain. The current deployment uses OpenRouter (`qwen/qwen-2.5-72b-instruct`) as the primary provider for every feature, with Groq (`llama-3.3-70b-versatile`), Cerebras, xAI Grok, and Google Gemini as free fallback tiers; multi-key rotation within each provider handles per-minute and per-day rate limits. (The version-trail evaluation of Section 7 was run with Groq Llama-3.3-70B as primary; the move to a funded OpenRouter key removed the free-tier token-per-minute ceiling that capped the multi-call retry/rescue/agent requests.) Per-feature model routing (`MODEL_BY_FEATURE`) sends reasoning-heavy work to the strongest model and high-volume Q&A/chat to a faster tier.

**Confidence scoring.** A weighted heuristic over four signals — mean rerank score (weight 0.30), source count (0.20, capped at 5), article-validation pass/fail (0.30, Section 5.3), and topic-match (0.20, encyclopedia folder taxonomy) — produces a confidence value in [0, 1] alongside a structured breakdown. Because the BGE reranker's sigmoid scores sit low (~0.1) on Arabic legal text, the rerank signal is gamma-calibrated (`RERANK_CALIBRATION_GAMMA = 0.5`) to lift the squashed mid-range without saturating strong matches. Answers below a configurable threshold (default 0.5) carry a low-confidence clarification warning.

**Conversational layer.** Beyond stateless Q&A, the system exposes a multi-turn chat endpoint backed by a thread-safe `SessionManager`. Each `session_id` accumulates a turn history; once it exceeds `SESSION_MAX_TURNS` (default 6) the older turns are LLM-summarised into a single `[ملخص المحادثة السابقة]` block while the most recent `SESSION_KEEP_RECENT` (default 3) turns are kept verbatim, bounding prompt growth without losing legal context. Sessions persist to disk as atomically-written JSON (write-to-`.tmp` + `os.replace`) so they survive restarts, and a background pruner evicts sessions idle longer than `SESSION_TTL_HOURS` (default 168 h). Each session can additionally carry a list of *attached documents* (Section 6). A streaming variant (`POST /api/v1/chat/stream`) returns Server-Sent Events as tokens are generated, with the terminal event carrying the full confidence, sources, and grounding warnings once post-generation validation completes.

**Multi-provider resilience.** Generation is served by an ordered, five-provider chain — OpenRouter (`qwen-2.5-72b`, primary) → Groq (`llama-3.3-70b-versatile`) → Cerebras → xAI Grok → Google Gemini — in which each provider rotates across its own list of API keys on HTTP 429/quota errors before the chain falls through to the next provider. (Google Gemini supports several per-project keys, each with its own free daily quota, so listing several multiplies the free budget.) This multi-key, multi-provider rotation is what lets the service survive free-tier per-minute and per-day token caps. A complementary **big-context routing** rule guards against oversized prompts: each provider declares a maximum request size (`PROVIDER_MAX_PROMPT_CHARS`; e.g. 26,000 chars for Groq, 16,000 for Cerebras, 80,000 for OpenRouter, no cap for Gemini), and a request whose assembled prompt exceeds a provider's budget transparently skips that provider and is routed to a larger-context tier — allowing the case-analysis endpoints to accept full case files (up to 50,000 characters) without failing on a provider's context window.

**Free-tier and latency safeguards.** Three coordinated mechanisms keep the service usable under free-tier budgets and a fixed client timeout. *(i) A zero-token answer cache* (`AnswerCache`, LRU-bounded, optional atomic disk persistence) serves the stateless `/qa` endpoint: an exact normalised-question match (whitespace + Arabic-Indic digits folded only — never fuzzy or semantic, since serving a cached answer to a *different* legal question is unacceptable) returns instantly and at zero LLM cost, which also lets a demo be "pre-warmed." *(ii) An in-process rate gate* (`LLM_MIN_INTERVAL_S = 2 s`) enforces a minimum spacing between physical provider requests, smoothing the multi-call-per-request pattern (draft → retry → rescue) and concurrent users so a per-minute budget is not burst-exhausted. *(iii) A wall-clock cascade budget* (`QA_CASCADE_BUDGET_S = 90 s`) bounds worst-case latency: the initial retrieve-and-answer always runs, but once cumulative request time crosses the budget no *new* corrective stage (iterative retrieval, retry, rescue) is launched and the best-grounded answer so far is returned — keeping the worst case under the ~180 s client timeout regardless of how slow a fallback LLM is.

**Configurable embeddings and reranker.** The embedding backend is pluggable: the default is a local sentence-transformer on CPU (`BAAI/bge-m3`, 1024-dim, in the reference configuration; the deployed index uses `paraphrase-multilingual-MiniLM-L12-v2`, 384-dim), and setting `USE_REMOTE_EMBEDDINGS` switches to a remote provider (Google Gemini embeddings, rate-limited to 15 RPM via batching, or OpenRouter). Because vector dimensionality and semantics differ, switching the embedding provider requires rebuilding the FAISS index. The cross-encoder reranker (`BGE-Reranker-v2-M3`) is likewise toggleable via `USE_RERANKER`; if disabled or unable to load, retrieval degrades gracefully to RRF-only ordering with no break in behaviour.

### 4.1 Technology stack

The stack is deliberately built on open, self-hostable components for the retrieval core, so the only externally-billed dependency is the generation LLM — and even that is abstracted behind a five-provider chain that runs on free tiers.

**Table 2: Technology stack by architectural layer.**

| Layer | Technology | Role |
|-------|-----------|------|
| Service framework | FastAPI + Uvicorn (ASGI) | Async HTTP API, dependency-injected singletons, SSE streaming |
| Dense retrieval | FAISS (inner-product index) | Approximate nearest-neighbour over chunk embeddings |
| Embeddings | sentence-transformers: MiniLM-L12-v2 (384-d, live); BGE-M3 (1024-d, prepared) | Encode chunks/queries into the vector space |
| Sparse retrieval | `rank_bm25` (BM25Okapi) | Lexical recall of exact terms / article numbers |
| Fusion | Reciprocal Rank Fusion (k = 60) | Calibration-free merge of dense + sparse ranks |
| Reranking | BGE-Reranker-v2-M3 (CrossEncoder, sigmoid) | Re-score top-60 fused candidates → top-10 |
| Generation | OpenRouter Qwen-2.5-72B (primary) + Groq / Cerebras / xAI / Gemini | Grounded Arabic answer synthesis with failover |
| Agent orchestration | `asyncio` (bounded-concurrency `gather`) | Fan-out of per-weakness research; semaphore-capped |
| Document parsing | poppler `pdftotext`, PyPDF2, `docx2txt` | PDF/DOCX/TXT extraction with Arabic-encoding fallback |
| Persistence | Atomic JSON files (sessions, cache, ingest registry) | Restart-survivable state without a database |
| Reference UI | Streamlit | Reference client exercising every endpoint |
| Deployment | Docker + Compose; Cloudflare / ngrok tunnel; systemd | Reproducible packaging and laptop-hosted remote exposure |

### 4.2 Service modules and API surface

The application is a thin router layer over process-wide singleton services loaded once at start-up (Table 3), exposing the REST surface in Table 4. Centralising state in singletons (`retrieval_service`, `reranker_service`, `article_lookup_service`, `session_manager`, `answer_cache`) means the expensive models are paid for once, and the hot-reload path can swap index state in place without re-initialising them.

**Table 3: Core service modules (`app/services/`) and responsibilities.**

| Module | Responsibility |
|--------|----------------|
| `retrieval.py` | FAISS + BM25 + RRF + domain-boost + reranker pipeline; `reload_indices()` hot-reload |
| `reranker.py` | BGE cross-encoder singleton (sigmoid-bounded); graceful no-op on load failure |
| `chunking.py` | Article-aware, doc-type-aware chunker — single source of truth for both index writers |
| `preprocessing.py` | Arabic normalisation, BM25 tokenisation, doc-type/topic classifiers, article-reference extractor |
| `confidence.py` | `validate_evidence`, `topic_match`, `compute_confidence` (pure functions) |
| `article_lookup.py` | {article → chunks} index for the rescue path; three-tier quality ranking |
| `llm.py` | Sync/async provider-chain client + token-streaming generator; rate gate; big-context routing |
| `agents.py` | Six-agent weakness/defence orchestrators + deterministic timing verdict |
| `memo_agent.py` | Draft → verify → revise memo self-check |
| `postprocessing.py` | Casual-opener stripping + embedded-refusal rewrite |
| `session.py` | Sliding-window chat compaction, atomic persistence, TTL pruner, attachments |
| `answer_cache.py` | LRU, exact-match, zero-token `/qa` cache with optional disk persistence |
| `document_loader.py` / `upload_helper.py` | PDF/DOCX/TXT loading + shared upload parsing/cleaning |

**Table 4: REST API surface (prefix `/api/v1`).**

| Verb | Path | Function |
|------|------|----------|
| POST | `/qa` | Stateless grounded Q&A (full defence pipeline + cache) |
| POST | `/qa/upload` | Q&A with a per-question document attachment |
| POST | `/chat` | Multi-turn chat (Gemini-first fallback chain, sessions) |
| POST | `/chat/stream` | Server-Sent-Events token streaming |
| POST | `/chat/attach` | Bind a document to a session (persisted) |
| GET/DEL | `/chat/{sid}/attachments[/{id}]` | Attachment lifecycle management |
| POST | `/weakness` | Agentic weakness analysis of a case file |
| POST | `/defense` | Agentic defence-memorandum drafting + self-check |
| POST | `/forensic` | Forensic-consistency / charge-fidelity analysis |
| POST | `/summarize` | Legal-document summarisation |
| POST | `/parse` | Preview-parse an uploaded document to clean text |
| POST | `/ingest`, `/ingest/scan` | Permanent corpus ingestion + watch-folder scan with hot-reload |
| GET | `/health` | Liveness + loaded-index status |

---

## 5. Grounding-Defence Pipeline

The core contribution of this paper is the layered defence pipeline that sits on top of the architecture in Section 4. Each layer targets a specific failure mode observed in baseline measurements (Section 7.1).

### 5.1 Input gating

**Failure mode.** In baseline evaluation, fragmentary user inputs (markdown headings such as `## المستوى الأول`, single-word bullets such as `* الجناية`, or numbered list-items ending in a colon such as `1. ما المقصود بمبدأ:`) frequently triggered the LLM to *fabricate* a plausible-sounding question and answer it. These cases accounted for 5 out of 28 (17.9 %) hallucination instances in baseline measurements.

**Mechanism.** A pure function `is_meaningful_query` evaluates the user's input against a sequence of regex-based predicates: minimum Arabic-character count, leading markdown-header detection, leading list-bullet detection, leading numbered-fragment detection, and a trailing-colon check that rejects sentences that are clearly anticipating a continuation (e.g., `ما المقصود بمبدأ:`). Inputs that fail any predicate are short-circuited with a 9-millisecond response containing a polite Arabic prompt to reformulate the question (`يبدو أن السؤال غير مكتمل أو غير واضح. يرجى صياغة سؤال قانوني كامل …`). The input gate is **bypassed** when an attached document is present (Section 6), since "لخّص" + a file is a meaningful request even though "لخّص" alone is not.

**Result.** In the v8 evaluation, the input gate caught and rejected 8 of 28 inputs (28.6 %) without incurring an LLM call. The latency reduction on these inputs is roughly 3,000× compared to the full pipeline.

### 5.2 Citation-grounding prompts and law-naming disambiguation

**Failure mode.** Two distinct prompt-level failure modes were observed: (a) the LLM cites article numbers from training memory that *appear* plausible but are not in the retrieved context; (b) the LLM confuses substantive (Penal Code) with procedural (Code of Criminal Procedure) law, citing the wrong code for a given query. For example, baseline answers to questions about pre-trial detention (`التوقيف الاحتياطي`) frequently cited Article 300 *of the Penal Code* — when Article 300 of the Penal Code does not address that topic; the relevant article is in the *Code of Criminal Procedure*.

**Mechanism.** Both `qa_restrictive` and `chat` prompts are extended with explicit citation-grounding rules:
- *Citation grounding (absolute):* "Cite an article number ONLY if those exact digits appear in the Context below."
- *Law-naming disambiguation:* procedural matters (التوقيف الاحتياطي، التحقيق، التفتيش، الطعن، النقض) must be cited from `قانون الإجراءات الجنائية`; substantive matters (تعريف الجرائم، أركانها، عقوباتها) from `قانون العقوبات`.
- *No casual openings:* explicit prohibition on `حسناً`, `بالتأكيد`, `تمام`, and similar persona-breaking conversational fillers.

The mechanism follows the spirit of the citation-validity recommendations of Hrimech et al. [2], who emphasise the importance of "content validity of legal clauses reproduced from retrieval systems."

### 5.3 Evidence validation

**Mechanism.** Following the legal-grounding-fidelity philosophy of [3], every LLM output is post-processed by an evidence validator. A regex-based extractor (`extract_article_references`) — operating on Arabic-Indic-digit-normalised text — identifies every article number cited in the answer. The set of cited articles is compared against the set of article numbers extracted from the *retrieved context* (concatenated). Articles cited but not present in the context are flagged as **missing**, and the answer is marked as failing validation. The same extractor is used on both sides (answer and context) to avoid spurious mismatches caused by plural-form variations (`المواد 211، 212، 213` vs `المادة 211`).

### 5.4 Corrective retry with forbidden-article block-list

**Mechanism.** When evidence validation flags missing articles, the system re-prompts the LLM with a *correction* prompt (`qa_retry_ungrounded`) that includes the original draft answer and an *explicit list* of article numbers the LLM must NOT cite (because they are known not to be in the context). The retry is run at most `RETRY_MAX_ATTEMPTS` times (default 1 — a careful tuning informed by the Groq free-tier 6,000 tokens-per-minute budget, beyond which a second retry cascades into 429 rate-limit errors). The retry's output is accepted only if it strictly improves the missing-article count.

### 5.5 Article-Lookup Rescue

**Failure mode.** Even after retry, a class of hallucinations persisted: the LLM would cite an article that *did* exist in the dataset, just not in the top-k chunks for that specific query. Inspection of the chunk metadata revealed that 14,882 of 47,028 chunks (31.6 %) carried `referenced_articles` metadata, covering 1,494 distinct article numbers — including, for instance, 34 chunks referencing Article 87 and 14 chunks referencing Article 300, both articles the baseline LLM cited "from memory."

**Mechanism.** At startup, `ArticleLookupService.load()` performs a single pass over `chunks.pkl` and builds an index `{article_number → [chunk_index, …]}`. Chunks are ranked within each article's list by a three-tier quality score:

1. **Document-type tier**: `penal_code` and `criminal_procedure` (the actual law text) > `cassation_ruling`, `criminal_law_reference`, `legal_rules_collection` > `cassation_encyclopedia`, `legal_reference`. This prevents, for example, Rome-Statute (ICC) passages from the Encyclopedia of the Counsellor from being preferred over chunks from the Egyptian code.
2. **Article focus**: chunks referencing *fewer distinct articles* are preferred, on the heuristic that they are more focused on the queried article specifically rather than mentioning it as a passing cross-reference.
3. **Text length**: longer chunks are preferred as a final tiebreaker, on the assumption that they are more likely to contain a complete article definition.

When post-retry evidence validation still flags missing articles, the rescue path is triggered: the top-2 chunks for each missing article are retrieved, concatenated into a `rescue_context` block, and a *third* LLM call is made with the `qa_rescue_with_lookup` prompt. This prompt explicitly forbids the LLM from inventing *neighbouring* articles (e.g., if the references contain Article 87, the LLM must not write "88, 89, 90, 91, 92" as a sequential list unless each of those numbers also appears in the provided text) and explicitly forbids citing a passage that is about a *different law or topic* than the user's question.

The rescued answer is validated against the *combined* context (original retrieval + rescue chunks). It is accepted only if grounding improves; otherwise the original answer is preserved with a hallucination warning.

**Result.** Three of the four remaining hallucinations after the v4 retry pipeline (specifically Q18 citing Articles 30/31, the cases corresponding to Articles 122/123 of the Penal Code, and the case for Article 134) were converted to passing answers by the article-lookup rescue. Q18 went from a fail (in v3 through v5) to a pass at 48 % confidence in v6 with the explicit warning `تم استرجاع مواد إضافية من قاعدة البيانات للتحقق من الاستشهادات (مواد: 30, 31)`.

### 5.6 Iterative Retrieval (Adaptive k-Expansion)

**Mechanism.** Following the spirit of [4]'s recommendation that retrieval quality is the dominant determinant of downstream correctness, we treat the retrieval window-size `k` as an *adaptive* parameter rather than a fixed hyperparameter. After the initial generation and evidence validation, if validation flagged missing articles AND `USE_ITERATIVE_RETRIEVAL` is enabled (default), the system re-runs *only* the retrieval stage — not the LLM — at successively larger `k` values from the `ITERATIVE_K_SEQUENCE` (default `[7, 14, 21]`). Any new chunks not already in the context are appended, and the validator is re-run on the combined context. If any of these widened windows surfaces chunks containing the missing articles, validation passes *for free* (no LLM call). Only if this cheap expansion still fails does the corrective retry (Section 5.4) and rescue (Section 5.5) take over, but now with a wider context already in hand.

This mechanism has favourable cost characteristics: easy queries (those that pass at the initial `k`) pay nothing extra; hard queries pay 2–3 cheap retrieval cycles (~1 s each on CPU) before incurring an LLM call. A user-facing Arabic warning (`تم توسيع نطاق البحث تلقائياً إلى k مرجعاً`) is surfaced whenever the expansion fires.

### 5.7 Answer Post-Processing

**Mechanism.** A deterministic, regex-driven post-processing layer (`postprocess_answer`) executes after the final answer (post-retry, post-rescue) is selected and before it is returned to the user. Two transformations are applied:

1. **Casual-opener stripping.** A curated list of conversational fillers (`حسناً`, `بالتأكيد`, `تمام`, `طبعاً`, `سوف أجيب`, `لقد قرأت السؤال`, `بالنسبة لسؤالك`, …) is stripped from the answer's leading position if present. The list also includes longer multi-word openers (`سوف أجاوب على أسئلتك بتفصيل`) ordered longest-first to prevent stranded fragments.
2. **Refusal-phrase normalisation.** When the LLM splices the template-refusal phrase `المادة المطلوبة غير متوفرة في السياق المقدم` mid-clause (a frequent observation in pre-v6 outputs, e.g., `وفقاً للمادة المطلوبة غير متوفرة في السياق المقدم، يبدو أن السؤال غير مكتمل`), the regex `_EMBEDDED_REFUSAL_RE` matches the awkward construction — accounting for Arabic prefix-contraction rules (`ل + ال → لل`) — and rewrites it as a standalone sentence (`النصوص المقدمة لا تتضمن المادة المطلوبة.`).

A separate classifier `looks_like_refusal` detects whether the post-processed answer is dominated by refusal markers; if so, a `is_refusal` warning is surfaced. Confidence is not capped automatically (an earlier design choice we reverted in v5 after observing it dragged down the mean on borderline partial-but-valid answers).

---

### 5.8 Procedural-Defence Reasoning Checklist (Case Analysis)

**Failure mode.** The grounding mechanisms in Sections 4.1–4.7 ensure that what the system *cites* is correct; a complementary failure mode concerns what it *omits*. The case-analysis endpoints (`/weakness`, `/defense`, `/forensic`) consume a full criminal case file rather than a single question, and a domain-expert review of their output scored the legal reasoning at roughly 70 %: the analyses were well-grounded but missed several high-value procedural-nullity and criminal-intent arguments that a practising Egyptian criminal-defence lawyer applies routinely — and in one case *over-claimed* a defect that did not exist.

**Mechanism.** We encode the expert feedback as a **procedural-defence checklist** of five reasoning procedures, appended to the system prompts of the three case-analysis endpoints and reinforced in their user templates:

- **A. Timeline vs. prosecution warrant** — compare the exact arrest/search time against the time the prosecution warrant (`إذن النيابة`) issued; a seizure preceding the warrant voids the arrest and everything built on it (`ما بُني على باطل فهو باطل`). The rule explicitly instructs the model to read `الساعة 12:00 صباحاً` as the *start* of the day, correcting a clock-reading error observed in baseline output.
- **B. Identifier mismatch** — compare every identifier in the warrant (vehicle plate numbers, names, IDs) against the seizure record; any discrepancy places the seized item outside the warrant and may evidence `تلفيق`.
- **C. Territorial jurisdiction** — apply correctly and do *not* over-claim spatial excess for a location inside the precinct. This point is a precision correction (a false-positive suppressor), not an additional-recall rule.
- **D. Chain of custody** — challenge the integrity of the sealed exhibits (`التحريز`) when the officer who sealed them differs from the one who wrote the seizure record.
- **E. Intent from profession** — weigh the defendant's occupation against the nature of any seized cash to contest trafficking intent (`انتفاء قصد الاتجار`).

The checklist is paired with a single fully-worked, *fictional* few-shot exemplar demonstrating all five points end-to-end. Crucially, the exemplar is **grounding-safe by construction**: it names defence doctrines (`بطلان القبض والتفتيش`, `انتفاء قصد الاتجار`) but contains *no* `المادة N` article number, so it cannot teach the model to emit an ungrounded citation that the evidence validator (Section 5.3) would flag — raising reasoning coverage while preserving the 0 % hallucinated-citation property. Each checklist item is explicitly conditioned on the facts supporting it ("raise a point only when the facts genuinely support it — never invent one"), so the additions trade no precision for their gain in recall.

A companion **charge-fidelity** rule extends the same precision principle to the offences themselves: the case-analysis endpoints are constrained to enumerate only offences actually charged or described in the case file, and are explicitly forbidden from inventing an uncharged offence (e.g., adding `قيادة بدون رخصة` when the file never mentions a licence). This closes a fabrication mode observed during live validation, where the generator appended a plausible-but-uncharged offence to an otherwise grounded analysis.

### 5.9 Agentic Self-Check for Defence Memoranda

The defence-memorandum endpoint (`/defense`) applies an additional agentic grounding pass on top of the per-answer evidence validation of Section 5.3. Because a memorandum is a long, multi-paragraph document, a single evidence-validation gate is coarse: it flags an ungrounded article but does not by itself repair the surrounding argument. Instead, the draft memorandum is fed back to the LLM under a dedicated reviewer prompt (`verify_memo`) whose sole instruction is to make the memo *fully grounded* in the provided material — removing or correcting any article number not present in the retrieved texts, deleting any assertion the case facts do not support, and introducing no new citation or fact — while preserving the four-part structure (`الوقائع`, `الإطار القانوني`, `أوجه الدفاع`, `الطلبات`). This draft → verify → revise loop runs up to `MEMO_SELF_CHECK_MAX_ITERS` times (default 1) and stops early once a pass introduces no further change. The number of revision passes applied is surfaced to the client as the `self_check_revisions` response field, and the mechanism is toggleable via `MEMO_SELF_CHECK`. This self-check is one stage of the larger multi-agent pipeline described next.

### 5.10 Multi-Agent Agentic Case-Analysis Pipeline

Single-question Q&A is well served by the layered defence pipeline of Sections 5.1–5.7. *Open-ended case work* — reading a full criminal case file and producing a weakness analysis or a defence memorandum — is qualitatively harder: it requires extracting structured facts, hypothesising multiple independent lines of defence, grounding each one in its own statutory and precedential authority, judging which arguments actually hold, and only then writing. A single-shot "case → LLM → output" prompt collapses all of these into one undifferentiated generation and tends both to miss high-value arguments and to assert claims it cannot support. We therefore replace the single-shot path for `/weakness` and `/defense` with an **agentic, multi-agent pipeline** (`USE_AGENTIC_PIPELINE`, on by default) that decomposes the task into six specialised agents with a deterministic orchestrator.

**Agents and orchestration.** Intermediate agents are forced to emit strict JSON (robustly parsed with a balanced-brace fallback), so each stage's output is machine-checkable before the next consumes it.

**Table 7: Roles of the six agents in the case-analysis pipeline.**

| # | Agent | Function | Output |
|---|-------|----------|--------|
| 1 | Document Analysis | Extract structured facts, parties, charges, evidence items, and a dated/timed chronological timeline from the case file | JSON: facts, charges, `timeline` |
| 2 | Weakness Detection | Turn the analysis into typed weakness hypotheses, each with a focused legal-research query | JSON list of weaknesses |
| 3–4 | Legal Research & Precedent Retrieval | For each top weakness, one hybrid retrieval whose pool yields both doctrine/statute and court-precedent chunks (split by `doc_type`); run concurrently | Contexts + sources per weakness |
| 5 | Defence Strategy | Rank weaknesses against the *retrieved* authorities, keep the strong, discard the disproven, order the argument | JSON: kept + ordered IDs |
| 6 | Synthesis | Write the final Arabic weakness analysis or four-part defence memorandum from the kept, grounded authorities only | Arabic text + sources |

The flow is: **Document Analysis → Weakness Detection → (Legal Research ‖ Precedent Retrieval, per weakness) → Defence Strategy → Synthesis**, with the memo self-check (Section 5.9) applied to the synthesised memorandum. The orchestrators `run_weakness_pipeline` and `run_defense_pipeline` share one private `_run_pipeline`. After Agent 2, only the top `AGENT_MAX_WEAKNESSES` (= 6) weaknesses are researched in depth; the rest are carried with their hypothesis text. The per-weakness retrievals (Agents 3–4) are launched concurrently with `asyncio.gather`, but actual CPU concurrency is bounded by the reranker semaphore, so a full memo stays CPU-affordable (~2–3 minutes rather than ~10). A single weakness failing to retrieve does not sink the run (`return_exceptions=True` filters it out); only if *every* weakness fails, or a critical agent errors, does the orchestrator raise `AgentPipelineError` — at which point the router silently falls back to the single-shot path. The pipeline is thus never a hard dependency, and it returns the structured intermediates (timeline, weaknesses, authorities, sources) alongside the final text, so the client can render the reasoning trace, not just the conclusion.

**Neuro-symbolic grounding: the deterministic timing verdict.** The single most damaging recurring error in case analysis was *chronological*: the model repeatedly mis-read whether the arrest/search happened before or after the prosecution warrant (`إذن النيابة`) — inverting the order even when the timeline was extracted correctly — and a wrong verdict here flips the entire defence. Rather than prompt harder, we move this judgement out of the LLM entirely. `compute_timing_verdict` parses each timeline event's Arabic date and time string (handling `صباحاً`/`مساءً`/`منتصف الليل` and the `الساعة 12:00 صباحاً`-as-midnight convention), computes the signed gap between warrant and seizure in code, and emits one of three deterministic verdicts: seizure *before* the warrant (the nullity argument `بطلان القبض والتفتيش` applies, with `ما بُني على باطل فهو باطل`), *within* the 24-hour validity window (the timing is sound and the "arrest preceded the warrant" plea is *forbidden* as contrary to the record), or *after* the window expired (a different nullity applies). This verdict is injected into the downstream agents' context as a hard fact they cannot re-derive incorrectly — a small but decisive neuro-symbolic component that converts a stubborn reasoning failure into a solved sub-problem.

---

## 6. Document Upload Pipeline

The system supports three distinct retention semantics for user-supplied documents, addressing different operational use cases observed during user testing.

### 6.1 Per-Question Attachment (`POST /api/v1/qa/upload`)

A `multipart/form-data` endpoint that accepts a `question` field, an optional `file` field (`.txt` / `.pdf` / `.docx`), and an optional `text` field (raw string). The attached content is parsed (Section 6.4), wrapped in `[المستند المرفق]` delimiters, and *prepended* to the retrieved legal context for **this request only**. The attachment is not persisted and does not enter the permanent index. Article numbers cited in the answer can come from *either* the attachment or the retrieved corpus — the evidence validator (Section 5.3) operates on the union. The input gate (Section 5.1) is skipped when an attachment is present.

### 6.2 Session-Attached Document (`POST /api/v1/chat/attach`)

A multipart endpoint that *binds* a document to a session, persisting it alongside the conversation history. Each subsequent chat turn prepends *every* attachment to the LLM context (via `Session.format_attachments()`, which applies a per-document length cap to bound the context budget). Attachments survive server restarts via the existing JSON session-persistence machinery. Companion endpoints `GET /chat/{sid}/attachments`, `DELETE /chat/{sid}/attachments/{doc_id}`, and `DELETE /chat/{sid}/attachments` provide the full lifecycle.

### 6.3 Permanent Corpus Ingestion (`POST /api/v1/ingest`)

The pre-existing ingest endpoint, surfaced in the Streamlit UI through this work. Uploaded documents are chunked using the doc-type-aware chunking configuration (`CHUNKING_CONFIGS`), embedded via the same embedding model used at startup, and merged into the FAISS and BM25 indices on disk. The retrieval service is then hot-reloaded so subsequent queries see the new data without a server restart.

### 6.4 Shared Parsing Layer (`POST /api/v1/parse`, `services/upload_helper.py`)

To avoid behavioural divergence across the three modes, all parsing flows through a single helper module. The parser dispatches by file extension:

- `.txt`: read with a fallback sequence of Arabic encodings (UTF-8 → CP1256 → ISO-8859-6);
- `.pdf`: extract via `pdftotext -layout -enc UTF-8` (poppler-utils) when available, falling back to PyPDF2;
- `.docx`: extract via `docx2txt`;
- `.doc` (legacy binary): explicitly rejected with a helpful Arabic error directing the user to save as `.docx` or `.pdf`.

All extracted text is Arabic-cleaned (diacritic removal, Arabic-Indic digit normalisation, alef-form unification, page-number stripping). A 100,000-character cap bounds memory; a 30-character minimum filters empty or OCR-failed uploads. A `/parse` endpoint exposes this layer directly, allowing UI clients to preview and edit the parsed text before submitting it for analysis.

### 6.5 Automated Ingestion and Hot-Reload

Permanent ingestion is also available without a manual API call. A watch-folder daemon (`watch_ingest.py`) polls an inbox directory (`INGEST_INBOX_DIR`) every `INGEST_WATCH_INTERVAL_S` seconds (default 30) and ingests any new files through the same parser and Arabic-cleaning path as the manual pipeline, tracking already-processed paths in `data/.ingested_files.json` so the operation is idempotent. The same scan is exposed as `POST /api/v1/ingest/scan`, which runs the scan in a worker thread and then calls `reload_indices()` to hot-swap the FAISS, BM25, chunk, and tokenised-corpus state of the running service *in place* — without re-initialising the embedding model or restarting the process. The four index files are always written and reloaded together to keep retrieval state coherent.

---

## 7. Evaluation

### 7.1 Benchmark

We constructed a 28-question Arabic legal benchmark spanning three difficulty tiers (basics, application, advanced) and covering the substantive (Penal Code) and procedural (Code of Criminal Procedure) law domains. The benchmark deliberately includes:

- **5 well-formed substantive questions** (e.g., penalty for armed robbery, conditions of legitimate self-defence, elements of premeditated murder);
- **1 procedural question** (`شروط التوقيف الاحتياطي`) that historically triggered wrong-law confusion in baseline systems;
- **5 input-gate test cases**: markdown headings (`## المستوى الأول`), bullet fragments (`* الجناية`, `* الجنحة`, `* المخالفة`), and colon-terminated incomplete sentences (`1. ما المقصود بمبدأ:`);
- **17 application and reasoning questions** drawn from undergraduate criminal-law curricula, ranging from simple case analyses to comparative questions (`ما الفرق بين القتل العمد والقتل الخطأ؟`).

Each question is evaluated against two metrics: (i) **evidence-validation status** — whether every article number cited in the answer can be traced to the retrieved (or rescued) context, and (ii) **confidence score** as computed by Section 4's heuristic. The primary axis of evaluation is hallucination rate: the percentage of questions for which evidence validation fails.

### 7.2 Evaluation Protocol

A CLI harness (`run_eval.py`) posts each of the 28 questions to `POST /api/v1/qa` with `k = 7` and writes the per-question result (question, first 150 characters of answer, confidence %, validation status, warning count, source count, API/total/retrieval latency) to a CSV. A 4-second pause separates consecutive requests to keep the Groq free-tier 6,000-tokens-per-minute budget refilled. A second harness (`test_qa.py`) provides a human-readable per-question report with pass/fail/refusal classification and an exit code suitable for CI integration.

### 7.3 Results

Table 1 summarises the headline metrics across five system versions corresponding to the cumulative introduction of each grounding-defence layer.

**Table 1: Evaluation across five system versions (28-question benchmark)**

| Metric                          |   v3 baseline |        v4 |        v6 |        v7 |              v8 |
|---------------------------------|--------------:|----------:|----------:|----------:|----------------:|
| Passed evidence validation      |     20/28 (71.4 %) | 26/28 (92.9 %) | 27/28 (96.4 %) | 27/28 (96.4 %) | **28/28 (100.0 %)** |
| Hallucinated citations          |      8/28 (28.6 %) |  2/28 (7.1 %) |  1/28 (3.6 %) |  1/28 (3.6 %) |        **0/28 (0.0 %)** |
| Average confidence              |          41.2 % |    38.0 % |    39.8 % |    40.0 % |          41.9 % |
| Input-gated (no LLM call)       |              0 |        8 |        8 |        8 |               8 |

The progression of hallucination rate across versions corresponds to the cumulative introduction of each layer:

- **v3 → v4**: input gate + corrective retry + tightened prompts → **−21.5 percentage points** (8 → 2 hallucinations).
- **v4 → v6**: article-lookup rescue → **−3.5 percentage points** (2 → 1).
- **v6 → v7**: rescue prompt tightening + quality-tiered chunk ranking → **0 net change** in pass rate, but the *specific* failing question shifted (LLM non-determinism).
- **v7 → v8**: LLM upgrade from `llama-3.1-8b-instant` to `llama-3.3-70b-versatile` → **−3.6 percentage points** (1 → 0).

We note that the v5 measurement (multi-query synonym expansion) is omitted from Table 1 because that mechanism was found to *regress* performance on the current index: synonym expansion diluted retrieval quality by pulling in less-relevant chunks. The multi-query implementation is retained in the codebase but disabled by default (`USE_MULTI_QUERY=False`) — a useful negative result we report in detail in Section 9.

### 7.4 Qualitative Observations

Two questions are particularly informative:

- **Q3 — التوقيف الاحتياطي.** In v3, this question triggered the wrong-law failure mode: the LLM cited Article 300 of the Penal Code. After the prompt-level disambiguation in v4, the LLM correctly identifies the question as procedural and produces a refusal-style answer when retrieval does not surface the relevant Code of Criminal Procedure chunks. Validation passes (no fabricated citation), at 59 % confidence with a refusal-detection warning surfaced.
- **Q18 — هل تعتبر الجريمة تامة أم شروع؟** This question proved the most persistent failure: it failed in v3 (at 53 % confidence), regressed under v4's pure-prompt approach (citing Articles 30 and 31), and recovered to passing only with the article-lookup rescue in v6 (`تم استرجاع مواد إضافية من قاعدة البيانات للتحقق من الاستشهادات (مواد: 30, 31)`, 48 % confidence). The v7 stricter rescue prompt initially introduced a regression on this question (the LLM hallucinated a fresh Article 32 instead) which the v8 LLM upgrade resolved.

These observations underline a key point: **no single mechanism is sufficient**. The v3-baseline failures distribute across distinct root causes (input fragments, prompt ambiguity, retrieval coverage gaps, LLM memory citations), and each requires its own layer of defence. The cumulative pipeline reduces every category to zero in v8, but each layer contributes meaningfully and removing any one would degrade performance.

Table 1b disaggregates the headline pass-rate by question category, exposing which defence layers recover which failure modes.

**Table 1b: Pass rate by question category across versions.**

| Category | v3 | v4 | v6 | v7 | v8 | total |
|----------|----|----|----|----|----|-------|
| Substantive | 8/11 | 9/11 | 10/11 | 10/11 | **11/11** | 11 |
| Procedural | 0/1 | 1/1 | 1/1 | 1/1 | **1/1** | 1 |
| Reasoning | 7/8 | 8/8 | 8/8 | 8/8 | **8/8** | 8 |
| Fragment | 5/8 | 8/8 | 8/8 | 8/8 | **8/8** | 8 |
| **Total** | 20/28 | 26/28 | 27/28 | 27/28 | **28/28** | 28 |

Beyond headline metrics, Table 1c contrasts *capabilities* with the related Arabic-legal-RAG work. The cited prior work concentrates on the retrieval stage; none reports a post-generation grounding-defence pipeline, an agentic case-analysis flow, or an open Egyptian-criminal-law corpus.

**Table 1c: Capability comparison with related Arabic-legal-RAG work (✓ present, ✗ not reported).**

| Capability | El-Beltagy [1] | Hrimech [2] | Aboasal [5] | ALARB [3] | Conan |
|------------|:--:|:--:|:--:|:--:|:--:|
| Hybrid retrieval (dense+sparse) | ✗ | ✗ | ✓ | ✗ | ✓ |
| Cross-encoder reranking | ✗ | ✗ | ✗ | ✗ | ✓ |
| Post-gen. hallucination defence | ✗ | ✗ | ✗ | ✗ | ✓ |
| Evidence/citation validation | ✗ | ✗ | ✗ | ✗ | ✓ |
| Agentic multi-agent pipeline | ✗ | ✗ | ✗ | ✗ | ✓ |
| Neuro-symbolic reasoning tool | ✗ | ✗ | ✗ | ✗ | ✓ |
| Both substantive + procedural law | ✗ | ✗ | ✗ | ✗ | ✓ |
| Document upload / ingestion | ✗ | ✗ | ✗ | ✗ | ✓ |
| Production service + wire contract | ✗ | ✗ | ✗ | ✗ | ✓ |
| Open released corpus | ✗ | ✗ | ✗ | ✓ | ✓ |

### 7.5 Cross-Model Benchmark (Protocol)

To situate Conan against general-purpose frontier models on Arabic legal questions, we provide a head-to-head harness (`benchmark_llms.py`) that queries our full RAG system alongside `openai/gpt-4o-mini` and `anthropic/claude-sonnet-4.5` (both routed through a single OpenRouter key for cost parity). It runs in two modes that isolate two distinct questions: **RAG mode**, where all three models answer from the *same* retrieved context (isolating answer-synthesis quality from retrieval), and **raw mode**, where each model answers from its own parametric knowledge with no retrieval (probing baseline Arabic criminal-law knowledge). Each run emits a per-model CSV recording latency, answer length, and cited article numbers, against which the same evidence-validation extractor (Section 5.3) is applied to measure citation grounding on equal footing.

*Quantitative results are pending a full benchmark run and will be reported in a subsequent revision;* at the time of writing the comparison could not be executed because of API-access limits on the shared OpenRouter key. The protocol is documented here so the comparison is reproducible once access is restored.

---

### 7.6 Retrieval-Quality Optimisation (v9): Embeddings, Chunking, Reranking, and a Scored Harness

The v3→v8 evolution (Table 1) drove *citation fidelity* — what the system cites, given its context — to a 0 % hallucination rate. It did not, however, address the complementary ceiling: whether the authoritative article ever *reaches* the context in the first place. A grounding pipeline can only validate citations against what retrieval surfaces; if the relevant article never enters the top-`k`, the best achievable outcome is an honest refusal, not a correct answer. The v9 work therefore targets retrieval quality directly, along four axes, and — crucially — introduces the measurement instrument that makes such tuning auditable.

**Motivation: the retrieval ceiling.** Three structural weaknesses bounded retrieval recall in the deployed system. (i) The embedding model was `paraphrase-multilingual-MiniLM-L12-v2`, a 384-dimensional general-purpose multilingual encoder chosen for fast CPU rebuilds; it is not domain-tuned and is comparatively weak on Arabic legal phrasing, and its 512-token cap silently truncated the longer statute chunks. (ii) Chunking was purely character-recursive, so a single article's text could be split across a chunk boundary — or share a chunk with an unrelated neighbouring article — diluting the dense and lexical signal for the exact unit a citation-driven system must retrieve. (iii) The cross-encoder reranker was configured to re-score only the top-10 RRF candidates, so a relevant chunk ranked 11th or lower by fusion could never be promoted, regardless of its true relevance.

**Enhancement 1 — Embedding upgrade (MiniLM-384 → BGE-M3-1024).** We integrate `BAAI/bge-m3` (1,024-dimensional) behind the embedding configuration as the target encoder: a multilingual model with strong Arabic retrieval performance and native long-context support (sequence length raised from 512 to 1,024, eliminating chunk truncation). This is consistent with Aboasal et al. [5], who report substantially higher MAP on a comparable Arabic legal retrieval task with BGE-M3 over lighter encoders. Because vector dimensionality changes, deploying it requires a full FAISS rebuild, performed into a staging directory so the live index keeps serving and is swapped atomically (`scripts/swap_new_index.sh`). The rollout is deferred (see *Deployment status* below) but the path is fully prepared.

**Enhancement 2 — Article-aware chunking.** For statute-structured document types (Penal Code, Code of Criminal Procedure, and legal-reference collections), the chunker (`app/services/chunking.py`) now splits on article-header boundaries (`المادة N`, with Arabic-Indic digits already normalised), so each article is emitted as a single intact chunk; only an article whose text exceeds the doc-type chunk size is sub-split, and narrative material (case files, cassation rulings, the encyclopedia) retains recursive character splitting where no reliable article scaffold exists. Each article-anchored chunk carries a new `primary_article` field — the article the chunk *is*, as opposed to one it merely cross-references — which is preferred when populating the per-source `article` field and feeds the article-validation and topic-match confidence signals more precisely. The chunker is the single source of truth shared by the batch index builder and the incremental ingest pipeline, so the two index-writers cannot drift.

**Enhancement 3 — Wider reranker candidate pool.** The cross-encoder now re-scores the top-50 RRF candidates rather than the top-10 (`RETRIEVAL_K_RERANK`), allowing it to rescue relevant chunks that fusion ranked low. The cost is a modest increase in per-query CPU reranking time; no rebuild is required, making this the one zero-cost, immediately-deployable lever of the four.

**Enhancement 4 — A scored evaluation harness.** Earlier evaluation (Sections 7.1–7.3) relied on per-question CSV dumps read by a human. To make retrieval tuning measurable rather than impressionistic, we add `scripts/eval_harness.py`, which scores a labeled gold set and emits a machine-readable scorecard for before/after comparison. It runs in two modes: a **retrieval mode** (LLM-free) that measures the retrieval ceiling — *recall@k* (fraction of queries whose context contained the authoritative article), *article-recall* (mean fraction of a query's expected articles retrieved), *primary-hit@k* (whether a retrieved chunk *is* an expected article — a direct test of article-aware chunking), and *MRR*; and an **end-to-end mode** that scores citation recall, hallucination rate, out-of-scope abstention, confidence calibration (mean confidence on correct vs. incorrect answers), and latency percentiles against the live API. A `--compare` mode diffs two scorecards directly. This harness is the backbone for the before/after comparison once the embedding rollout lands, and for all future tuning.

**Deployment status and the CPU-embedding constraint.** The four enhancements differ in deployment cost. The widened reranker pool (Enhancement 3) is a runtime configuration change and is **live** with no rebuild. Article-aware chunking (Enhancement 2) is **implemented and validated**, and is applied online to every newly-ingested document; on a 100-file validation sample it produced 14,060 chunks of which **11,152 (79 %) were article-anchored**, confirming that the bulk of statute content is now cut at article boundaries rather than arbitrary character offsets. The scored harness (Enhancement 4) is **delivered**.

The embedding upgrade (Enhancement 1), however, requires re-embedding the entire corpus, and here we report a concrete systems result: on the CPU-only reference machine (the GPU is unusable — its compute capability 5.0 is below the floor required by current PyTorch builds), BGE-M3 embeds at roughly **5.5 s per chunk** at 1,024-dim/1,024-token, which — compounded by the ~3× chunk-count increase from article-aware splitting — projects to **multiple days** of continuous computation for a full-corpus rebuild. We therefore **defer the embedding rollout to a GPU-equipped environment** rather than peg the production CPU for days; the configuration, build/swap tooling, and evaluation harness are all in place so the rebuild and its before/after scorecard (`eval_harness.py --mode retrieval`, then `--compare`) reduce to a single batch run once a GPU is available. This is itself a useful deployment lesson for resource-constrained Arabic-RAG teams: the embedding model that maximises retrieval quality may be infeasible to *build* without accelerator access, making the embedder choice a joint quality/operability decision rather than a pure quality one.

**Positioning.** v9 is qualitatively different from v3–v8: the earlier layers are *defensive* (they prevent the model from asserting what it cannot support), whereas v9 is *generative of recall* (it raises what the model can legitimately support). The two are complementary — a higher retrieval ceiling means the grounding pipeline converts more queries into grounded answers rather than honest refusals, without relaxing the 0 % hallucination property, since every citation still passes the same evidence validator.

---

## 8. Implementation, Deployment, and Integration

**Service and persona.** Conan is packaged as a FastAPI service with a thin router layer over the singleton retrieval, reranker, article-lookup, and session services, all loaded once at start-up. All user-facing output is in Modern Standard Arabic under a consistent government-style persona (`كونان`), enforced by the system prompts; the persona is deliberately formal and citation-bound rather than conversational.

**Reference UI.** A Streamlit application ships as the reference client, exercising every endpoint (Q&A, multi-turn chat with streaming, weakness analysis, defence memo, forensic check, summarisation, and document upload) and serving as living documentation of the wire contract.

**Containerisation and exposure.** The service is containerised with a `Dockerfile` and `docker-compose.yml` (documented in `DEPLOY.md`) for reproducible deployment. For demonstration and remote integration the laptop-hosted backend has been exposed over a Cloudflare quick tunnel, allowing an external frontend team to integrate against a live instance without dedicated hosting.

**Frontend integration contract.** The API is consumed by a separate .NET frontend. To support this, the response schema is treated as a frozen wire contract: the structured shape returned by `/qa`, `/chat`, `/weakness`, `/defense`, and `/forensic` — `answer`/`analysis`/`memorandum`, plus `confidence_score`, `confidence_factors`, `sources[]` (with per-source `legal_topic`, `article`, `referenced_articles`, `page`, `retrieval_score`, `rerank_score`), `warnings[]`, `conflicts_detected`, `latency_ms`, and `model` — is documented field-by-field in two contract files (`api_contract.md`, `backend_contract.md`) and field names cross the wire verbatim. Arabic-language clarification `warnings` are designed to be forwarded to the end-user as-is.

**Configurability.** Operational parameters are centralised in a single `config.py` (every value overridable by an environment variable; new vars are mirrored in `.env.example`), so behaviour can be tuned for a deployment without code changes. Table 8 lists the most consequential tunables and their production defaults.

**Table 8: Principal configuration parameters and production defaults.**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `RETRIEVAL_K` | 10 | Final chunks passed to the LLM |
| `RETRIEVAL_K_DENSE/SPARSE` | 60 / 60 | FAISS / BM25 candidate pool before fusion |
| `RETRIEVAL_K_RERANK` | 60 | Fused candidates re-scored by the cross-encoder |
| `RRF_K` | 60 | Reciprocal-Rank-Fusion constant |
| `RERANK_MAX_CONCURRENCY` | 2 | Cap on simultaneous CPU reranks |
| `CONFIDENCE_WEIGHTS` | .30/.20/.30/.20 | rerank / sources / article-valid. / topic-match |
| `RERANK_CALIBRATION_GAMMA` | 0.5 | Lifts squashed low sigmoid rerank scores |
| `ITERATIVE_K_SEQUENCE` | 10, 18 | Adaptive *k*-expansion on validation failure |
| `RETRY_MAX_ATTEMPTS` | 1 | Block-list corrective retries |
| `QA_CASCADE_BUDGET_S` | 90 | Wall-clock budget for the corrective cascade |
| `LLM_MIN_INTERVAL_S` | 2.0 | In-process rate gate between provider calls |
| `AGENT_MAX_WEAKNESSES` | 6 | Weaknesses given deep per-weakness research |
| `AGENT_RESEARCH_K` | 6 | Chunks retrieved per weakness |
| `MEMO_SELF_CHECK_MAX_ITERS` | 1 | Memo verify → revise passes |
| `MAX_CONTEXT_CHARS` / `MAX_INPUT_CHARS` | 12k / 50k | Context budget / max case-file size |
| `SESSION_MAX_TURNS` / `KEEP_RECENT` | 6 / 3 | Compaction trigger / verbatim window |
| `SESSION_TTL_HOURS` | 168 | Idle-session pruning (7 days) |

---

## 9. Strengths, Limitations, Challenges, and Lessons Learned

### 9.1 Strengths and practical impact

The system's principal strengths follow directly from its design philosophy. **(1) Verifiable grounding**: every statutory citation is machine-checked against retrieved text, yielding a 0 % hallucinated-citation rate on the benchmark — the property that most directly governs whether a legal-AI tool can be trusted in practice. **(2) Defence in depth**: seven independently-toggleable grounding layers plus a six-agent case-analysis pipeline cover heterogeneous failure modes that no single mechanism addresses. **(3) Operability under constraint**: a five-provider failover chain, zero-token cache, rate gate, and cascade budget let the full system run on free LLM tiers and CPU-only hardware — the realistic setting for most academic and public-sector Arabic-legal deployments. **(4) An open corpus**: the released 1,057-document corpus lowers the entry barrier for future Egyptian-criminal-law work. **(5) Real integration**: a frozen wire contract and a live-tunnel deployment let an external .NET team build against the service, demonstrating the architecture beyond a notebook prototype.

### 9.2 Challenges encountered and how they were addressed

**Table 9: Key challenges and the mechanisms they motivated.**

| Challenge | Resolution |
|-----------|-----------|
| Legacy `.doc` / scanned-PDF corpus, unreadable to NLP tooling | Manual OCR + multi-stage Arabic cleaning of 933 files (Section 3) |
| LLM cites articles from parametric memory | Evidence validation + block-list retry + article-lookup rescue (Sections 5.3–5.5) |
| Relevant article never enters top-*k* | Iterative *k*-expansion + widened rerank pool + article-aware chunking (Sections 5.6, 7.6) |
| Substantive-vs-procedural law confusion | Prompt-level law-naming disambiguation (Section 5.2) |
| Model inverts arrest-vs-warrant chronology | Deterministic, code-computed timing verdict injected as a hard fact (Section 5.10) |
| Free-tier token/minute exhaustion under multi-call requests | Rate gate, answer cache, single-retry cap, OpenRouter-primary (Section 4) |
| CPU rerank latency dominating per-query cost | Concurrency semaphore + cascade budget + bounded iterative-*k* |
| Long memos hallucinate *arguments*, not just article numbers | Agentic draft → verify → revise self-check (Section 5.9) |
| BGE-M3 infeasible to build on CC-5.0 GPU | Defer rollout; prepare staging-build + atomic-swap tooling (Section 7.6) |

### 9.3 Lessons learned

Four lessons generalise beyond this system. **(L1) Grounding is a post-generation problem as much as a retrieval problem** — improving the retriever alone never removed the parametric-memory citation failure; the defence layers did. **(L2) Move brittle reasoning out of the LLM when it can be computed**: the single largest case-analysis error (chronology) was eliminated not by a better prompt but by a few lines of deterministic code. **(L3) Not every clever idea helps**: heuristic synonym expansion measurably *regressed* retrieval (v5), a result we keep visible rather than bury. **(L4) The best model can be the wrong model**: on CPU-only hardware the highest-quality embedder is infeasible to *build*, making the embedder a joint quality/operability choice rather than a pure quality one.

### 9.4 Known limitations, weaknesses, and risks

**Multi-query expansion regression.** The deliberate negative result reported in v5 — that heuristic synonym expansion *reduces* pass rate on the current index — is consistent with the [1] observation that embedding-model semantics for Arabic do not translate cleanly across closely-related but lexically-distinct legal terms (e.g., `التوقيف الاحتياطي` vs `الحبس الاحتياطي`). We retain the implementation (`USE_MULTI_QUERY=False` by default) and report this as a cautionary data point. A semantically-aware query expansion using the LLM itself, rather than a static synonym table, may recover this lost potential.

**Retrieval ceiling (the dominant residual risk).** The grounding pipeline can only validate citations against what retrieval surfaces; if the authoritative article never reaches the context, the best outcome is an honest refusal, not a correct answer. With the live MiniLM-384 index this ceiling is the single largest bound on answer correctness, and it is exactly what the deferred BGE-M3 upgrade targets (Section 7.6).

**LLM cost and latency.** The full pipeline can incur up to 3 LLM calls per hard Q&A (initial generation, corrective retry, article-lookup rescue) and one call per agent for case work. The OpenRouter-primary move removed the token-per-minute ceiling, but CPU rerank latency still makes a full agentic memo a multi-minute operation — acceptable for considered legal drafting, but a bottleneck for interactive use until a GPU retrieval backend is available.

**Article-lookup precision.** The rescue mechanism operates on chunk *references* to article numbers, not on chunks that *define* those articles. In approximately 5 % of rescue invocations, the surfaced chunk text mentions the queried article in passing rather than providing the article's text. This suffices for validation but is suboptimal for answer quality.

**Evaluation scope.** Our 28-question benchmark is comparatively small relative to the 13,000-case ALARB benchmark of [3] and the 2,500-pair dataset of [2]. It was sized for fast, repeatable, manually-auditable iteration; the cross-model benchmark against frontier LLMs (Section 7.5) is implemented but its quantitative run is still pending API access.

### 9.5 Future Work

We group planned work into five themes, ordered roughly by expected impact on answer correctness.

**Retrieval ceiling: complete the BGE-M3 rollout.** The highest-leverage next step is the embedding upgrade already prepared behind configuration (Section 7.6): re-embedding the full corpus with BGE-M3 (1,024-d, 1,024-token, eliminating MiniLM's 512-token truncation) on GPU-equipped hardware, then running the scored harness's `--compare` to quantify the recall gain. A dedicated *article-definition* index — paragraph-level segmentation of the Penal Code so the rescue path returns the article's *text* rather than a passing cross-reference — would further raise rescue answer quality.

**Toward GraphRAG and a knowledge graph.** The current system is the Phase-1 instantiation of a longer roadmap. The natural architectural evolution is a legal knowledge graph linking articles, offences, defences, and precedents, over which GraphRAG-style multi-hop retrieval could answer questions that require chaining several articles (e.g. aggravating-circumstance + base-offence + penalty) rather than retrieving each in isolation. The `primary_article` / `referenced_articles` metadata already extracted is the seed for such a graph.

**Deeper and more autonomous agents.** The six-agent pipeline is a fixed DAG; future versions could let the orchestrator decide *dynamically* how many weaknesses to research and how deep to retrieve, add a dedicated *cross-examination* agent that adversarially attacks each drafted argument, and add tool-use agents (statute-version lookup, deadline calculators) alongside the existing deterministic timing tool. A semantically-aware, LLM-driven query expansion — the principled successor to the synonym-table approach that regressed in v5 — belongs here too.

**Evaluation at scale.** Scaling to a labelled gold set of comparable size to ALARB [3] (using the released corpus), completing the pending cross-model benchmark against frontier LLMs (Section 7.5), and adding human expert-rated reasoning-quality scoring would strengthen the camera-ready evaluation. Confidence-calibration curves over the larger set would also let the clarification threshold be set empirically rather than heuristically.

**Scalability, robustness, and a possible rebuild.** For higher load the singleton-on-one-process design would move to a horizontally-scaled deployment: the read-only FAISS/BM25 indices replicated behind a load balancer, sessions and the answer cache backed by a shared store (Redis) rather than per-process JSON, and the CPU reranker moved to a GPU inference service to remove the dominant latency cost. A future major version could also be *rebuilt* natively around the knowledge graph and an agent framework rather than retrofitting agents onto the RAG core — trading the current minimal, dependency-light design for richer orchestration once the retrieval upgrade's quality ceiling has been realised. Security and compliance hardening (authentication, rate limiting, audit logging, and PII handling for uploaded case files) is a prerequisite for any real-world legal deployment.

---

## 10. Conclusion

We have presented three complementary contributions: an open Arabic criminal-law corpus, manually constructed from 933 legacy-format source files into 1,057 cleaned documents and released publicly on HuggingFace; **Conan**, a hallucination-resistant Arabic legal RAG system whose seven grounding-defence mechanisms — input gating, citation-grounding prompts, evidence validation, corrective retry with a block-list, article-lookup rescue, iterative retrieval, and deterministic answer post-processing — drive the rate of hallucinated statutory citations from a baseline of 28.6 % down to **0.0 %** on a 28-question Egyptian-criminal-law benchmark; and a six-stage multi-agent agentic pipeline that brings the same grounding discipline to open-ended case work, anchoring every argument in retrieved authority and replacing a stubborn chronological-reasoning failure with a deterministic, code-computed verdict. We additionally introduce a unified document-upload subsystem supporting three retention semantics across `.txt`, `.pdf`, and `.docx` formats, and release the system, the evaluation harness, and the multi-version CSV evaluation trail to support reproducible auditing.

Our key methodological contribution is the demonstration that **no single mechanism is sufficient** for hallucination suppression in Arabic legal RAG; instead, a layered approach matching the heterogeneous failure modes of the underlying components (LLM memory citations, retrieval coverage gaps, fragmentary user inputs, wrong-law confusion, prompt-template leakage) is required, and open-ended legal reasoning additionally benefits from decomposition into specialised, individually-grounded agents and from moving brittle reasoning out of the model when it can be computed. The system's practical impact is to show that a trustworthy, verifiable Arabic legal assistant can be built and operated under genuine resource constraints — free LLM tiers and CPU-only hardware — the realistic setting for most academic and public-sector deployments. We hope this work informs and accelerates the deployment of Arabic legal-AI systems in production settings.

---

## Acknowledgements

The authors thank the open-source community behind FAISS, BM25, BGE-Reranker-v2-M3, the Groq LLM platform, Google Gemini, and OpenRouter, without whose tools this work would not be possible.

---

## References

[1] S. R. El-Beltagy and M. A. Abdallah, "Exploring Retrieval Augmented Generation in Arabic," *Procedia Computer Science*, vol. 244, pp. 296–307, 2024. *6th International Conference on AI in Computational Linguistics (ACLing 2024)*. DOI: 10.1016/j.procs.2024.10.203.

[2] J. Hrimech, M. Mghari, and Y. Zaz, "Retrieval-augmented generation for Arabic legal information: the family code case study," *TELKOMNIKA Telecommunication Computing Electronics and Control*, vol. 23, no. 6, pp. 1495–1505, Dec. 2025. DOI: 10.12928/TELKOMNIKA.v23i6.27400. ISSN: 1693-6930.

[3] H. Abu Shairah, S. AlHarbi, A. AlHussein, S. Alsabea, O. Shaqaqi, H. AlShamlan, O. Knio, and G. Turkiyyah, "ALARB: An Arabic Legal Argument Reasoning Benchmark," in *Proceedings of the Third Arabic Natural Language Processing Conference*, pp. 389–406, Association for Computational Linguistics, Nov. 8–9, 2025.

[4] M. Alghamdi, M. Abushawarib, M. Ellouh, M. Ghaleb, and M. Felemban, "Enhancing Arabic Information Retrieval for Question Answering," in *ICFNDS '23: Proceedings of the 7th International Conference on Future Networks and Distributed Systems*, Dubai, UAE, Dec. 21–22, 2023. DOI: 10.1145/3644713.3644763. ISBN: 9798400709036.

[5] R. Aboasal, S. Montasser, F. Hossam Eldin, A. Abdelwahab, and H. Abdelazim, "Arabic Legal Information Retrieval: The Impact of Morphological Segmentation and Semantic Embeddings," *Procedia Computer Science*, vol. 275, pp. 275–282, 2026. *7th International Conference on AI in Computational Linguistics (ACLing 2025)*.

[6] "Mini-RAG Project: Learnings & Technologies," internal RAG-implementation documentation, on file with the authors.

---

*Manuscript prepared as a draft for a future conference submission. Source code, evaluation CSVs (versions v3–v8), and the evaluation harness are available alongside this document in the project repository.*
