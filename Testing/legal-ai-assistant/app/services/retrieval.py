"""
FAISS + BM25 hybrid retrieval with Reciprocal Rank Fusion.
"""
import pickle
import time
import logging
import json
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

import config
from app.services.preprocessing import (
    preprocess_arabic_for_bm25,
    classify_query_domain,
    expand_query_synonyms,
    DOMAIN_TO_DOC_TYPES,
)
from app.services.reranker import reranker_service

logger = logging.getLogger(__name__)


class GoogleEmbedding:
    """Wrapper for Google Gemini Embeddings API."""

    def __init__(self, model_name: str, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import time
        # Gemini free tier has a strict limits
        all_embeddings = []
        batch_size = 20 # Lower to avoid strict TPM limits
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            current_batch = i // batch_size + 1
            logger.info(f"   ☁️ Sending batch {current_batch}/{total_batches} to Gemini Cloud (Throttled)...")
            
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                )
                if hasattr(result, 'embeddings'):
                    all_embeddings.extend([e.values for e in result.embeddings])
                else:
                    all_embeddings.extend([e.values for e in result])
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg:
                    logger.warning(f"   ⚠️ Hit rate limit. Sleeping for 30s...")
                    time.sleep(30)
                    # Retry once
                    result = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                    )
                    if hasattr(result, 'embeddings'):
                        all_embeddings.extend([e.values for e in result.embeddings])
                    else:
                        all_embeddings.extend([e.values for e in result])
                else:
                    logger.error(f"   ❌ Fatal Embedding Error: {err_msg}")
                    raise e
                    
            time.sleep(4) # Cooldown to stay under TPM/RPM limits (Free Tier is 15 RPM = 1 every 4 secs)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        if hasattr(result, 'embeddings'):
            return result.embeddings[0].values
        return result[0].values


class SBertEmbedding:
    """LangChain-compatible wrapper for SentenceTransformer or Remote APIs."""

    def __init__(self, model: Optional[SentenceTransformer] = None):
        self.model = model
        self.remote_impl = None
        
        if config.USE_REMOTE_EMBEDDINGS:
            if config.EMBED_PROVIDER == "google":
                self.remote_impl = GoogleEmbedding(config.EMBED_MODEL_NAME, config.GOOGLE_API_KEY)
            else:
                # Default to OpenAI-compatible (OpenRouter)
                from app.services.llm import get_client
                self.remote_impl = get_client().embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if config.USE_REMOTE_EMBEDDINGS:
            if config.EMBED_PROVIDER == "google":
                return self.remote_impl.embed_documents(texts)
            else:
                all_embeddings = []
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = self.remote_impl.create(
                        model=config.EMBED_MODEL_NAME,
                        input=batch
                    )
                    all_embeddings.extend([data.embedding for data in response.data])
                    time.sleep(0.1)
                return all_embeddings
        
        return self.model.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=True, batch_size=config.EMBED_BATCH_SIZE,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        if config.USE_REMOTE_EMBEDDINGS:
            if config.EMBED_PROVIDER == "google":
                return self.remote_impl.embed_query(text)
            else:
                return self.embed_documents([text])[0]
        
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)



class RetrievalService:
    """Manages FAISS + BM25 indices and hybrid retrieval."""

    def __init__(self):
        self.vectorstore = None
        self.bm25 = None
        self.chunks = None
        self.tokenized_corpus = None
        self.embed_model = None
        self.embed_model = None
        self.expert_rules = []
        self._loaded = False
        self._load_errors = []

    def load(self):
        """Load all indices and models from disk with graceful fallbacks."""
        self._load_errors = []
        
        # 1. Load Embedding Model
        try:
            if config.USE_REMOTE_EMBEDDINGS:
                logger.info(f"Using remote OpenRouter embeddings: {config.EMBED_MODEL_NAME}")
                wrapper = SBertEmbedding()
            else:
                logger.info(f"Loading local embedding model: {config.EMBED_MODEL_NAME}")
                sbert_model = SentenceTransformer(config.EMBED_MODEL_NAME, device='cpu')
                # Cap sequence length so an outlier-long chunk can't trigger O(n^2) OOM during attention.
                sbert_model.max_seq_length = config.EMBED_MAX_SEQ_LENGTH
                wrapper = SBertEmbedding(sbert_model)
        except Exception as e:
            msg = f"Failed to initialize embedding model: {e}"
            logger.error(f"❌ {msg}")
            self._load_errors.append(msg)
            wrapper = None

        # 2. Load FAISS (Dense)
        try:
            if os.path.exists(config.FAISS_INDEX_PATH) and wrapper:
                logger.info(f"Loading FAISS index from {config.FAISS_INDEX_PATH}")
                self.vectorstore = FAISS.load_local(
                    config.FAISS_INDEX_PATH, wrapper,
                    allow_dangerous_deserialization=True
                )
            else:
                logger.warning("⚠️ FAISS index not found or wrapper failed. Dense search disabled.")
        except Exception as e:
            logger.error(f"❌ Error loading FAISS: {e}")
            self._load_errors.append(f"FAISS Error: {e}")

        # 3. Load BM25 & Chunks (Sparse)
        try:
            if os.path.exists(config.BM25_PATH):
                with open(config.BM25_PATH, "rb") as f:
                    self.bm25 = pickle.load(f)
                with open(config.CHUNKS_PATH, "rb") as f:
                    self.chunks = pickle.load(f)
                with open(config.TOKENIZED_CORPUS_PATH, "rb") as f:
                    self.tokenized_corpus = pickle.load(f)
                logger.info(f"✅ BM25 and Chunks loaded ({len(self.chunks)} chunks)")
            else:
                logger.warning("⚠️ BM25/Chunks files not found. Sparse search disabled.")
        except Exception as e:
            logger.error(f"❌ Error loading BM25/Chunks: {e}")
            self._load_errors.append(f"BM25 Error: {e}")

        # 4. Load Expert Rules
        rules_path = os.path.join(config.DATA_DIR, "expert_rules.json")
        if os.path.exists(rules_path):
            try:
                with open(rules_path, "r", encoding="utf-8") as f:
                    self.expert_rules = json.load(f)
                logger.info(f"✅ Loaded {len(self.expert_rules)} expert rule topics.")
            except Exception as e:
                logger.error(f"❌ Error loading expert rules: {e}")

        self._loaded = True
        status = "Ready" if not self._load_errors else "Degraded"
        logger.info(f"=== Retrieval Service {status} ===")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def reload_indices(self) -> None:
        """Reload FAISS + BM25 + chunks + tokenized_corpus from disk without re-initializing
        the embedding model. Called after out-of-process ingest (watch script / /ingest/scan)
        so the running server picks up the new data without a restart."""
        if not self.vectorstore or not getattr(self.vectorstore, "embeddings", None):
            logger.warning("reload_indices called before initial load — falling back to load()")
            return self.load()

        wrapper = self.vectorstore.embeddings
        try:
            self.vectorstore = FAISS.load_local(
                config.FAISS_INDEX_PATH, wrapper, allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"FAISS reload failed: {e}")
            return

        try:
            with open(config.BM25_PATH, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(config.CHUNKS_PATH, "rb") as f:
                self.chunks = pickle.load(f)
            with open(config.TOKENIZED_CORPUS_PATH, "rb") as f:
                self.tokenized_corpus = pickle.load(f)
        except Exception as e:
            logger.error(f"BM25/Chunks reload failed: {e}")
            return

        logger.info(f"♻️ Indices reloaded ({len(self.chunks)} chunks, {self.vectorstore.index.ntotal} vectors)")

    def retrieve(
        self,
        query: str,
        k: int = None,
        method: str = "hybrid",
    ) -> Tuple[List[str], List[Dict], Dict]:
        """Hybrid retrieve: FAISS + BM25 → RRF fusion → cross-encoder rerank → context expansion.

        Returns:
            (context_texts, source_metadata_list, timing_info)
            Each source dict carries retrieval_score (dense), rerank_score (cross-encoder),
            legal_topic (encyclopedia folder taxonomy), and referenced_articles metadata.
        """
        if not self._loaded:
            raise RuntimeError("Retrieval service not loaded. Call load() first.")

        k = k or config.RETRIEVAL_K
        k_dense = config.RETRIEVAL_K_DENSE
        k_sparse = config.RETRIEVAL_K_SPARSE
        k_rerank = config.RETRIEVAL_K_RERANK
        rrf_k = config.RRF_K
        timing: Dict[str, float] = {}

        # --- Dense (FAISS) ---
        dense_indices: List[int] = []
        dense_scores_by_idx: Dict[int, float] = {}
        if method in ("hybrid", "dense") and self.vectorstore:
            t0 = time.time()
            dense_results = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=k_dense
            )
            timing["dense_ms"] = (time.time() - t0) * 1000

            for doc, score in dense_results:
                idx = doc.metadata.get("chunk_index")
                # Old indices stored chunk_index as a string; coerce defensively.
                if isinstance(idx, str):
                    try:
                        idx = int(idx)
                    except ValueError:
                        idx = None
                if not isinstance(idx, int) or idx < 0 or idx >= len(self.chunks):
                    # Legacy fallback: O(N) linear scan when chunk_index is missing/invalid.
                    idx = None
                    doc_src = doc.metadata.get("source") or doc.metadata.get("file_name")
                    for i, c in enumerate(self.chunks):
                        c_src = c.metadata.get("source") or c.metadata.get("file_name")
                        if c.page_content == doc.page_content and c_src == doc_src:
                            idx = i
                            break
                if idx is not None:
                    dense_indices.append(idx)
                    if idx not in dense_scores_by_idx:
                        dense_scores_by_idx[idx] = float(score)

        # --- Sparse (BM25) ---
        sparse_indices: List[int] = []
        if method in ("hybrid", "bm25") and self.bm25:
            t0 = time.time()
            q_tok = preprocess_arabic_for_bm25(query)
            bm25_scores = self.bm25.get_scores(q_tok)
            sparse_indices = np.argsort(bm25_scores)[::-1][:k_sparse].tolist()
            timing["bm25_ms"] = (time.time() - t0) * 1000

        # --- RRF fusion → top-N candidates for rerank ---
        if method == "dense":
            candidate_indices = dense_indices[:k_rerank]
        elif method == "bm25":
            candidate_indices = sparse_indices[:k_rerank]
        else:
            rrf_scores: Dict[int, float] = {}
            for rank, idx in enumerate(dense_indices):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            for rank, idx in enumerate(sparse_indices):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            candidate_indices = [idx for idx, _ in fused[:k_rerank]]

        # --- Cross-encoder rerank: top-N → final top-k ---
        rerank_scores_by_idx: Dict[int, float] = {}
        if candidate_indices:
            t0 = time.time()
            passages = [self._get_text(self.chunks[i]) for i in candidate_indices]
            ranked = reranker_service.rerank(query, passages, top_k=k)
            timing["rerank_ms"] = (time.time() - t0) * 1000
            final_indices: List[int] = []
            for pos, rscore in ranked:
                idx = candidate_indices[pos]
                final_indices.append(idx)
                rerank_scores_by_idx[idx] = float(rscore)
        else:
            final_indices = []

        # --- Expert rules (prepended, no scores) ---
        expert_contexts = self._match_expert_rules(query)

        # --- Build response with context expansion ---
        contexts: List[str] = []
        sources: List[Dict] = []
        seen_indices: set = set()

        for i in final_indices:
            if i in seen_indices or i >= len(self.chunks):
                continue
            chunk = self.chunks[i]
            meta = chunk.metadata
            expanded_text = self._get_text(chunk)
            # Legacy schema used `file_name` instead of `source`; use whichever's present.
            current_source = meta.get("source") or meta.get("file_name")

            # Expand only to immediate same-source neighbors so chunks from different
            # cases/statutes never get stitched together.
            if i > 0:
                prev_src = self.chunks[i - 1].metadata.get("source") or self.chunks[i - 1].metadata.get("file_name")
                if prev_src == current_source:
                    expanded_text = self._get_text(self.chunks[i - 1]) + "\n" + expanded_text
                    seen_indices.add(i - 1)
            if i < len(self.chunks) - 1:
                next_src = self.chunks[i + 1].metadata.get("source") or self.chunks[i + 1].metadata.get("file_name")
                if next_src == current_source:
                    expanded_text = expanded_text + "\n" + self._get_text(self.chunks[i + 1])
                    seen_indices.add(i + 1)

            contexts.append(expanded_text)
            seen_indices.add(i)

            referenced = meta.get("referenced_articles") or []
            # Legacy-key shim: old indices use file_name/category/subcategory/article_number.
            # New indices use filename/doc_type/legal_topic/referenced_articles. Read both.
            filename = meta.get("filename") or meta.get("file_name") or meta.get("display_name", "")
            doc_type = meta.get("doc_type") or meta.get("category", "")
            legal_category = meta.get("legal_category") or meta.get("category", "")
            legal_topic = meta.get("legal_topic", "") or (
                meta.get("subcategory", "") if meta.get("subcategory") not in (None, "", "عام") else ""
            )
            article = (referenced[0] if referenced else None) or (meta.get("article_number") or None)
            sources.append({
                "filename": filename,
                "source": meta.get("source") or meta.get("file_name", ""),
                "doc_type": doc_type,
                "legal_category": legal_category,
                "legal_topic": legal_topic,
                "article": article,
                "referenced_articles": list(referenced),
                "page": meta.get("page"),
                "retrieval_score": round(dense_scores_by_idx.get(i, 0.0), 4),
                "rerank_score": round(rerank_scores_by_idx.get(i, 0.0), 4),
            })

        contexts = expert_contexts + contexts
        timing["total_ms"] = round(sum(timing.values()), 2)
        return contexts, sources, timing

    def retrieve_multi_query(
        self,
        query: str,
        k: int = None,
    ) -> Tuple[List[str], List[Dict], Dict]:
        """Multi-query retrieval with domain-aware reranking.

        Pipeline (compared to retrieve()):
          1. Original query + up to N synonym variants — each runs through
             FAISS + BM25 (no rerank yet) to collect candidate chunk indices.
          2. RRF-fuse across ALL variants (one query's top-1 + another query's
             top-2 contributing more than either query alone).
          3. Optional domain boost: if the query is clearly procedural or
             substantive, add a small constant to RRF scores for chunks whose
             doc_type matches the inferred domain. Pushes the right code on top.
          4. Cross-encoder rerank the top-N candidates against the ORIGINAL
             query (synonyms helped recall; final ranking should reflect the
             user's actual wording).
          5. Same neighbor expansion + source-dict shape as retrieve().

        Falls back to the single-query path when USE_MULTI_QUERY is off or
        when synonym expansion yields no variants (so existing call sites get
        identical behavior unless they explicitly opt in).
        """
        if not self._loaded:
            raise RuntimeError("Retrieval service not loaded. Call load() first.")

        # Fast path: feature disabled, or query has no synonyms to expand —
        # delegate to the single-query retriever.
        if not config.USE_MULTI_QUERY:
            return self.retrieve(query, k=k)

        synonyms = expand_query_synonyms(query, max_extra=2)
        if not synonyms:
            return self.retrieve(query, k=k)

        k = k or config.RETRIEVAL_K
        k_dense = config.RETRIEVAL_K_DENSE
        k_sparse = config.RETRIEVAL_K_SPARSE
        k_rerank = config.RETRIEVAL_K_RERANK
        rrf_k = config.RRF_K
        timing: Dict[str, float] = {}

        # All variants share the same RRF pool. Original query first so its
        # ranks dominate ties (rank+1 = 1 is the highest contribution).
        variants = [query] + synonyms

        # Aggregated dense/sparse rank lists across variants.
        rrf_scores: Dict[int, float] = {}
        dense_scores_by_idx: Dict[int, float] = {}

        # Dense (FAISS) — per-variant retrieval, accumulate ranks.
        if self.vectorstore:
            t0 = time.time()
            for v in variants:
                dense_results = self.vectorstore.similarity_search_with_relevance_scores(v, k=k_dense)
                for rank, (doc, score) in enumerate(dense_results):
                    idx = self._resolve_chunk_index(doc)
                    if idx is None:
                        continue
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
                    # Record the FIRST (highest) dense score we saw for this chunk
                    # across all variants — used as retrieval_score in the sources.
                    if idx not in dense_scores_by_idx:
                        dense_scores_by_idx[idx] = float(score)
            timing["dense_ms"] = (time.time() - t0) * 1000

        # Sparse (BM25) — per-variant retrieval, accumulate ranks.
        if self.bm25:
            t0 = time.time()
            for v in variants:
                q_tok = preprocess_arabic_for_bm25(v)
                bm25_scores = self.bm25.get_scores(q_tok)
                top = np.argsort(bm25_scores)[::-1][:k_sparse].tolist()
                for rank, idx in enumerate(top):
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            timing["bm25_ms"] = (time.time() - t0) * 1000

        # Domain-aware boost: nudge chunks whose doc_type matches the inferred
        # query domain (procedural vs substantive). Small constant so it tips
        # ties without overwhelming RRF signal.
        if config.USE_DOMAIN_BOOST:
            domain = classify_query_domain(query)
            target_doc_types = DOMAIN_TO_DOC_TYPES.get(domain, set())
            if target_doc_types:
                w = config.DOMAIN_BOOST_WEIGHT
                for idx in list(rrf_scores.keys()):
                    if 0 <= idx < len(self.chunks):
                        meta = self.chunks[idx].metadata
                        dt = meta.get("doc_type") or meta.get("category", "")
                        if dt in target_doc_types:
                            rrf_scores[idx] += w
                logger.info(
                    f"🎯 domain={domain}, boosted {target_doc_types} (+{w}) across {len(rrf_scores)} candidates"
                )

        # Top-N from fused scores → rerank against the ORIGINAL query.
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_indices = [idx for idx, _ in fused[:k_rerank]]

        rerank_scores_by_idx: Dict[int, float] = {}
        if candidate_indices:
            t0 = time.time()
            passages = [self._get_text(self.chunks[i]) for i in candidate_indices]
            ranked = reranker_service.rerank(query, passages, top_k=k)
            timing["rerank_ms"] = (time.time() - t0) * 1000
            final_indices: List[int] = []
            for pos, rscore in ranked:
                idx = candidate_indices[pos]
                final_indices.append(idx)
                rerank_scores_by_idx[idx] = float(rscore)
        else:
            final_indices = []

        # Expert rules (matched against the original query).
        expert_contexts = self._match_expert_rules(query)

        # Build response with neighbor expansion (same pattern as retrieve()).
        contexts, sources = self._build_response(
            final_indices, dense_scores_by_idx, rerank_scores_by_idx
        )
        contexts = expert_contexts + contexts
        timing["variants_used"] = len(variants)
        timing["total_ms"] = round(sum(v for k_, v in timing.items() if k_.endswith("_ms")), 2)
        return contexts, sources, timing

    def _resolve_chunk_index(self, doc) -> Optional[int]:
        """Map a FAISS doc back to its position in self.chunks (handles legacy schema)."""
        idx = doc.metadata.get("chunk_index")
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except ValueError:
                idx = None
        if isinstance(idx, int) and 0 <= idx < len(self.chunks):
            return idx
        # Legacy fallback: O(N) linear scan.
        doc_src = doc.metadata.get("source") or doc.metadata.get("file_name")
        for i, c in enumerate(self.chunks):
            c_src = c.metadata.get("source") or c.metadata.get("file_name")
            if c.page_content == doc.page_content and c_src == doc_src:
                return i
        return None

    def _build_response(
        self,
        final_indices: List[int],
        dense_scores_by_idx: Dict[int, float],
        rerank_scores_by_idx: Dict[int, float],
    ) -> Tuple[List[str], List[Dict]]:
        """Build (contexts, sources) from final reranked chunk indices.

        Same neighbor-expansion + source-dict shape as retrieve(). Extracted
        so retrieve_multi_query() doesn't duplicate the assembly logic.
        """
        contexts: List[str] = []
        sources: List[Dict] = []
        seen: set = set()

        for i in final_indices:
            if i in seen or i >= len(self.chunks):
                continue
            chunk = self.chunks[i]
            meta = chunk.metadata
            expanded_text = self._get_text(chunk)
            current_source = meta.get("source") or meta.get("file_name")

            if i > 0:
                prev_src = self.chunks[i - 1].metadata.get("source") or self.chunks[i - 1].metadata.get("file_name")
                if prev_src == current_source:
                    expanded_text = self._get_text(self.chunks[i - 1]) + "\n" + expanded_text
                    seen.add(i - 1)
            if i < len(self.chunks) - 1:
                next_src = self.chunks[i + 1].metadata.get("source") or self.chunks[i + 1].metadata.get("file_name")
                if next_src == current_source:
                    expanded_text = expanded_text + "\n" + self._get_text(self.chunks[i + 1])
                    seen.add(i + 1)

            contexts.append(expanded_text)
            seen.add(i)

            referenced = meta.get("referenced_articles") or []
            filename = meta.get("filename") or meta.get("file_name") or meta.get("display_name", "")
            doc_type = meta.get("doc_type") or meta.get("category", "")
            legal_category = meta.get("legal_category") or meta.get("category", "")
            legal_topic = meta.get("legal_topic", "") or (
                meta.get("subcategory", "") if meta.get("subcategory") not in (None, "", "عام") else ""
            )
            article = (referenced[0] if referenced else None) or (meta.get("article_number") or None)
            sources.append({
                "filename": filename,
                "source": meta.get("source") or meta.get("file_name", ""),
                "doc_type": doc_type,
                "legal_category": legal_category,
                "legal_topic": legal_topic,
                "article": article,
                "referenced_articles": list(referenced),
                "page": meta.get("page"),
                "retrieval_score": round(dense_scores_by_idx.get(i, 0.0), 4),
                "rerank_score": round(rerank_scores_by_idx.get(i, 0.0), 4),
            })
        return contexts, sources

    def _match_expert_rules(self, query: str) -> List[str]:
        """Match query against loaded expert rules."""
        results = []
        q_lower = query.lower()
        for topic in self.expert_rules:
            if any(kw.lower() in q_lower for kw in topic.get("keywords", [])):
                logger.info(f"🎯 Expert rule match: {topic.get('topic')}")
                rule_text = f"### [قاعدة خبيرة: {topic.get('topic')}] ###\n"
                for rule in topic.get("rules", []):
                    rule_text += f"- الحالة: {rule['condition']}\n"
                    rule_text += f"  المواد: {', '.join(rule['articles'])}\n"
                    rule_text += f"  العقوبة: {rule['penalty']}\n"
                    if "note" in rule: rule_text += f"  ملاحظة: {rule['note']}\n"
                    if "context" in rule: rule_text += f"  سياق: {rule['context']}\n"
                if "expert_advice" in topic:
                    rule_text += f"\nنصيحة الخبير: {topic['expert_advice']}"
                results.append(rule_text)
        return results

    def _get_text(self, obj):
        """Extract text content from various chunk types."""
        if hasattr(obj, 'page_content'): return obj.page_content
        if hasattr(obj, 'content'): return obj.content
        return str(obj)


# ── Global singleton ──
retrieval_service = RetrievalService()
