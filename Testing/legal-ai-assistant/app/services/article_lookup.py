"""Article-number direct lookup over the existing chunks.pkl.

Background — why this exists
─────────────────────────────
Eval(v3/v4/v5) repeatedly showed the LLM citing real Egyptian-law article
numbers that are NOT in the chunks retrieved by FAISS+BM25, even though those
same articles ARE present elsewhere in the 47k-chunk index. Example:
  • سؤال "شروط التوقيف الاحتياطي" → LLM cites مادة 300; retrieval surfaces 7
    chunks, none containing 300; validator flags hallucination.
  • Inspection: 14 chunks in chunks.pkl actually reference article 300.
  • So the article exists in our data — retrieval just didn't surface it.

This service closes that gap with a deterministic article-number index built
ONCE at startup from `chunk.metadata["referenced_articles"]`. The router uses
it as a *post-validation rescue*: when validate_evidence flags missing
articles, look them up; if found, append the matching chunks to the context
and re-prompt. The LLM's "memory citation" was right — we just hadn't shown
it the relevant chunk.

This costs:
  • One pass over chunks.pkl at startup (~1s for 47k chunks).
  • One extra LLM call ONLY when validation fails AND the missing article
    exists in our dataset — i.e., exactly when it has the best chance of
    converting a hallucination into a grounded answer.
"""
import logging
from typing import Dict, List, Optional, Tuple

import config
from app.services.preprocessing import normalize_arabic_indic_digits

logger = logging.getLogger(__name__)


class ArticleLookupService:
    """Singleton: maps Egyptian-law article numbers → chunks that reference them.

    Loads from the SAME chunks.pkl the retrieval service uses (no separate
    pickle, no duplication), so a single re-load picks up any ingest changes.
    Build is O(N chunks) — cheap once at startup.
    """

    def __init__(self):
        # {article_no_as_str: [chunk_index, ...]} — sorted by length-of-source-text
        # so the FIRST chunk for an article is the longest/most-complete one.
        self._index: Dict[str, List[int]] = {}
        self._chunks = None  # reference to the retrieval service's chunks list
        self._loaded = False

    def load(self, chunks) -> None:
        """Build the article-number → chunk-indices map.

        `chunks` is the same list the RetrievalService loaded from disk; we
        keep a reference rather than copying so memory cost stays at O(map),
        not O(map + chunks).
        """
        self._chunks = chunks
        idx: Dict[str, List[int]] = {}

        # First pass: collect raw indices per article.
        for i, c in enumerate(chunks):
            for art in (c.metadata.get("referenced_articles") or []):
                # Normalize to Western digits so legacy Arabic-Indic keys (e.g. '۲۳۹')
                # match queries, which arrive Western-normalized from validate_evidence.
                key = normalize_arabic_indic_digits(str(art).strip())
                if not key:
                    continue
                idx.setdefault(key, []).append(i)

        # Sort each article's chunk list by a quality score so callers get the
        # most-useful passage first. Limits to top-5 per article so one
        # over-referenced article (e.g. مادة 32 with 712 chunks) can't blow up
        # the context budget.
        #
        # Quality score components (higher is better):
        #   • doc_type tier: 'penal_code' / 'criminal_procedure' (the actual law
        #     text) → tier 3; 'cassation_ruling' / 'criminal_law_reference' /
        #     'legal_rules_collection' → tier 2; everything else (encyclopedia
        #     including ICC/international material) → tier 1.
        #   • article-density: how prominently the article number appears in the
        #     chunk text vs. as a passing cross-reference. Approximated by how
        #     many DISTINCT articles the chunk references — fewer articles mean
        #     the chunk is more focused on the queried article specifically.
        #   • text length as a tiebreaker (longer passages tend to be more
        #     complete article definitions).
        #
        # Why this matters: eval(v6) revealed that lookup(87) was returning Rome
        # Statute / ICC chunks where "87" appears as a passing reference. Those
        # chunks confuse the rescue LLM into inventing neighbouring articles
        # (88, 89, 90, 91, 92) by analogy. Prefer chunks from the actual code.
        _DOC_TYPE_TIER = {
            'penal_code': 3,
            'criminal_procedure': 3,
            'cassation_ruling': 2,
            'criminal_law_reference': 2,
            'legal_rules_collection': 2,
            'cassation_encyclopedia': 1,
            'legal_reference': 1,
            'forensic_medicine': 1,
        }

        def quality_score(cidx: int) -> tuple:
            c = chunks[cidx]
            md = c.metadata
            dt = md.get('doc_type') or md.get('category', '')
            tier = _DOC_TYPE_TIER.get(dt, 1)
            # Fewer distinct articles in the chunk = more focused on this one
            ref_count = len(md.get('referenced_articles') or [])
            focus = -ref_count  # negate so smaller is better
            text = getattr(c, "page_content", None) or getattr(c, "content", "") or str(c)
            return (tier, focus, len(text))

        max_per_article = 5
        for k, vs in idx.items():
            vs.sort(key=quality_score, reverse=True)
            idx[k] = vs[:max_per_article]

        self._index = idx
        self._loaded = True
        logger.info(
            f"✅ ArticleLookupService ready — {len(idx)} distinct articles, "
            f"top-5 chunks each, indexed from {len(chunks)} chunks"
        )

    def reload(self, chunks) -> None:
        """Rebuild after an ingest. Same API as load()."""
        self.load(chunks)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def lookup(self, article_no: str, max_chunks: int = 2) -> List[Tuple[int, str]]:
        """Return [(chunk_index, text), ...] for chunks referencing `article_no`.

        Caller-controlled limit (`max_chunks`) keeps the rescue context tight —
        2 chunks is enough to satisfy validation while leaving room for the
        original retrieved context.
        """
        if not self._loaded or not self._chunks:
            return []
        key = normalize_arabic_indic_digits(str(article_no).strip())
        if not key or key not in self._index:
            return []
        out: List[Tuple[int, str]] = []
        for cidx in self._index[key][:max_chunks]:
            if 0 <= cidx < len(self._chunks):
                chunk = self._chunks[cidx]
                text = getattr(chunk, "page_content", None) or getattr(chunk, "content", "") or str(chunk)
                out.append((cidx, text))
        return out

    def lookup_many(self, article_numbers: List[str], max_chunks_per_article: int = 2) -> Tuple[List[str], List[str]]:
        """Look up multiple article numbers in one call.

        Returns (found_chunk_texts, articles_that_existed). The second item
        lets the caller distinguish 'hallucinated number that's nowhere in
        our data' (true hallucination) from 'memory citation we can verify
        and rescue' (recoverable).
        """
        if not self._loaded:
            return [], []
        all_texts: List[str] = []
        found_articles: List[str] = []
        seen_chunk_indices: set = set()
        for art in article_numbers:
            results = self.lookup(art, max_chunks=max_chunks_per_article)
            if not results:
                continue
            found_articles.append(art)
            for cidx, text in results:
                if cidx in seen_chunk_indices:
                    continue
                seen_chunk_indices.add(cidx)
                all_texts.append(text)
        return all_texts, found_articles


# ── Global singleton ──
article_lookup_service = ArticleLookupService()
