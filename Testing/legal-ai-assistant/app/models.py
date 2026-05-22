"""
Pydantic request/response schemas — these define the API contract.
The .NET backend team should implement their HTTP client based on these models.
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────

class QARequest(BaseModel):
    """Legal question-answering request."""
    question: str = Field(..., description="Legal question in Arabic", min_length=5, max_length=1000)
    k: int = Field(default=7, description="Number of context documents to retrieve", ge=1, le=20)
    prompt_style: str = Field(default="restrictive", description="Prompt style: 'standard' or 'restrictive'")

    model_config = {"json_schema_extra": {
        "examples": [{"question": "ما هي عقوبة السرقة في القانون المصري؟", "k": 7}]
    }}


class SummarizeRequest(BaseModel):
    """Legal text summarization request."""
    text: str = Field(..., description="Legal text to summarize in Arabic", min_length=50, max_length=15000)

    model_config = {"json_schema_extra": {
        "examples": [{"text": "المادة الأولى: يعاقب بالحبس كل من ارتكب جريمة السرقة..."}]
    }}


class WeaknessRequest(BaseModel):
    """Case weakness detection request."""
    case_facts: str = Field(..., description="Case facts text in Arabic", min_length=20, max_length=10000)
    evidence: str = Field(default="", description="Evidence items (cameras, medical reports, witness statements, prior history). Each will be analyzed explicitly.", max_length=15000)
    defendant_statement: str = Field(default="", description="The defendant's own narrative / defenses raised (e.g., self-defense, mutual assault, denial)", max_length=5000)

    model_config = {"json_schema_extra": {
        "examples": [{"case_facts": "المتهم متهم بالسرقة بالإكراه ليلاً. تم القبض عليه بناءً على بلاغ مجهول."}]
    }}


class DefenseRequest(BaseModel):
    """Defense memorandum generation request."""
    case_facts: str = Field(..., description="Case facts in Arabic", min_length=20, max_length=10000)
    weaknesses: str = Field(default="", description="Previously identified weaknesses (optional)", max_length=5000)
    evidence: str = Field(default="", description="Evidence items (cameras, medical reports, witness statements, prior history). Each will be analyzed explicitly.", max_length=15000)
    defendant_statement: str = Field(default="", description="The defendant's own narrative / defenses raised (e.g., self-defense, mutual assault, denial)", max_length=5000)

    model_config = {"json_schema_extra": {
        "examples": [{"case_facts": "المتهم متهم بالسرقة.", "weaknesses": "عدم وجود شهود عيان"}]
    }}


class ForensicRequest(BaseModel):
    """Forensic-consistency analysis request: cross-references facts + evidence against the law."""
    case_facts: str = Field(..., description="Case facts / incident or police report (Arabic)", min_length=20, max_length=10000)
    evidence: str = Field(default="", description="Additional evidence text (medical records, testimony, forensic reports). Will be sourced from the per-case evidence store once uploads land.", max_length=15000)

    model_config = {"json_schema_extra": {
        "examples": [{"case_facts": "ضُبط المتهم ليلاً وبحوزته سلاح.", "evidence": "التقرير الطبي: لا توجد إصابات على المجني عليه."}]
    }}


# ──────────────────────────────────────────────
# Source Attribution & Confidence
# ──────────────────────────────────────────────

class SourceInfo(BaseModel):
    """Metadata about a retrieved source document. Field names are stable wire contract."""
    filename: str = Field(default="", description="Source filename")
    source: str = Field(default="", description="Source file path")
    doc_type: str = Field(default="", description="Document type (penal_code, cassation_ruling, ...)")
    legal_category: str = Field(default="", description="Legal topic category (back-compat field)")
    legal_topic: str = Field(default="", description="Encyclopedia topic name from folder taxonomy")
    article: Optional[str] = Field(default=None, description="Primary article number cited in this chunk")
    referenced_articles: List[str] = Field(default_factory=list, description="All article numbers referenced in this chunk")
    page: Optional[int] = Field(default=None, description="Page number when extractable")
    retrieval_score: float = Field(default=0.0, description="Dense (FAISS) relevance score, 0-1")
    rerank_score: float = Field(default=0.0, description="Cross-encoder rerank score, 0-1 (sigmoid-bound)")


class ConfidenceFactors(BaseModel):
    """Per-component breakdown of inputs to the confidence score."""
    rerank_signal: float = Field(default=0.0, description="Mean rerank score over final top-k")
    source_count: int = Field(default=0, description="Number of source chunks returned")
    article_validation: str = Field(default="not_applicable", description="passed | failed | not_applicable")
    topic_match: bool = Field(default=False, description="Whether any source's legal_topic appears in the question")


# ──────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────

class QAResponse(BaseModel):
    """Q&A response with answer, confidence, sources, and metrics."""
    answer: str = Field(..., description="Generated answer in Arabic")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence 0-1")
    confidence_factors: ConfidenceFactors = Field(default_factory=ConfidenceFactors)
    sources: List[SourceInfo] = Field(default_factory=list, description="Retrieved source documents")
    warnings: List[str] = Field(default_factory=list, description="User-facing warnings (low confidence, uncited articles, ...)")
    conflicts_detected: bool = Field(default=False, description="Whether retrieved sources contradict each other (Phase 2)")
    latency_ms: float = Field(..., description="Total response time in milliseconds")
    retrieval_ms: float = Field(default=0, description="Retrieval time in milliseconds")
    model: str = Field(default="", description="LLM model used")


class SummarizeResponse(BaseModel):
    """Summarization response."""
    summary: str = Field(..., description="Generated summary in Arabic")
    input_length: int = Field(..., description="Input text length in characters")
    latency_ms: float = Field(..., description="Total response time in milliseconds")
    model: str = Field(default="", description="LLM model used")


class WeaknessResponse(BaseModel):
    """Weakness detection response."""
    analysis: str = Field(..., description="Weakness analysis in Arabic")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_factors: ConfidenceFactors = Field(default_factory=ConfidenceFactors)
    sources: List[SourceInfo] = Field(default_factory=list, description="Legal references used")
    warnings: List[str] = Field(default_factory=list)
    conflicts_detected: bool = Field(default=False)
    latency_ms: float = Field(..., description="Total response time in milliseconds")
    model: str = Field(default="", description="LLM model used")


class DefenseResponse(BaseModel):
    """Defense memorandum response."""
    memorandum: str = Field(..., description="Defense memorandum in formal Arabic")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_factors: ConfidenceFactors = Field(default_factory=ConfidenceFactors)
    sources: List[SourceInfo] = Field(default_factory=list, description="Legal references used")
    warnings: List[str] = Field(default_factory=list)
    conflicts_detected: bool = Field(default=False)
    latency_ms: float = Field(..., description="Total response time in milliseconds")
    model: str = Field(default="", description="LLM model used")
    self_check_revisions: int = Field(default=0, description="Number of agentic self-check revision passes applied to the memo")


class ForensicResponse(BaseModel):
    """Forensic-consistency analysis response."""
    analysis: str = Field(..., description="Consistency analysis in Arabic (contradictions, evidence mismatches, unmet legal elements, gaps)")
    conflicts_detected: bool = Field(default=False, description="True if any contradiction / evidence mismatch / unmet element was found")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_factors: ConfidenceFactors = Field(default_factory=ConfidenceFactors)
    sources: List[SourceInfo] = Field(default_factory=list, description="Legal references used")
    warnings: List[str] = Field(default_factory=list)
    latency_ms: float = Field(..., description="Total response time in milliseconds")
    model: str = Field(default="", description="LLM model used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok")
    vectors: int = Field(default=0, description="Number of vectors in FAISS index")
    chunks: int = Field(default=0, description="Number of text chunks loaded")
    model: str = Field(default="", description="Configured LLM model")
    embedding_model: str = Field(default="", description="Configured embedding model")
    reranker_loaded: bool = Field(default=False, description="Whether the cross-encoder reranker is active")
    reranker_model: str = Field(default="", description="Reranker model name (empty if disabled/failed)")
    load_errors: List[str] = Field(default_factory=list, description="Any errors during service initialization")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: str = Field(default="", description="Detailed error information")


# ──────────────────────────────────────────────
# Chat Models (with session & history)
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Chat request with session ID for conversation continuity."""
    message: str = Field(..., description="User message in Arabic", min_length=1)
    session_id: str = Field(default="default", description="Session ID for conversation continuity")
    k: int = Field(default=7, description="Number of context documents to retrieve", ge=1, le=20)

    model_config = {"json_schema_extra": {
        "examples": [{"message": "ما عقوبة السرقة بالإكراه؟", "session_id": "user-123"}]
    }}


class ChatResponse(BaseModel):
    """Chat response with answer, confidence, sources, and session info."""
    answer: str = Field(..., description="Generated answer in Arabic")
    session_id: str = Field(..., description="Session ID")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_factors: ConfidenceFactors = Field(default_factory=ConfidenceFactors)
    sources: List[SourceInfo] = Field(default_factory=list, description="Retrieved source documents")
    warnings: List[str] = Field(default_factory=list)
    conflicts_detected: bool = Field(default=False)
    latency_ms: float = Field(..., description="Total response time in milliseconds")
    turn_count: int = Field(default=0, description="Number of conversation turns in this session")
    was_compacted: bool = Field(default=False, description="Whether history was compacted this turn")
    model: str = Field(default="", description="LLM model used")


class SessionInfoResponse(BaseModel):
    """Session information response."""
    session_id: str
    turn_count: int = 0
    message_count: int = 0
    has_summary: bool = False
    created_at: float = 0
    last_active: float = 0


# ──────────────────────────────────────────────
# Ingestion Models
# ──────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response from data ingestion pipeline."""
    status: str = Field(default="ok", description="Ingestion status")
    files_processed: int = Field(default=0, description="Number of files processed")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    vectors_added: int = Field(default=0, description="Number of vectors added to FAISS")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    duration_s: float = Field(default=0, description="Total processing time in seconds")
