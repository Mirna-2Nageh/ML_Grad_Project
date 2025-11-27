"""
RAG (Retrieval-Augmented Generation) Module
"""

from .vector_store import LegalVectorStore
from .retrieval import LegalRAGRetriever

__all__ = ['LegalVectorStore', 'LegalRAGRetriever']