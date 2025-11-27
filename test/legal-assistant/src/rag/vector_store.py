"""
Vector Store - Create and manage vector database for RAG

Usage:
    vector_store = LegalVectorStore(db_path="data/vector_db/chroma_db")
    vector_store.add_documents_from_chunks("data/processed/chunks/all_chunks.json")
    results = vector_store.query("ما هي عقوبة السرقة؟", n_results=5)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalVectorStore:
    """
    إدارة قاعدة البيانات الشعاعية للمستندات القانونية
    """
    
    def __init__(self, 
                 db_path: str = "data/vector_db/chroma_db",
                 collection_name: str = "egyptian_legal_docs",
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔄 Initializing ChromaDB at: {self.db_path}")
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Egyptian legal documents for RAG"}
        )
        
        logger.info(f"🔄 Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        logger.info("✅ Vector store initialized")
        
        self.collection_name = collection_name
    
    def add_documents_from_chunks(self, chunks_file: str, batch_size: int = 50):
        """
        إضافة المستندات من ملف الـ chunks
        """
        logger.info(f"📚 Loading chunks from: {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"📄 Found {len(chunks)} chunks to process")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to vector DB"):
            batch = chunks[i:i + batch_size]
            
            texts = [chunk['text'] for chunk in batch]
            ids = [chunk['chunk_id'] for chunk in batch]
            metadatas = [
                {
                    "source_doc": chunk['source_doc'],
                    "chunk_type": chunk['chunk_type'],
                    "text_length": chunk['text_length'],
                    **chunk['metadata']
                }
                for chunk in batch
            ]
            
            embeddings = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
        
        logger.info(f"✅ Added {len(chunks)} documents to vector store")
        logger.info(f"📊 Total documents in collection: {self.collection.count()}")
    
    def query(self, query_text: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> Dict:
        """
        البحث في قاعدة البيانات
        """
        query_embedding = self.embedder.encode([query_text])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_dict
        )
        
        return {
            "documents": results['documents'][0],
            "metadatas": results['metadatas'][0],
            "distances": results['distances'][0],
            "ids": results['ids'][0]
        }
    
    def get_statistics(self) -> Dict:
        """احصائيات عن قاعدة البيانات"""
        total_docs = self.collection.count()
        
        sample = self.collection.get(limit=min(100, total_docs))
        
        stats = {
            "total_documents": total_docs,
            "collection_name": self.collection_name,
            "document_types": {},
            "source_documents": {}
        }
        
        for metadata in sample['metadatas']:
            chunk_type = metadata.get('chunk_type', 'unknown')
            source_doc = metadata.get('source_doc', 'unknown')
            
            stats["document_types"][chunk_type] = stats["document_types"].get(chunk_type, 0) + 1
            stats["source_documents"][source_doc] = stats["source_documents"].get(source_doc, 0) + 1
        
        return stats


if __name__ == "__main__":
    vector_store = LegalVectorStore(db_path="data/vector_db/chroma_db")
    
    vector_store.add_documents_from_chunks("data/processed/chunks/all_chunks.json")
    
    stats = vector_store.get_statistics()
    print("\n📊 Vector Store Statistics:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    results = vector_store.query("ما هي عقوبة السرقة؟", n_results=3)
    print(f"\n🔍 Query Results:")
    for i, doc in enumerate(results['documents'], 1):
        print(f"\n{i}. {doc[:200]}...")