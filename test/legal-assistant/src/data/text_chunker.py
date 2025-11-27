"""
Text Chunker - Split legal documents into meaningful chunks

Usage:
    chunker = LegalTextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.process_extracted_documents("data/processed/extracted_text")
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import logging
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Class representing a text chunk"""
    text: str
    chunk_id: str
    source_doc: str
    chunk_type: str
    metadata: Dict
    text_length: int


class LegalTextChunker:
    """
    تقسيم النصوص القانونية بطريقة ذكية
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, 
                 output_dir: str = "data/processed/chunks"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Chunk output directory: {self.output_dir}")
    
    def chunk_legal_code(self, text: str, doc_name: str) -> List[TextChunk]:
        """
        تقسيم قانون العقوبات حسب المواد
        """
        logger.info(f"📚 Chunking legal code: {doc_name}")
        
        chunks = []
        article_pattern = r'((?:المادة|مادة)\s*\d+[أ-ي]*(?:\s*(?:مكرر|ثانياً|ثالثاً))*)'
        
        parts = re.split(article_pattern, text)
        
        current_article_num = None
        current_text = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if re.match(article_pattern, part):
                if current_text and current_article_num:
                    chunk = TextChunk(
                        text=current_text,
                        chunk_id=f"{doc_name}_article_{len(chunks)}",
                        source_doc=doc_name,
                        chunk_type="article",
                        metadata={"article_number": current_article_num},
                        text_length=len(current_text)
                    )
                    chunks.append(chunk)
                
                current_article_num = part
                current_text = part + "\n"
            else:
                current_text += part + "\n"
        
        if current_text and current_article_num:
            chunk = TextChunk(
                text=current_text,
                chunk_id=f"{doc_name}_article_{len(chunks)}",
                source_doc=doc_name,
                chunk_type="article",
                metadata={"article_number": current_article_num},
                text_length=len(current_text)
            )
            chunks.append(chunk)
        
        logger.info(f"✅ Created {len(chunks)} article chunks")
        return chunks
    
    def chunk_by_paragraphs(self, text: str, doc_name: str) -> List[TextChunk]:
        """
        تقسيم حسب الفقرات مع overlap
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_count = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunk = TextChunk(
                    text=current_chunk,
                    chunk_id=f"{doc_name}_chunk_{chunk_count}",
                    source_doc=doc_name,
                    chunk_type="paragraph",
                    metadata={"chunk_number": chunk_count},
                    text_length=len(current_chunk)
                )
                chunks.append(chunk)
                
                current_chunk = current_chunk[-self.chunk_overlap:] + "\n\n" + para
                chunk_count += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        if current_chunk:
            chunk = TextChunk(
                text=current_chunk,
                chunk_id=f"{doc_name}_chunk_{chunk_count}",
                source_doc=doc_name,
                chunk_type="paragraph",
                metadata={"chunk_number": chunk_count},
                text_length=len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_extracted_documents(self, input_dir: str) -> List[TextChunk]:
        """
        معالجة جميع المستندات المستخرجة
        """
        input_dir = Path(input_dir)
        json_files = list(input_dir.glob("*_extracted.json"))
        
        if not json_files:
            logger.warning(f"⚠️  No extracted JSON files found in {input_dir}")
            return []
        
        logger.info(f"📂 Found {len(json_files)} documents to chunk")
        
        all_chunks = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                text = data.get('text', '')
                if not text:
                    logger.warning(f"⚠️  Empty text in {json_file.name}")
                    continue
                
                doc_name = data['metadata']['filename'].replace('.pdf', '')
                
                if 'penal_code' in doc_name.lower() or 'قانون' in text[:500]:
                    chunks = self.chunk_legal_code(text, doc_name)
                else:
                    chunks = self.chunk_by_paragraphs(text, doc_name)
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"❌ Error chunking {json_file.name}: {e}")
        
        self._save_chunks(all_chunks)
        
        logger.info(f"✅ Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _save_chunks(self, chunks: List[TextChunk]):
        """Save chunks to JSON file"""
        chunks_data = [asdict(chunk) for chunk in chunks]
        
        output_file = self.output_dir / "all_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        stats = {
            "total_chunks": len(chunks),
            "by_type": {},
            "by_source": {},
            "avg_chunk_length": sum(c.text_length for c in chunks) / len(chunks) if chunks else 0
        }
        
        for chunk in chunks:
            stats["by_type"][chunk.chunk_type] = stats["by_type"].get(chunk.chunk_type, 0) + 1
            stats["by_source"][chunk.source_doc] = stats["by_source"].get(chunk.source_doc, 0) + 1
        
        stats_file = self.output_dir / "chunking_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Chunks saved to: {output_file}")


if __name__ == "__main__":
    chunker = LegalTextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.process_extracted_documents("data/processed/extracted_text")
    print(f"\n✅ Created {len(chunks)} chunks")