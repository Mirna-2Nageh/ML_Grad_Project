"""
Data Validator - Validate quality of extracted and chunked data

Usage:
    validator = DataValidator()
    validator.validate_extraction("data/processed/extracted_text")
    validator.validate_chunks("data/processed/chunks")
"""

import json
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    التحقق من جودة البيانات المستخرجة
    """
    
    def __init__(self):
        self.min_text_length = 100
        self.min_chunk_size = 50
    
    def validate_extraction(self, extraction_dir: str) -> Dict:
        """
        التحقق من جودة النصوص المستخرجة
        """
        logger.info(f"🔍 Validating extracted texts in: {extraction_dir}")
        
        extraction_dir = Path(extraction_dir)
        json_files = list(extraction_dir.glob("*_extracted.json"))
        
        results = {
            "total_files": len(json_files),
            "valid_files": 0,
            "invalid_files": 0,
            "issues": []
        }
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text = data.get('text', '')
            metadata = data.get('metadata', {})
            
            if not metadata.get('success'):
                results["invalid_files"] += 1
                results["issues"].append({
                    "file": json_file.name,
                    "issue": "Extraction failed"
                })
            elif len(text) < self.min_text_length:
                results["invalid_files"] += 1
                results["issues"].append({
                    "file": json_file.name,
                    "issue": f"Text too short ({len(text)} chars)"
                })
            else:
                results["valid_files"] += 1
        
        logger.info(f"✅ Validation complete: {results['valid_files']}/{results['total_files']} valid")
        
        return results
    
    def validate_chunks(self, chunks_dir: str) -> Dict:
        """
        التحقق من جودة الـ chunks
        """
        logger.info(f"🔍 Validating chunks in: {chunks_dir}")
        
        chunks_file = Path(chunks_dir) / "all_chunks.json"
        
        if not chunks_file.exists():
            logger.error(f"❌ Chunks file not found: {chunks_file}")
            return {"error": "File not found"}
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        results = {
            "total_chunks": len(chunks),
            "valid_chunks": 0,
            "invalid_chunks": 0,
            "issues": []
        }
        
        for chunk in chunks:
            text_length = chunk.get('text_length', 0)
            
            if text_length < self.min_chunk_size:
                results["invalid_chunks"] += 1
                results["issues"].append({
                    "chunk_id": chunk.get('chunk_id'),
                    "issue": f"Chunk too short ({text_length} chars)"
                })
            else:
                results["valid_chunks"] += 1
        
        logger.info(f"✅ Validation complete: {results['valid_chunks']}/{results['total_chunks']} valid")
        
        return results


if __name__ == "__main__":
    validator = DataValidator()
    
    # Validate extraction
    extraction_results = validator.validate_extraction("data/processed/extracted_text")
    print("\n📊 Extraction Validation:")
    print(json.dumps(extraction_results, ensure_ascii=False, indent=2))
    
    # Validate chunks
    chunks_results = validator.validate_chunks("data/processed/chunks")
    print("\n📊 Chunks Validation:")
    print(json.dumps(chunks_results, ensure_ascii=False, indent=2))