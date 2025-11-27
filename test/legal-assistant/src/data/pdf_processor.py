"""
PDF Processor - Extract text from legal documents
Handles both text-based and scanned PDFs (optimized for large PDFs)

Usage:
    processor = PDFProcessor(output_dir="data/processed/extracted_text")
    results = processor.process_directory("data/raw/pdfs/court_cases")
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

# PDF Libraries
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image

# OCR Libraries
try:
    import pytesseract
    from pdf2image import convert_from_path, pdfinfo_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  OCR libraries not installed. Run: pip install pytesseract pdf2image")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    PDF Processor with OCR support for scanned documents.
    Optimized for large PDFs by processing pages one by one.
    """
    
    def __init__(self, output_dir: str = "data/processed/extracted_text"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def check_if_scanned(self, pdf_path: str) -> bool:
        """Check if a PDF is scanned (needs OCR)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(min(3, len(pdf_reader.pages))):
                    text = pdf_reader.pages[page_num].extract_text()
                    if text and len(text.strip()) > 50:
                        return False
            return True
        except Exception as e:
            logger.warning(f"Error checking PDF type: {e}")
            return True
    
    def extract_text_native(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text from regular PDFs"""
        logger.info(f"📄 Extracting text from: {Path(pdf_path).name}")
        full_text = ""
        metadata = {"filename": Path(pdf_path).name, "method": "native_extraction", "pages": 0, "success": False}
        try:
            doc = fitz.open(pdf_path)
            metadata["pages"] = len(doc)
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    full_text += f"\n--- صفحة {page_num} ---\n{text}"
            doc.close()
            full_text = self._clean_text(full_text)
            metadata["success"] = True
            metadata["text_length"] = len(full_text)
            return full_text, metadata
        except Exception as e:
            logger.error(f"❌ Error extracting text: {e}")
            metadata["error"] = str(e)
            return "", metadata
    
    def extract_text_ocr(self, pdf_path: str, language: str = 'ara') -> Tuple[str, Dict]:
        """Extract text from scanned PDFs using OCR (memory-efficient)"""
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR libraries not installed!")
        
        logger.info(f"🔍 Running OCR on: {Path(pdf_path).name}")
        full_text = ""
        metadata = {"filename": Path(pdf_path).name, "method": "ocr", "pages": 0, "success": False}
        
        try:
            info = pdfinfo_from_path(pdf_path)
            total_pages = info["Pages"]
            metadata["pages"] = total_pages
            
            for page_num in range(1, total_pages + 1):
                # Convert single page at lower DPI and grayscale
                image = convert_from_path(
                    pdf_path,
                    dpi=150,
                    grayscale=True,
                    first_page=page_num,
                    last_page=page_num
                )[0]
                
                text = pytesseract.image_to_string(image, lang=language, config='--psm 6')
                if text.strip():
                    full_text += f"\n--- صفحة {page_num} ---\n{text}"
            
            full_text = self._clean_text(full_text)
            metadata["success"] = True
            metadata["text_length"] = len(full_text)
            
            return full_text, metadata
        except Exception as e:
            logger.error(f"❌ OCR Error: {e}")
            metadata["error"] = str(e)
            return "", metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        import re
        # Remove unwanted characters
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\d\.\,\:\;\?\!\-\(\)\[\]\"\'\/]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def process_single_pdf(self, pdf_path: str, force_ocr: bool = False) -> Dict:
        """Process a single PDF"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        needs_ocr = force_ocr or self.check_if_scanned(str(pdf_path))
        if needs_ocr:
            if not OCR_AVAILABLE:
                logger.warning("⚠️ OCR needed but not available. Using native extraction...")
                text, metadata = self.extract_text_native(str(pdf_path))
            else:
                text, metadata = self.extract_text_ocr(str(pdf_path))
        else:
            text, metadata = self.extract_text_native(str(pdf_path))
        
        result = {"text": text, "metadata": metadata, "source_file": str(pdf_path)}
        
        # Save result
        output_file = self.output_dir / f"{pdf_path.stem}_extracted.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved to: {output_file}")
        return result
    
    def process_directory(self, input_dir: str, pattern: str = "*.pdf", force_ocr: bool = False) -> List[Dict]:
        """Process all PDFs in a directory"""
        input_dir = Path(input_dir)
        pdf_files = list(input_dir.glob(pattern))
        if not pdf_files:
            logger.warning(f"⚠️ No PDF files found in {input_dir}")
            return []
        
        logger.info(f"📚 Found {len(pdf_files)} PDF files")
        results = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                result = self.process_single_pdf(str(pdf_file), force_ocr=force_ocr)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
        
        # Save summary
        summary = {
            "total_documents": len(results),
            "successful": sum(1 for r in results if r['metadata']['success']),
            "failed": sum(1 for r in results if not r['metadata']['success']),
            "total_pages": sum(r['metadata']['pages'] for r in results),
            "documents": [r['metadata'] for r in results]
        }
        
        summary_file = self.output_dir / "extraction_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Summary: {summary['successful']}/{summary['total_documents']} successful")
        return results


if __name__ == "__main__":
    processor = PDFProcessor(output_dir="data/processed/extracted_text")
    results = processor.process_directory(
        input_dir="data/raw/pdfs/court_cases",
        pattern="*.pdf",
        force_ocr=False
    )
    print(f"\n✅ Processed {len(results)} documents")
