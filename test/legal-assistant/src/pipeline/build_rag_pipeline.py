"""
Main RAG Pipeline - Orchestrates the entire data processing and RAG setup

Usage:
    python src/pipeline/build_rag_pipeline.py
"""

import sys
from pathlib import Path
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pdf_processor import PDFProcessor
from src.data.text_chunker import LegalTextChunker
from src.data.data_validator import DataValidator
from src.rag.vector_store import LegalVectorStore
from src.rag.retrieval import LegalRAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


def main():
    """
    تشغيل Pipeline كامل لمعالجة البيانات وإنشاء RAG system
    """
    
    console.print("\n[bold green]🚀 Starting Legal Assistant RAG Pipeline[/bold green]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Step 1: Extract text from PDFs
        task1 = progress.add_task("[cyan]Step 1/5: Extracting text from PDFs...", total=None)
        
        processor = PDFProcessor(output_dir="data/processed/extracted_text")
        
        # Process all PDFs from legal_documents
        console.print("[yellow]📄 Processing all legal documents...[/yellow]")
        all_results = processor.process_directory(
            input_dir="data/raw/pdfs/legal_documents",
            pattern="*.pdf",
            force_ocr=False  # سيتحقق تلقائياً من الحاجة للـ OCR
        )
        
        total_docs = len(all_results)
        console.print(f"[green]✅ Extracted {total_docs} documents[/green]\n")
        progress.update(task1, completed=True)
        
        # Step 2: Validate extraction
        task2 = progress.add_task("[cyan]Step 2/5: Validating extracted data...", total=None)
        
        validator = DataValidator()
        validation_results = validator.validate_extraction("data/processed/extracted_text")
        
        console.print(f"[green]✅ Validation: {validation_results['valid_files']}/{validation_results['total_files']} valid[/green]\n")
        progress.update(task2, completed=True)
        
        # Step 3: Chunk documents
        task3 = progress.add_task("[cyan]Step 3/5: Chunking documents...", total=None)
        
        chunker = LegalTextChunker(
            chunk_size=500,
            chunk_overlap=100,
            output_dir="data/processed/chunks"
        )
        
        chunks = chunker.process_extracted_documents("data/processed/extracted_text")
        
        console.print(f"[green]✅ Created {len(chunks)} chunks[/green]\n")
        progress.update(task3, completed=True)
        
        # Step 4: Build vector database
        task4 = progress.add_task("[cyan]Step 4/5: Building vector database...", total=None)
        
        vector_store = LegalVectorStore(
            db_path="data/vector_db/chroma_db",
            collection_name="egyptian_legal_docs"
        )
        
        vector_store.add_documents_from_chunks("data/processed/chunks/all_chunks.json")
        
        stats = vector_store.get_statistics()
        console.print(f"[green]✅ Vector DB ready with {stats['total_documents']} documents[/green]\n")
        progress.update(task4, completed=True)
        
        # Step 5: Test RAG system
        task5 = progress.add_task("[cyan]Step 5/5: Testing RAG system...", total=None)
        
        try:
            rag = LegalRAGRetriever(vector_store)
            
            test_question = "ما هي عقوبة السرقة في القانون المصري؟"
            result = rag.query(test_question)
            
            console.print(f"[green]✅ RAG system working![/green]")
            console.print(f"\n[cyan]Test Query:[/cyan] {test_question}")
            console.print(f"[cyan]Answer:[/cyan] {result['answer'][:200]}...\n")
        except ValueError as e:
            console.print(f"[yellow]⚠️  RAG test skipped: {e}[/yellow]")
            console.print("[yellow]Set OPENAI_API_KEY environment variable to enable RAG queries[/yellow]\n")
        
        progress.update(task5, completed=True)
    
    console.print("\n[bold green]🎉 Pipeline Complete![/bold green]\n")
    console.print("[cyan]Summary:[/cyan]")
    console.print(f"  📄 Documents processed: {total_docs}")
    console.print(f"  📦 Chunks created: {len(chunks)}")
    console.print(f"  🗄️  Vector DB size: {stats['total_documents']} documents")
    console.print(f"\n[cyan]Next steps:[/cyan]")
    console.print("  1. Test queries: python -c 'from src.rag.retrieval import *; ...'")
    console.print("  2. Check data/processed/ for processed files")
    console.print("  3. Query vector DB: data/vector_db/chroma_db\n")


if __name__ == "__main__":
    main()
