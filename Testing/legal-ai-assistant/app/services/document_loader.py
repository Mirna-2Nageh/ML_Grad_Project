"""Multi-format document loaders. PDFs prefer poppler's `pdftotext` (cleaner Arabic layout)
with PyPDF2 as fallback when poppler isn't installed."""
import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def load_text(path: str) -> str:
    """Read a text file, trying common Arabic encodings before giving up."""
    for encoding in ("utf-8", "utf-8-sig", "cp1256", "iso-8859-6"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def load_pdf(path: str) -> str:
    """Extract text from a PDF. Tries `pdftotext -layout` first, falls back to PyPDF2."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", path, "-"],
            capture_output=True, text=True, timeout=120, check=True,
        )
        if result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.debug(f"pdftotext unavailable/failed for {os.path.basename(path)}: {type(e).__name__}")

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        logger.warning(f"PDF extraction failed for {path}: {e}")
        return ""


def load_docx(path: str) -> str:
    """Extract text from a DOCX file via docx2txt."""
    try:
        import docx2txt
        return docx2txt.process(path) or ""
    except Exception as e:
        logger.warning(f"DOCX extraction failed for {path}: {e}")
        return ""


def load_any(path: str) -> str:
    """Dispatch by file extension. Returns '' for unsupported types."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return load_text(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    return ""
