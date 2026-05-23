"""Shared helper for the three upload modes (/qa/upload, /chat/attach, /ingest).

Centralises the file-parsing + sanity-checking so the three endpoints stay
consistent (same allowed extensions, same min-content threshold, same
character cap) and the document_loader stays free of HTTP concerns.
"""
import os
import logging
import tempfile
from typing import Tuple, Optional

from fastapi import UploadFile

from app.services.document_loader import load_text, load_pdf, load_docx
from app.services.preprocessing import clean_arabic_legal_text

logger = logging.getLogger(__name__)


# Single source of truth for what we accept across all upload endpoints.
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

# Hard cap on extracted text per upload. 100k chars ≈ 25-30 pages of dense
# Arabic legal text — plenty for a single document while bounding memory and
# context-budget cost. Larger uploads should be split or ingested instead.
MAX_DOC_CHARS = 100_000

# Below this, the file is likely empty / OCR'd badly / wrong format.
MIN_DOC_CHARS = 30


def _content_type_for(ext: str) -> str:
    """Map extension to a stable content_type string for the API response.
    Keeps frontend display logic simple ('.text' for pasted strings, '.txt'/'.pdf'/'.docx' for files)."""
    return ext or ".text"


def parse_uploaded_text(
    text: Optional[str], filename: Optional[str] = None,
) -> Tuple[str, str, str, list]:
    """Parse a pasted-text input. Returns (cleaned_text, filename, content_type, warnings)."""
    warnings: list = []
    if not text:
        return "", filename or "", ".text", warnings
    raw = text
    if len(raw) > MAX_DOC_CHARS:
        warnings.append(
            f"تم اقتطاع النص المُرفق إلى {MAX_DOC_CHARS} حرف "
            f"(الأصلي: {len(raw)} حرف)."
        )
        raw = raw[:MAX_DOC_CHARS]
    cleaned = clean_arabic_legal_text(raw)
    if len(cleaned) < MIN_DOC_CHARS:
        warnings.append(
            f"النص المُرفق قصير جداً ({len(cleaned)} حرفاً) ولم يتم استخدامه. "
            "يرجى إدخال نص أطول."
        )
        return "", filename or "pasted-text", ".text", warnings
    return cleaned, filename or "pasted-text", ".text", warnings


async def parse_uploaded_file(upload: UploadFile) -> Tuple[str, str, str, list]:
    """Parse an UploadFile. Returns (cleaned_text, filename, content_type, warnings).

    Cleaned text is empty (and warnings non-empty) if the file is unsupported,
    too short, or fails to extract. Callers should check `cleaned_text` before
    using it instead of inspecting warnings.
    """
    warnings: list = []
    filename = upload.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".doc":
        warnings.append(
            "الملفات بصيغة .doc (Word 97-2003 binary) غير مدعومة مباشرة. "
            "يرجى حفظ الملف بصيغة .docx أو .pdf ثم إعادة المحاولة."
        )
        return "", filename, ext, warnings

    if ext not in ALLOWED_EXTENSIONS:
        warnings.append(
            f"الصيغة {ext or '(غير معروفة)'} غير مدعومة. "
            f"الصيغ المدعومة: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
        return "", filename, ext, warnings

    # Stream upload to a temp file, parse, then unlink. Done this way so the
    # poppler `pdftotext` subprocess can read by path (it can't take stdin
    # in the layout-preserving mode we use).
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            chunk = await upload.read()
            tmp.write(chunk)
            tmp_path = tmp.name
        if ext == ".txt":
            raw = load_text(tmp_path)
        elif ext == ".pdf":
            raw = load_pdf(tmp_path)
        elif ext == ".docx":
            raw = load_docx(tmp_path)
        else:
            raw = ""
    except Exception as e:
        logger.warning(f"upload parse failed for {filename}: {e}")
        warnings.append(f"فشل في قراءة الملف: {str(e)[:120]}")
        return "", filename, ext, warnings
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not raw or len(raw.strip()) < MIN_DOC_CHARS:
        warnings.append(
            f"تعذّر استخراج نص كافٍ من الملف "
            f"({len(raw or '')} حرفاً). قد يكون الملف صورة ممسوحة ضوئياً تحتاج OCR، "
            "أو فارغاً، أو محمياً بكلمة مرور."
        )
        return "", filename, ext, warnings

    if len(raw) > MAX_DOC_CHARS:
        warnings.append(
            f"تم اقتطاع المستند إلى {MAX_DOC_CHARS} حرف "
            f"(الأصلي: {len(raw)} حرف)."
        )
        raw = raw[:MAX_DOC_CHARS]

    cleaned = clean_arabic_legal_text(raw)
    if len(cleaned) < MIN_DOC_CHARS:
        warnings.append(
            f"النص بعد التنظيف قصير جداً ({len(cleaned)} حرفاً) — ربما يكون الملف غير قابل للقراءة."
        )
        return "", filename, ext, warnings

    return cleaned, filename, ext, warnings


def format_doc_for_context(text: str, filename: str, content_type: str) -> str:
    """Wrap a single uploaded doc as a delimited context block.

    Same marker scheme as Session.format_attachments() so a one-shot
    /qa/upload looks identical to the LLM as a session-attached doc.
    """
    return (
        "[المستند المرفق]\n"
        f"--- مستند: {filename or 'مستند'} ({content_type or '.text'}) ---\n"
        f"{text}\n"
        "[نهاية المستند المرفق]"
    )
