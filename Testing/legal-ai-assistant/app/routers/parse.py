"""One-shot document parsing endpoint.

Lets clients extract text from a file (.txt/.pdf/.docx) or a pasted string
without running the full LLM pipeline. Powers the Streamlit "upload to
auto-populate" UX on the weakness / defense / summarize tabs — users can
preview and edit the parsed text before submitting for analysis.

No LLM call, no retrieval, no validation — purely a parse + clean pass."""
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.models import ParseResponse, ErrorResponse
from app.services.upload_helper import parse_uploaded_file, parse_uploaded_text

router = APIRouter()


@router.post(
    "/parse",
    response_model=ParseResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Parse a document or string to extracted text",
    description=(
        "Accepts a file upload (.txt / .pdf / .docx) OR a pasted-text string, "
        "runs the same parsing + Arabic cleaning pipeline used by /qa/upload, "
        "and returns the extracted text. No LLM, no retrieval — fast.\n\n"
        "Send as `multipart/form-data` with either `file` or `text`. If both, "
        "`file` wins. .doc (legacy binary) is not supported — please save as "
        ".docx or .pdf first."
    ),
)
async def parse_document(
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
):
    if file is not None and file.filename:
        cleaned, fname, ctype, warnings = await parse_uploaded_file(file)
    elif text:
        cleaned, fname, ctype, warnings = parse_uploaded_text(text)
    else:
        raise HTTPException(
            status_code=400, detail="Provide either a `file` or a `text` field.",
        )

    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail=" / ".join(warnings) or "تعذّر قراءة المرفق.",
        )

    return ParseResponse(
        text=cleaned,
        filename=fname,
        content_type=ctype,
        char_count=len(cleaned),
        warnings=warnings,
    )
