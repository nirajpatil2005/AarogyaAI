"""
Medical Report Processor — Handles upload, text extraction, and indexing
of user medical reports (PDF, TXT, DOCX).

All reports are stored locally. PII stays on device.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "user_reports"
REPORTS_DIR.mkdir(exist_ok=True)

REPORTS_INDEX = REPORTS_DIR / "reports_index.json"


def _load_index() -> list[dict]:
    """Load the reports index."""
    if REPORTS_INDEX.exists():
        with open(REPORTS_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_index(reports: list[dict]):
    """Save the reports index."""
    with open(REPORTS_INDEX, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    try:
        from PyPDF2 import PdfReader
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text.strip())
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[PDF extraction error: {e}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        import io
        doc = Document(io.BytesIO(file_bytes))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX extraction error: {e}]"


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain text file."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def process_report(filename: str, file_bytes: bytes) -> dict:
    """
    Process an uploaded medical report:
    1. Extract text based on file type
    2. Store locally
    3. Add to reports index
    Returns the report metadata.
    """
    ext = Path(filename).suffix.lower()

    # Extract text
    if ext == ".pdf":
        extracted_text = extract_text_from_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        extracted_text = extract_text_from_docx(file_bytes)
    elif ext in (".txt", ".text"):
        extracted_text = extract_text_from_txt(file_bytes)
    else:
        # Try treating as plain text
        extracted_text = extract_text_from_txt(file_bytes)

    # Create report record
    report_id = f"report_{uuid.uuid4().hex[:8]}"
    report = {
        "id": report_id,
        "filename": filename,
        "file_type": ext,
        "uploaded_at": datetime.now().isoformat(),
        "extracted_text": extracted_text,
        "char_count": len(extracted_text),
        "word_count": len(extracted_text.split()),
    }

    # Save the raw file locally
    raw_file = REPORTS_DIR / f"{report_id}{ext}"
    with open(raw_file, "wb") as f:
        f.write(file_bytes)

    # Update index
    reports = _load_index()
    reports.append(report)
    _save_index(reports)

    return {
        "id": report_id,
        "filename": filename,
        "char_count": report["char_count"],
        "word_count": report["word_count"],
        "status": "processed",
    }


def get_all_reports() -> list[dict]:
    """Get metadata for all uploaded reports."""
    reports = _load_index()
    return [
        {
            "id": r["id"],
            "filename": r["filename"],
            "file_type": r["file_type"],
            "uploaded_at": r["uploaded_at"],
            "word_count": r["word_count"],
        }
        for r in reports
    ]


def get_report_text(report_id: str) -> Optional[str]:
    """Get the extracted text of a specific report."""
    reports = _load_index()
    for r in reports:
        if r["id"] == report_id:
            return r["extracted_text"]
    return None


def delete_report(report_id: str) -> bool:
    """Delete a report from the index and filesystem."""
    reports = _load_index()
    updated = [r for r in reports if r["id"] != report_id]
    if len(updated) == len(reports):
        return False

    _save_index(updated)

    # Remove raw file
    for f in REPORTS_DIR.glob(f"{report_id}.*"):
        f.unlink(missing_ok=True)

    return True
