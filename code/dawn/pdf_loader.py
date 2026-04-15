from pathlib import Path

from pypdf import PdfReader


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = " ".join(text.split())
        if cleaned:
            pages.append({"page": page_number, "text": cleaned})

    if not pages:
        raise ValueError(f"Aucun texte exploitable trouve dans {pdf_path}.")

    return pages
