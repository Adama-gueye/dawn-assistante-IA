from pathlib import Path

from pypdf import PdfReader


class KnowledgeLoadError(ValueError):
    pass


def _build_loaded_document_info(pdf_path: Path, pages: list[dict]) -> dict:
    return {
        "source_name": pdf_path.name,
        "source_path": str(pdf_path),
        "pages_loaded": len(pages),
    }


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = " ".join(text.split())
        if cleaned:
            pages.append(
                {
                    "page": page_number,
                    "text": cleaned,
                    "source_path": str(pdf_path),
                    "source_name": pdf_path.name,
                }
            )

    if not pages:
        raise KnowledgeLoadError(f"Aucun texte exploitable trouve dans {pdf_path}.")

    return pages


def load_knowledge_pages(knowledge_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    if knowledge_path.is_file():
        if knowledge_path.suffix.lower() != ".pdf":
            raise ValueError(f"Le fichier {knowledge_path} n'est pas un PDF.")
        pages = load_pdf_pages(knowledge_path)
        return pages, [_build_loaded_document_info(knowledge_path, pages)], []

    if not knowledge_path.exists():
        raise FileNotFoundError(f"Chemin introuvable : {knowledge_path}")

    pdf_files = sorted(knowledge_path.rglob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"Aucun PDF trouve dans {knowledge_path}.")

    all_pages: list[dict] = []
    loaded_documents: list[dict] = []
    skipped_files: list[dict] = []
    for pdf_file in pdf_files:
        try:
            pages = load_pdf_pages(pdf_file)
            all_pages.extend(pages)
            loaded_documents.append(_build_loaded_document_info(pdf_file, pages))
        except KnowledgeLoadError as exc:
            skipped_files.append(
                {
                    "source_name": pdf_file.name,
                    "source_path": str(pdf_file),
                    "reason": str(exc),
                }
            )

    if not all_pages:
        joined_errors = (
            "\n".join(item["reason"] for item in skipped_files) if skipped_files else str(knowledge_path)
        )
        raise KnowledgeLoadError(
            "Aucun texte exploitable n'a pu etre extrait du corpus.\n"
            f"{joined_errors}"
        )

    return all_pages, loaded_documents, skipped_files
