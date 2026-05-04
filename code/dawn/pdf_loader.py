from pathlib import Path
import shutil
import unicodedata

from pypdf import PdfReader


class KnowledgeLoadError(ValueError):
    pass


def _ocr_dependencies_status() -> dict[str, bool]:
    try:
        import fitz  # type: ignore
    except Exception:
        fitz = None

    try:
        import pytesseract  # type: ignore
    except Exception:
        pytesseract = None

    try:
        from PIL import Image  # noqa: F401
    except Exception:
        Image = None  # type: ignore

    return {
        "fitz": fitz is not None,
        "pytesseract": pytesseract is not None,
        "tesseract_binary": shutil.which("tesseract") is not None,
    }

def _normalize_filename(filename: str) -> str:
    normalized = unicodedata.normalize("NFKD", filename.lower())
    return normalized.encode("ascii", "ignore").decode("ascii")


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return normalized.encode("ascii", "ignore").decode("ascii")


def _detect_document_language(pdf_path: Path, sample_text: str) -> str:
    normalized_name = _normalize_filename(pdf_path.name)
    if any(marker in normalized_name for marker in ("-fre", "_fre", "francais")):
        return "fr"
    if any(marker in normalized_name for marker in ("-eng", "_eng", "english")):
        return "en"

    normalized_text = _normalize_text(sample_text)
    french_markers = {" les ", " des ", " une ", " dans ", " pour ", " avec ", " enfants ", " prise en charge "}
    english_markers = {" the ", " and ", " with ", " children ", " treatment ", " management ", " guideline "}
    french_score = sum(marker in f" {normalized_text} " for marker in french_markers)
    english_score = sum(marker in f" {normalized_text} " for marker in english_markers)

    if french_score > english_score:
        return "fr"
    if english_score > french_score:
        return "en"
    return "unknown"


def _build_loaded_document_info(pdf_path: Path, pages: list[dict]) -> dict:
    return {
        "source_name": pdf_path.name,
        "source_path": str(pdf_path),
        "relative_path": str(pdf_path),
        "storage_folder": str(pdf_path.parent),
        "document_language": pages[0].get("document_language", "unknown") if pages else "unknown",
        "ocr_used": any(page.get("extraction_method") == "ocr" for page in pages),
        "pages_loaded": len(pages),
    }


def _relative_storage_folder(pdf_path: Path, knowledge_root: Path) -> str:
    try:
        return str(pdf_path.parent.relative_to(knowledge_root))
    except ValueError:
        return str(pdf_path.parent)


def _build_text_document_pages(
    document_path: Path,
    storage_folder: str,
) -> list[dict]:
    text = document_path.read_text(encoding="utf-8")
    cleaned = " ".join(text.split())
    if not cleaned:
        raise KnowledgeLoadError(f"Aucun texte exploitable trouve dans {document_path}.")

    # Detect language for text files - markdown files are assumed French for medical context
    sample_text = cleaned[:500]
    detected_lang = _detect_document_language(document_path, sample_text)
    # Override to French for markdown files as they're likely local medical protocols
    if document_path.suffix.lower() == ".md":
        detected_lang = "fr"

    return [
        {
            "page": 1,
            "text": cleaned,
            "source_path": str(document_path),
            "source_name": document_path.name,
            "storage_folder": storage_folder,
            "extraction_method": "text-file",
            "document_language": detected_lang,
        }
    ]


def _ocr_pdf_pages(
    pdf_path: Path,
    storage_folder: str,
    ocr_language: str,
) -> list[dict]:
    dependency_status = _ocr_dependencies_status()
    if not all(dependency_status.values()):
        missing = [name for name, available in dependency_status.items() if not available]
        raise KnowledgeLoadError(
            "Aucun texte exploitable trouve et OCR indisponible pour "
            f"{pdf_path}. Dependances manquantes: {', '.join(missing)}."
        )

    import fitz  # type: ignore
    import pytesseract  # type: ignore
    from PIL import Image

    document = fitz.open(str(pdf_path))
    pages: list[dict] = []

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        text = pytesseract.image_to_string(image, lang=ocr_language) or ""
        cleaned = " ".join(text.split())
        if cleaned:
            pages.append(
                {
                    "page": page_index + 1,
                    "text": cleaned,
                    "source_path": str(pdf_path),
                    "source_name": pdf_path.name,
                    "storage_folder": storage_folder,
                    "extraction_method": "ocr",
                }
            )

    if not pages:
        raise KnowledgeLoadError(f"Aucun texte exploitable trouve dans {pdf_path}, meme apres OCR.")

    return pages


def load_pdf_pages(
    pdf_path: Path,
    knowledge_root: Path | None = None,
    enable_ocr_fallback: bool = True,
    ocr_language: str = "fra",
) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []
    storage_folder = _relative_storage_folder(pdf_path, knowledge_root) if knowledge_root else str(pdf_path.parent)
    sampled_text_parts: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = " ".join(text.split())
        if cleaned:
            if len(sampled_text_parts) < 3:
                sampled_text_parts.append(cleaned[:1500])
            pages.append(
                {
                    "page": page_number,
                    "text": cleaned,
                    "source_path": str(pdf_path),
                    "source_name": pdf_path.name,
                    "storage_folder": storage_folder,
                    "extraction_method": "text",
                }
            )

    if not pages:
        if enable_ocr_fallback:
            pages = _ocr_pdf_pages(
                pdf_path=pdf_path,
                storage_folder=storage_folder,
                ocr_language=ocr_language,
            )
            sampled_text_parts = [page["text"][:1500] for page in pages[:3]]
        else:
            raise KnowledgeLoadError(f"Aucun texte exploitable trouve dans {pdf_path}.")

    detected_language = _detect_document_language(pdf_path, " ".join(sampled_text_parts))
    for page in pages:
        page["document_language"] = detected_language

    return pages


def load_knowledge_pages(
    knowledge_path: Path,
    enable_ocr_fallback: bool = True,
    ocr_language: str = "fra",
) -> tuple[list[dict], list[dict], list[dict]]:
    if knowledge_path.is_file():
        suffix = knowledge_path.suffix.lower()
        if suffix == ".pdf":
            pages = load_pdf_pages(
                knowledge_path,
                knowledge_root=knowledge_path.parent,
                enable_ocr_fallback=enable_ocr_fallback,
                ocr_language=ocr_language,
            )
        elif suffix in {".txt", ".md"}:
            pages = _build_text_document_pages(knowledge_path, storage_folder=".")
        else:
            raise ValueError(f"Le fichier {knowledge_path} n'est pas supporte.")
        document_info = _build_loaded_document_info(knowledge_path, pages)
        document_info["relative_path"] = knowledge_path.name
        document_info["storage_folder"] = "."
        return pages, [document_info], []

    if not knowledge_path.exists():
        raise FileNotFoundError(f"Chemin introuvable : {knowledge_path}")

    supported_files = sorted(
        [
            path
            for path in knowledge_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
        ]
    )
    if not supported_files:
        raise ValueError(f"Aucun document supporte trouve dans {knowledge_path}.")

    all_pages: list[dict] = []
    loaded_documents: list[dict] = []
    skipped_files: list[dict] = []
    for document_file in supported_files:
        try:
            if document_file.suffix.lower() == ".pdf":
                pages = load_pdf_pages(
                    document_file,
                    knowledge_root=knowledge_path,
                    enable_ocr_fallback=enable_ocr_fallback,
                    ocr_language=ocr_language,
                )
            else:
                pages = _build_text_document_pages(
                    document_file,
                    storage_folder=_relative_storage_folder(document_file, knowledge_path),
                )
            all_pages.extend(pages)
            document_info = _build_loaded_document_info(document_file, pages)
            document_info["relative_path"] = str(document_file.relative_to(knowledge_path))
            document_info["storage_folder"] = _relative_storage_folder(document_file, knowledge_path)
            loaded_documents.append(document_info)
        except KnowledgeLoadError as exc:
            skipped_files.append(
                {
                    "source_name": document_file.name,
                    "source_path": str(document_file),
                    "relative_path": str(document_file.relative_to(knowledge_path)),
                    "storage_folder": _relative_storage_folder(document_file, knowledge_path),
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
