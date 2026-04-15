def chunk_page_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks


def build_chunks(pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    chunk_records: list[dict] = []

    for page in pages:
        page_chunks = chunk_page_text(page["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_id, chunk_text in enumerate(page_chunks, start=1):
            chunk_records.append(
                {
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )

    if not chunk_records:
        raise ValueError("Le chunking n'a produit aucun segment.")

    return chunk_records
