"""PDF -> pages -> sentence-aware chunks -> embeddings -> vector DB."""
import re

import pymupdf  # PyMuPDF

from retriever import invalidate_index


def extract_pages(pdf_bytes: bytes):
    """Yield (page_number, text) for every page. Page numbers start at 1."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            yield page_no, text


def chunk_text(text: str, size: int = 900, overlap_sentences: int = 1):
    """Group sentences into chunks of about `size` characters.

    Chunks end on sentence boundaries (better than cutting mid-sentence) and
    repeat the last sentence of the previous chunk so context isn't lost.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    sentences = [re.sub(r"\s+", " ", p).strip() for p in parts]
    sentences = [s for s in sentences if s]

    chunks, current, length = [], [], 0
    for s in sentences:
        if current and length + len(s) > size:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            length = sum(len(x) + 1 for x in current)
        current.append(s)
        length += len(s) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def ingest_pdf(pdf_bytes: bytes, filename: str, collection, embedder) -> int:
    """Chunk a PDF, embed the chunks and store them. Returns the chunk count."""
    ids, docs, metas = [], [], []
    for page_no, text in extract_pages(pdf_bytes):
        for j, chunk in enumerate(chunk_text(text)):
            ids.append(f"{filename}-p{page_no}-c{j}")
            docs.append(chunk)
            metas.append({"file": filename, "page": page_no})

    if not docs:
        return 0  # e.g. a scanned PDF with no extractable text

    embeddings = embedder.encode(
        docs, normalize_embeddings=True, show_progress_bar=False
    ).tolist()
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    invalidate_index(collection.name)  # keyword index must be rebuilt
    return len(docs)
