"""Ingest page-text files into the local ChromaDB collection.

Run as a script: `python -m backend.utils.ingestion [--clear]`.
"""

import os
import glob
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from backend.services.rag_tools import (
    CHROMA_COLLECTION,
    CHROMA_DB_PATH,
    get_collection,
    sync_embed_texts,
)

load_dotenv(override=True)

DEFAULT_PAGES_DIR = str(
    Path(__file__).resolve().parents[2] / "pages"
)
DEFAULT_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "1600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "200"))


def get_source_category(filename: str) -> dict:
    """Categorize HugoBank pages based on filename patterns."""
    name = os.path.basename(filename).replace(".txt", "")

    if name.startswith("index"):
        category = "Home"
        subcategory = "HugoBank Overview"
    elif name in ("our-story", "team"):
        category = "About HugoBank"
        subcategory = name.replace("-", " ").title()
    elif name == "frequently-asked-questions":
        category = "FAQ"
        subcategory = "Frequently Asked Questions"
    elif name in ("waitlist", "join-hugo"):
        category = "Waitlist & Onboarding"
        subcategory = name.replace("-", " ").title()
    elif name == "careers":
        category = "Careers"
        subcategory = "Career Opportunities at HugoBank"
    elif name == "contact-us":
        category = "Contact"
        subcategory = "Contact HugoBank"
    elif name in ("fraud-awareness", "whistle-blowing"):
        category = "Security & Compliance"
        subcategory = name.replace("-", " ").title()
    elif name in ("privacy-policy", "terms-and-conditions"):
        category = "Policies"
        subcategory = name.replace("-", " ").title()
    else:
        category = "General"
        subcategory = name.replace("-", " ").replace("_", " ").title()

    return {
        "category": category,
        "subcategory": subcategory,
        "source_file": name,
    }

def _flush(collection, ids, vectors, metadatas, documents):
    if not ids:
        return
    collection.add(
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas,
        documents=documents,
    )
    print(f"  ✓ Upserted batch of {len(ids)} vectors")

def ingest_text_file(file_path: str):
    print(f"📄 Ingesting {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print(f"⚠️ Skipping empty file: {file_path}")
        return

    source_info = get_source_category(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    print(
        f"  → Split into {len(chunks)} chunks "
        f"(size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_CHUNK_OVERLAP})"
    )
    collection = get_collection()

    ids: list[str] = []
    vectors: list[list[float]] = []
    metadatas: list[dict] = []
    documents: list[str] = []

    chunk_vectors = sync_embed_texts(chunks)

    for i, (chunk, vector) in enumerate(zip(chunks, chunk_vectors)):
        doc_id = str(uuid.uuid4())
        metadata = {
            "text": chunk,
            "category": source_info["category"],
            "subcategory": source_info["subcategory"],
            "source_file": source_info["source_file"],
            "chunk_index": i,
            "total_chunks": len(chunks)
        }

        ids.append(doc_id)
        vectors.append(vector)
        metadatas.append(metadata)
        documents.append(chunk)

        if len(ids) >= 50:
            _flush(collection, ids, vectors, metadatas, documents)
            ids, vectors, metadatas, documents = [], [], [], []

    if ids:
        _flush(collection, ids, vectors, metadatas, documents)

    print(f"✅ Completed: {file_path} ({len(chunks)} chunks)")

def ingest_pdf_file(file_path: str, category: str = "Product Toolkit", subcategory: str = "Product Toolkit 2026"):
    print(f"📄 Ingesting PDF {file_path}...")

    from pypdf import PdfReader

    reader = PdfReader(file_path)
    source_name = os.path.basename(file_path).rsplit(".", 1)[0]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    collection = get_collection()
    total_chunks_ingested = 0

    page_chunks: list[tuple[int, str]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            print(f"  ⚠️ Failed to extract page {page_num}: {e}")
            continue
        if not page_text.strip():
            continue
        for chunk in text_splitter.split_text(page_text):
            if chunk.strip():
                page_chunks.append((page_num, chunk))

    if not page_chunks:
        print(f"⚠️ No extractable text in {file_path}")
        return

    print(f"  → {len(reader.pages)} pages → {len(page_chunks)} chunks "
          f"(size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_CHUNK_OVERLAP})")

    chunk_texts = [c for _, c in page_chunks]
    BATCH = 50
    for start in range(0, len(chunk_texts), BATCH):
        batch_pairs = page_chunks[start:start + BATCH]
        batch_texts = [c for _, c in batch_pairs]
        vectors = sync_embed_texts(batch_texts)
        ids, metadatas, documents, embs = [], [], [], []
        for i, ((page_num, chunk), vec) in enumerate(zip(batch_pairs, vectors)):
            ids.append(str(uuid.uuid4()))
            metadatas.append({
                "text": chunk,
                "category": category,
                "subcategory": subcategory,
                "source_file": source_name,
                "page_number": page_num,
                "chunk_index": start + i,
                "total_chunks": len(page_chunks),
            })
            documents.append(chunk)
            embs.append(vec)
        _flush(collection, ids, embs, metadatas, documents)
        total_chunks_ingested += len(ids)

    print(f"✅ Completed: {file_path} ({total_chunks_ingested} chunks)")


def ingest_pdf_ocr(file_path: str, category: str, subcategory: str, lang: str = "eng", dpi: int = 300):
    """OCR-based PDF ingestion using Tesseract via pdf2image + pytesseract.

    `lang` is a Tesseract language code (e.g. "eng", "urd", "urd+eng").
    """
    print(f"📄 OCR-ingesting PDF {file_path} (lang={lang}, dpi={dpi})...")

    import pytesseract
    from pdf2image import convert_from_path

    source_name = os.path.basename(file_path).rsplit(".", 1)[0]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", "۔ ", ". ", " ", ""],
    )

    images = convert_from_path(file_path, dpi=dpi)
    print(f"  → Rasterized {len(images)} pages at {dpi} DPI")

    page_chunks: list[tuple[int, str]] = []
    for page_num, img in enumerate(images, start=1):
        try:
            page_text = pytesseract.image_to_string(img, lang=lang)
        except Exception as e:
            print(f"  ⚠️ OCR failed page {page_num}: {e}")
            continue
        if not page_text.strip():
            print(f"  ⚠️ Page {page_num} produced no text")
            continue
        for chunk in text_splitter.split_text(page_text):
            if chunk.strip():
                page_chunks.append((page_num, chunk))

    if not page_chunks:
        print(f"⚠️ No OCR text extracted from {file_path}")
        return

    print(f"  → {len(page_chunks)} chunks (size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_CHUNK_OVERLAP})")

    collection = get_collection()
    total = 0
    BATCH = 50
    for start in range(0, len(page_chunks), BATCH):
        batch = page_chunks[start:start + BATCH]
        texts = [c for _, c in batch]
        vectors = sync_embed_texts(texts)
        ids, metadatas, documents, embs = [], [], [], []
        for i, ((page_num, chunk), vec) in enumerate(zip(batch, vectors)):
            ids.append(str(uuid.uuid4()))
            metadatas.append({
                "text": chunk,
                "category": category,
                "subcategory": subcategory,
                "source_file": source_name,
                "page_number": page_num,
                "chunk_index": start + i,
                "total_chunks": len(page_chunks),
                "ocr_lang": lang,
            })
            documents.append(chunk)
            embs.append(vec)
        _flush(collection, ids, embs, metadatas, documents)
        total += len(ids)

    print(f"✅ Completed OCR: {file_path} ({total} chunks)")


def ingest_all_pages(pages_dir: str = DEFAULT_PAGES_DIR):
    txt_files = glob.glob(os.path.join(pages_dir, "*.txt"))

    if not txt_files:
        print(f"❌ No .txt files found in {pages_dir}")
        return

    print(
        f"\n🚀 Starting ingestion of {len(txt_files)} files into "
        f"Chroma collection '{CHROMA_COLLECTION}' at {CHROMA_DB_PATH}...\n"
    )

    for file_path in sorted(txt_files):
        try:
            ingest_text_file(file_path)
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    collection = get_collection()
    count = collection.count()
    print(
        f"\n✅ Ingestion complete! Collection '{CHROMA_COLLECTION}' now has "
        f"{count} vectors."
    )

def clear_collection():
    print(f"🗑️ Clearing Chroma collection '{CHROMA_COLLECTION}'...")
    import chromadb
    from chromadb.config import Settings

    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception as e:
        print(f"  (collection did not exist or was already empty: {e})")
    print(f"✅ Collection '{CHROMA_COLLECTION}' cleared")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        clear_collection()
        ingest_all_pages()
    elif len(sys.argv) > 1 and sys.argv[1] == "--ocr":
        if len(sys.argv) < 3:
            print("Usage: python -m backend.utils.ingestion --ocr <path> [lang] [category] [subcategory]")
            sys.exit(1)
        pdf_path = sys.argv[2]
        lang = sys.argv[3] if len(sys.argv) > 3 else "eng"
        cat = sys.argv[4] if len(sys.argv) > 4 else "Islamic Banking"
        sub = sys.argv[5] if len(sys.argv) > 5 else "Islamic Banking Booklet"
        ingest_pdf_ocr(pdf_path, category=cat, subcategory=sub, lang=lang)
        c = get_collection().count()
        print(f"\n✅ Collection '{CHROMA_COLLECTION}' now has {c} vectors.")
    elif len(sys.argv) > 1 and sys.argv[1] == "--pdf":
        if len(sys.argv) < 3:
            print("Usage: python -m backend.utils.ingestion --pdf <path> [category] [subcategory]")
            sys.exit(1)
        pdf_path = sys.argv[2]
        cat = sys.argv[3] if len(sys.argv) > 3 else "Product Toolkit"
        sub = sys.argv[4] if len(sys.argv) > 4 else "Product Toolkit 2026"
        ingest_pdf_file(pdf_path, category=cat, subcategory=sub)
        c = get_collection().count()
        print(f"\n✅ Collection '{CHROMA_COLLECTION}' now has {c} vectors.")
    else:
        ingest_all_pages()
