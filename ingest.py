
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DB_PATH = str(ROOT / "chroma_db")
COLLECTION_NAME = "universal_tech_privacy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CHARS = 1200
OVERLAP = 150


def find_documents():
    """Find Markdown files regardless of folder-name capitalization."""
    found = []

    for path in ROOT.rglob("*.md"):
        # Ignore virtual environment, database files, and README.
        if any(part.lower() in {".venv", "chroma_db"} for part in path.parts):
            continue

        if path.name.lower() == "readme.md":
            continue

        found.append(path)

    # Remove duplicate paths and sort for predictable output.
    found = sorted(set(found))

    print("\n========== DOCUMENT DISCOVERY ==========")
    print(f"Project folder: {ROOT}")
    print(f"Database folder: {DB_PATH}")
    print(f"Documents found: {len(found)}")

    for path in found:
        print(f"  FOUND: {path.relative_to(ROOT)}")

    if not found:
        raise FileNotFoundError(
            "No Markdown policy documents were found. "
            "Place your .md files inside the project folder."
        )

    return found


def split_oversized(text):
    """Split long sections with a small overlap."""
    chunks = []
    remaining = text.strip()

    while len(remaining) > MAX_CHARS:
        cut = remaining.rfind("\n", 0, MAX_CHARS)

        if cut < MAX_CHARS // 2:
            cut = remaining.rfind(" ", 0, MAX_CHARS)

        if cut < MAX_CHARS // 2:
            cut = MAX_CHARS

        piece = remaining[:cut].strip()

        if piece:
            chunks.append(piece)

        remaining = remaining[max(0, cut - OVERLAP):].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def chunk_document(text):
    """Split by Markdown headings, keeping headings with their sections."""
    text = text.replace("\r\n", "\n").strip()
    sections = re.split(r"(?m)(?=^#{1,3}\s+)", text)

    chunks = []
    current = ""

    for section in sections:
        section = section.strip()

        if not section:
            continue

        if len(section) > MAX_CHARS:
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(split_oversized(section))
            continue

        combined = f"{current}\n\n{section}".strip() if current else section

        if current and len(combined) > MAX_CHARS:
            chunks.append(current)
            current = section
        else:
            current = combined

    if current:
        chunks.append(current)

    return chunks


def get_section(chunk):
    match = re.search(r"(?m)^#{1,3}\s+(.+)$", chunk)
    return match.group(1).strip() if match else "General"


def main():
    files = find_documents()

    all_chunks = []
    all_metadata = []

    for path in files:
        text = path.read_text(encoding="utf-8-sig").strip()

        if not text:
            print(f"WARNING: Empty file: {path.name}")
            continue

        chunks = chunk_document(text)
        relative_path = str(path.relative_to(ROOT))

        print(f"\nProcessing: {relative_path}")
        print(f"Characters: {len(text)}")
        print(f"Chunks: {len(chunks)}")

        for index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": relative_path,
                "section": get_section(chunk),
                "chunk": index,
            })

    if not all_chunks:
        raise ValueError("No chunks were generated from the documents.")

    print(f"\nTotal documents: {len(files)}")
    print(f"Total chunks: {len(all_chunks)}")
    print("\nLoading embedding model...")

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = embedder.encode(
        all_chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    db = chromadb.PersistentClient(path=DB_PATH)

    try:
        db.delete_collection(COLLECTION_NAME)
        print("Deleted previous collection.")
    except Exception:
        print("Creating a new collection.")

    collection = db.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100

    for start in range(0, len(all_chunks), batch_size):
        end = min(start + batch_size, len(all_chunks))

        collection.add(
            ids=[f"chunk_{i}" for i in range(start, end)],
            documents=all_chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=all_metadata[start:end],
        )

    print("\n========== INGESTION COMPLETE ==========")
    print(f"Indexed chunks: {collection.count()}")

    print("\nIndexed sources:")
    for source in sorted({m["source"] for m in all_metadata}):
        print(f"  - {source}")

    # Test whether shipping information is retrievable.
    question = (
        "According to the shipping and returns policy, "
        "how long does standard shipping take after dispatch?"
    )

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(5, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    print("\n========== SHIPPING RETRIEVAL TEST ==========")

    for index, metadata in enumerate(results["metadatas"][0]):
        print(f"\nResult {index + 1}")
        print(f"Source: {metadata.get('source')}")
        print(f"Section: {metadata.get('section')}")
        print(results["documents"][0][index][:500])

    print("\nFinished. You can now restart Streamlit.")


if __name__ == "__main__":
    main()
