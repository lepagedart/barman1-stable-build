# kb_loader.py
# ----------------------------
# Build & save a FAISS index from knowledge_base/ .txt files.
# - Reads Title/Tags headers
# - Adds rich metadata (title, tags, category, source)
# - Chunks content
# - Uses HuggingFaceEmbeddings (all-MiniLM-L6-v2)
# - Saves to codex_faiss_index/
# ----------------------------

import os
import re
import sys
import glob
import argparse
from typing import List, Tuple, Dict

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# === Config ===
KB_ROOT = "knowledge_base"
VECTORSTORE_DIR = "codex_faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Which folders to include (relative to KB_ROOT). Add/adjust as you expand.
INCLUDE_DIRS = [
    "training_modules/scenarios_prompts",
    "training_modules/scenarios_reference",
    "training_modules/scenarios_rubric",
    "training_modules/Frameworks",            # if you store frameworks here
    "frameworks",                             # legacy / optional
    "methods_techniques",                     # optional
]

# Basic text chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

HEADER_TITLE_RE = re.compile(r"^\s*title\s*:\s*(.+)$", re.IGNORECASE)
HEADER_TAGS_RE  = re.compile(r"^\s*tags\s*:\s*(.+)$", re.IGNORECASE)

# Cache
_embeddings_cache = None


# ---------- Utilities ----------

def normalize_tag(t: str) -> str:
    t = t.strip().lower()
    t = t.replace(" ", "_")
    t = re.sub(r"[^a-z0-9_,-]", "", t)
    return t

def parse_header(lines: List[str]) -> Tuple[str, str]:
    """Parse the first two lines for Title: and Tags: (optional)."""
    title, tags = None, ""
    if lines:
        m = HEADER_TITLE_RE.match(lines[0].strip())
        if m:
            title = m.group(1).strip()
    if len(lines) > 1:
        m = HEADER_TAGS_RE.match(lines[1].strip())
        if m:
            raw = m.group(1)
            parts = [normalize_tag(p) for p in raw.split(",")]
            parts = [p for p in parts if p]
            tags = ",".join(parts)
    return title or "", tags

def read_txt_with_header(path: str) -> Tuple[str, str, str]:
    """Return (title, tags_csv, body_text). If no header, title=basename, tags=''. """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    title, tags = parse_header(lines[:2])
    body = "".join(lines[2:]) if title else "".join(lines)
    if not title:
        title = os.path.basename(path)
    return title.strip(), tags.strip(), body.strip()

def simple_chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunk = text[i:end]
        chunks.append(chunk)
        if end == n:
            break
        i = max(0, end - overlap)
    return chunks

def get_category(path: str) -> str:
    """Derive a category label from subdirs under KB_ROOT."""
    rel = os.path.relpath(path, KB_ROOT)
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        # training_modules/<subdir>/...
        return f"{parts[0]}/{parts[1]}"
    elif parts:
        return parts[0]
    return "unknown"

def collect_txt_files() -> List[str]:
    files = []
    for sub in INCLUDE_DIRS:
        root = os.path.join(KB_ROOT, sub)
        if os.path.isdir(root):
            files.extend(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True))
    # de-dup & sort for determinism
    files = sorted(list({os.path.normpath(p) for p in files}))
    return files

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        print(f"🔧 Loading embeddings: {EMBEDDING_MODEL} (cpu)")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        print("✅ Embeddings ready")
    return _embeddings_cache


# ---------- Build index ----------

def build_documents() -> List[Document]:
    files = collect_txt_files()
    docs: List[Document] = []
    print(f"📚 Scanning KB: {len(files)} .txt files discovered")

    for path in files:
        try:
            title, tags, body = read_txt_with_header(path)
            if not body:
                continue
            category = get_category(path)
            chunks = simple_chunk(body, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, chunk in enumerate(chunks):
                md: Dict[str, str] = {
                    "title": title,
                    "tags": tags,                         # comma-separated
                    "category": category,
                    "source": os.path.relpath(path, KB_ROOT),
                    "chunk_index": str(idx),
                }
                docs.append(Document(page_content=chunk, metadata=md))
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")

    print(f"🧩 Prepared {len(docs)} chunks from KB")
    return docs

def save_faiss(docs: List[Document]) -> int:
    if not docs:
        raise ValueError("No documents to index.")
    embeddings = get_embeddings()
    print("🧠 Building FAISS index...")
    vectordb = FAISS.from_documents(docs, embeddings)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectordb.save_local(VECTORSTORE_DIR)
    write_manifest(docs)
    print(f"💾 Saved FAISS to {VECTORSTORE_DIR}/ (index.faiss + index.pkl)")
    return len(docs)

import json
def write_manifest(docs):
    vocab = {}
    for d in docs:
        tags = (d.metadata or {}).get("tags","")
        for t in [x.strip() for x in tags.split(",") if x.strip()]:
            vocab[t] = vocab.get(t, 0) + 1
    manifest = {"tag_vocab": sorted(vocab.keys())}
    with open(os.path.join(VECTORSTORE_DIR, "vectorstore_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"🗂  Wrote tag vocab ({len(vocab)} tags) to {VECTORSTORE_DIR}/vectorstore_manifest.json")
    
def rebuild_vectorstore() -> int:
    """
    Public entry for app route (/reindex) and CLI.
    Returns the number of chunks indexed.
    """
    docs = build_documents()
    count = save_faiss(docs)
    return count


# ---------- Optional legacy API ----------

def load_knowledge_documents() -> List[Document]:
    """If other code still imports this, return the built doc list (no FAISS write)."""
    return build_documents()


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="KB Loader / FAISS index builder")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild FAISS index from KB")
    parser.add_argument("--dry-run", action="store_true", help="Parse KB and report counts (no index write)")
    parser.add_argument("--show", action="store_true", help="Print first 1-2 docs for sanity")
    args = parser.parse_args()

    if args.dry_run:
        docs = build_documents()
        print(f"✅ Dry-run OK. Chunks: {len(docs)}")
        if args.show and docs:
            for d in docs[:2]:
                print("---")
                print(d.metadata)
                print(d.page_content[:300], "...")
        return

    if args.rebuild:
        # Clean target dir if needed (optional)
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        print(f"🔁 Rebuilding FAISS in {VECTORSTORE_DIR} ...")
        n = rebuild_vectorstore()
        print(f"🎉 Rebuilt. Total chunks indexed: {n}")
        return

    if args.show:
        docs = build_documents()
        for d in docs[:2]:
            print("---")
            print(d.metadata)
            print(d.page_content[:300], "...")
        return

    # Default help
    parser.print_help()

if __name__ == "__main__":
    main()