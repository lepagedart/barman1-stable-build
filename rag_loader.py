import os
import hashlib
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === CONFIGURATION ===
KNOWLEDGE_DIR = "knowledge_base"
VECTORSTORE_DIR = "codex_faiss_index"
MANIFEST_FILE = "vectorstore_manifest.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === HELPER FUNCTIONS ===

def load_manifest():
    """Load the vectorstore manifest from disk."""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}

def save_manifest(manifest):
    """Save updated manifest to disk."""
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

def file_hash(path):
    """Compute SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

# === MAIN VECTORSTORE BUILDER ===

def create_vector_store(knowledge_dir=KNOWLEDGE_DIR):
    manifest = load_manifest()
    updated_manifest = {}
    docs_to_embed = []

    for root, _, files in os.walk(knowledge_dir):
        for file in files:
            if not (file.endswith(".txt") or file.endswith(".pdf") or file.endswith(".md")):
                continue

            path = os.path.join(root, file)
            hash_val = file_hash(path)
            updated_manifest[path] = hash_val

            if manifest.get(path) == hash_val:
                print(f"🟡 Skipping unchanged file: {file}")
                continue

            print(f"🟢 Processing: {file}")
            try:
                if file.endswith(".txt") or file.endswith(".md"):
                    loader = TextLoader(path)
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file
                docs_to_embed.extend(docs)
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")

    if not docs_to_embed:
        print("✅ No new or updated documents to embed.")
        return

    print(f"🔄 Embedding {len(docs_to_embed)} documents...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs_to_embed)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print("✅ Vectorstore updated successfully.")
    save_manifest(updated_manifest)

# === ENTRY POINT ===

if __name__ == "__main__":
    create_vector_store()