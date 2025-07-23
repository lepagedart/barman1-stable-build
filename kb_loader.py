import os
import hashlib
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === CONFIGURATION ===
KNOWLEDGE_DIR = "knowledge_base"
VECTORSTORE_DIR = "codex_faiss_index"
MANIFEST_FILE = "vectorstore_manifest.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}

def save_manifest(manifest):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

def file_hash(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def detect_category_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for keyword in ["recipes", "frameworks", "scenarios", "training_modules", "methods_techniques"]:
        for part in parts:
            if keyword.lower() in part.lower():
                return keyword.lower()
    return "general"

def load_knowledge_documents():
    manifest = load_manifest()
    updated_manifest = {}
    docs_to_embed = []

    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for file in files:
            if not file.lower().endswith((".txt", ".pdf", ".csv", ".md")):
                continue
            path = os.path.join(root, file)
            hash_val = file_hash(path)
            updated_manifest[path] = hash_val

            if manifest.get(path) == hash_val:
                continue

            try:
                if file.endswith(".txt") or file.endswith(".md"):
                    loader = TextLoader(path)
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif file.endswith(".csv"):
                    loader = CSVLoader(path)
                else:
                    continue

                docs = loader.load()
                category = detect_category_from_path(path)
                for doc in docs:
                    doc.metadata["source"] = path
                    doc.metadata["category"] = category
                docs_to_embed.extend(docs)
                print(f"📄 Loaded: {file} ({category})")
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")

    if not docs_to_embed:
        print("✅ No new or updated documents found.")
        return

    print(f"🔄 Embedding {len(docs_to_embed)} documents...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs_to_embed)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    save_manifest(updated_manifest)
    print("✅ Vectorstore updated successfully.")

if __name__ == "__main__":
    load_knowledge_documents()