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

def extract_tags_from_content(text):
    lines = text.splitlines()
    for line in lines[:5]:
        if "tags:" in line.lower():
            tag_str = line.split(":", 1)[1]
            return [t.strip() for t in tag_str.split(",")]
    return None

def fallback_tags(file, root):
    tags = []
    if "vodka" in root.lower() or "vodka" in file.lower():
        tags.append("Vodka")
    if "tequila" in root.lower() or "tequila" in file.lower():
        tags.append("Tequila")
    if "rum" in root.lower() or "rum" in file.lower():
        tags.append("Rum")
    if "gin" in root.lower() or "gin" in file.lower():
        tags.append("Gin")
    if "whiskey" in root.lower() or "bourbon" in root.lower():
        tags.append("Whiskey")
    if "brandy" in root.lower() or "brandy" in file.lower():
        tags.append("Brandy")
    if "mule" in file.lower() or "highball" in root.lower():
        tags.append("Highball")
    if "zero" in file.lower() or "non-alcoholic" in file.lower():
        tags.append("Non-Alcoholic")
    if "batch" in file.lower():
        tags.append("Batchable")
    if "classic" in file.lower():
        tags.append("Classic")
    if "house" in file.lower():
        tags.append("House Build")
    return tags

def create_vector_store(knowledge_dir=KNOWLEDGE_DIR):
    manifest = load_manifest()
    updated_manifest = {}
    docs_to_embed = []

    for root, _, files in os.walk(knowledge_dir):
        for file in files:
            if not (file.endswith(".txt") or file.endswith(".pdf") or file.endswith(".md") or file.endswith(".csv")):
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
                elif file.endswith(".csv"):
                    loader = CSVLoader(path)
                else:
                    continue

                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file
                    tags = extract_tags_from_content(doc.page_content)
                    if tags:
                        doc.metadata["tags"] = tags
                        print(f"🏷️  Tags from file content: {tags}")
                    else:
                        doc.metadata["tags"] = fallback_tags(file, root)
                        print(f"🔎 Tags from fallback: {doc.metadata['tags']}")

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

if __name__ == "__main__":
    create_vector_store()