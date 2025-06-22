import os
import pickle
import hashlib
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

VECTORSTORE_DIR = "vectorstore"
VECTORSTORE_FILE = os.path.join(VECTORSTORE_DIR, "faiss_index")
FILE_HASHES_PATH = os.path.join(VECTORSTORE_DIR, "file_hashes.pkl")

def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def get_all_file_hashes(folder: str) -> dict:
    hashes = {}
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.endswith(".pdf") or filename.endswith(".txt"):
            hashes[filename] = compute_file_hash(filepath)
    return hashes

def load_documents(knowledge_folder: str) -> List[str]:
    documents = []
    for filename in os.listdir(knowledge_folder):
        filepath = os.path.join(knowledge_folder, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue
        documents.extend(loader.load())
    return documents

def build_vectorstore(documents: List[str]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists(VECTORSTORE_DIR):
        os.makedirs(VECTORSTORE_DIR)

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTORSTORE_FILE)

def check_and_update_vectorstore(knowledge_folder: str = "knowledge_base"):
    current_hashes = get_all_file_hashes(knowledge_folder)

    if not os.path.exists(FILE_HASHES_PATH) or not os.path.exists(VECTORSTORE_FILE):
        print("Vectorstore or hash file missing. Rebuilding vectorstore.")
        documents = load_documents(knowledge_folder)
        build_vectorstore(documents)
        with open(FILE_HASHES_PATH, "wb") as f:
            pickle.dump(current_hashes, f)
        return

    with open(FILE_HASHES_PATH, "rb") as f:
        previous_hashes = pickle.load(f)

    if current_hashes != previous_hashes:
        print("Changes detected in knowledge base. Rebuilding vectorstore.")
        documents = load_documents(knowledge_folder)
        build_vectorstore(documents)
        with open(FILE_HASHES_PATH, "wb") as f:
            pickle.dump(current_hashes, f)
    else:
        print("No changes in knowledge base. Vectorstore is up to date.")

def retrieve_codex_context(prompt: str) -> str:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTORSTORE_FILE, embeddings, allow_dangerous_deserialization=True)
    results = db.similarity_search(prompt, k=3)
    return "\n\n".join([doc.page_content for doc in results])