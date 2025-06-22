from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def create_vector_store(knowledge_dir="knowledge_base"):
    all_docs = []

    for root, _, files in os.walk(knowledge_dir):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".txt"):
                loader = TextLoader(path)
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                continue
            all_docs.extend(loader.load())

    if not all_docs:
        print("❌ No documents loaded.")
        return

    # Ensure SAME model as retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = text_splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("codex_faiss_index")
    print("✅ FAISS vector store created successfully.")

if __name__ == "__main__":
    create_vector_store()