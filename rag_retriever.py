import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KB_FOLDER = "knowledge_base"
VECTORSTORE_DIR = "vectorstore"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vectordb = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
    return vectordb

def retrieve_codex_context(user_prompt, venue_concept):
    vectordb = load_vectorstore()

    query = f"Venue Concept: {venue_concept}. Question: {user_prompt}"
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def check_and_update_vectorstore(kb_folder):
    pass  # Your existing rebuild logic can go here if needed
