import os
from langchain_community.vectorstores import FAISS
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
    vectordb = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectordb
def retrieve_codex_context(user_prompt, venue_concept):
    vectordb = load_vectorstore()
    
    # Combine venue concept and user prompt into the retrieval query
    query = f"Venue Concept: {venue_concept}. Question: {user_prompt}"
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def check_and_update_vectorstore(kb_folder):
    pass  # Leave your existing vectorstore rebuild logic intact