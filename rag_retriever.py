import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model - must match rag_loader.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KB_FOLDER = "knowledge_base"
VECTORSTORE_DIR = "codex_faiss_index"

# Cache for embeddings and vectorstore to avoid reloading
_embeddings_cache = None
_vectorstore_cache = None

def get_embeddings():
    """Get embeddings with caching to avoid reloading"""
    global _embeddings_cache
    if _embeddings_cache is None:
        print(f"🔄 Loading embeddings model: {EMBEDDING_MODEL}")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        print("✅ Embeddings model loaded successfully")
    return _embeddings_cache

def load_vectorstore():
    """Load vectorstore with caching and error handling"""
    global _vectorstore_cache
    
    if _vectorstore_cache is None:
        try:
            print(f"🔄 Loading FAISS vectorstore from {VECTORSTORE_DIR}")
            
            # Check if vectorstore files exist
            if not os.path.exists(f"{VECTORSTORE_DIR}/index.faiss"):
                raise FileNotFoundError(f"FAISS index not found at {VECTORSTORE_DIR}/index.faiss")
            
            embeddings = get_embeddings()
            _vectorstore_cache = FAISS.load_local(
                VECTORSTORE_DIR, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS vectorstore loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            raise
    
    return _vectorstore_cache

def retrieve_codex_context(user_prompt, venue_concept, max_results=6):
    """
    Retrieve relevant context from the knowledge base
    
    Args:
        user_prompt (str): User's question
        venue_concept (str): Venue type/concept
        max_results (int): Maximum number of documents to retrieve
    
    Returns:
        str: Concatenated relevant context
    """
    try:
        print(f"🔄 Retrieving context for query: '{user_prompt[:50]}...'")
        
        vectordb = load_vectorstore()
        
        # Create enhanced query with venue context
        enhanced_query = f"Venue Type: {venue_concept}. Question: {user_prompt}"
        
        # Retrieve relevant documents
        retriever = vectordb.as_retriever(search_kwargs={"k": max_results})
        docs = retriever.invoke(enhanced_query)  # Use invoke instead of deprecated method
        
        if not docs:
            print("⚠️  No relevant documents found")
            return "No relevant context found in knowledge base."
        
        # Combine document contents
        context_parts = []
        for i, doc in enumerate(docs):
            print(f"\n📄 Retrieved Context {i+1}:\n{doc.page_content}\n")
            context_parts.append(f"[Context {i+1}]\n{doc.page_content}")
                    
        context = "\n\n".join(context_parts)
        print(f"✅ Retrieved {len(docs)} documents, {len(context)} characters total")
        
        return context
        
    except Exception as e:
        print(f"❌ Error retrieving context: {e}")
        # Return a fallback message instead of failing completely
        return f"Knowledge base temporarily unavailable. Providing general bartending advice for {venue_concept}."

def check_vectorstore_health():
    """Check if vectorstore is healthy and can be loaded"""
    try:
        vectordb = load_vectorstore()
        # Test a simple query
        test_docs = vectordb.similarity_search("cocktail", k=1)
        return True, f"Vectorstore healthy, {len(test_docs)} test documents found"
    except Exception as e:
        return False, f"Vectorstore error: {str(e)}"

def check_and_update_vectorstore(kb_folder):
    """Check vectorstore health and optionally rebuild"""
    try:
        health_ok, health_msg = check_vectorstore_health()
        print(f"📊 Vectorstore health: {health_msg}")
        
        if not health_ok:
            print("⚠️  Vectorstore issues detected, but continuing with degraded functionality")
            
    except Exception as e:
        print(f"⚠️  Vectorstore check failed: {e}")

def clear_cache():
    """Clear cached embeddings and vectorstore (useful for memory management)"""
    global _embeddings_cache, _vectorstore_cache
    _embeddings_cache = None
    _vectorstore_cache = None
    print("🗑️  Cleared vectorstore and embeddings cache")

if __name__ == "__main__":
    # Test the retriever
    print("Testing RAG retriever...")
    try:
        context = retrieve_codex_context("What is a martini?", "upscale cocktail bar")
        print(f"Test successful: {len(context)} characters retrieved")
    except Exception as e:
        print(f"Test failed: {e}")
