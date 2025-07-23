import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model - must match kb_loader.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KB_FOLDER = "knowledge_base"
VECTORSTORE_DIR = "codex_faiss_index"

# Cache
_embeddings_cache = None
_vectorstore_cache = None

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        print(f"🔄 Loading embeddings model: {EMBEDDING_MODEL}")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        print("✅ Embeddings model loaded")
    return _embeddings_cache

def load_vectorstore():
    global _vectorstore_cache
    if _vectorstore_cache is None:
        try:
            print(f"🔄 Loading FAISS vectorstore from {VECTORSTORE_DIR}")
            if not os.path.exists(f"{VECTORSTORE_DIR}/index.faiss"):
                raise FileNotFoundError(f"Vectorstore missing at {VECTORSTORE_DIR}/index.faiss")
            embeddings = get_embeddings()
            _vectorstore_cache = FAISS.load_local(
                VECTORSTORE_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Vectorstore loaded")
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            raise
    return _vectorstore_cache

def retrieve_codex_context(user_prompt, venue_context, max_results=6, use_live_search=False):
    """
    Retrieve contextual documents from vectorstore with optional filter logic.
    """
    try:
        print(f"🔍 RAG query: '{user_prompt[:50]}...' with venue '{venue_context}'")
        vectordb = load_vectorstore()
        query = f"Venue Type: {venue_context}. Question: {user_prompt}"

        retriever = vectordb.as_retriever(search_kwargs={"k": max_results})
        docs = retriever.invoke(query)

        if not docs:
            print("⚠️ No relevant documents found")
            return "No relevant context found in knowledge base."

        context_parts = []
        for i, doc in enumerate(docs):
            meta = doc.metadata or {}
            category = meta.get("category", "unspecified")
            print(f"\n📄 Context {i+1} (Category: {category})\n{doc.page_content}\n")
            context_parts.append(f"[Context {i+1} – {category}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)
        print(f"✅ {len(docs)} docs, {len(context)} characters total")
        return context

    except Exception as e:
        print(f"❌ Error retrieving RAG context: {e}")
        return f"Knowledge base temporarily unavailable. Providing general bartending advice for {venue_context}."

def check_vectorstore_health():
    try:
        vectordb = load_vectorstore()
        test_docs = vectordb.similarity_search("cocktail", k=1)
        return True, f"Vectorstore healthy – {len(test_docs)} test doc(s) retrieved"
    except Exception as e:
        return False, f"Vectorstore error: {str(e)}"

def check_and_update_vectorstore(kb_folder):
    try:
        health_ok, msg = check_vectorstore_health()
        print(f"📊 Vectorstore health check: {msg}")
        if not health_ok:
            print("⚠️ Issues detected — fallback may apply")
    except Exception as e:
        print(f"⚠️ Vectorstore check failed: {e}")

def clear_cache():
    global _embeddings_cache, _vectorstore_cache
    _embeddings_cache = None
    _vectorstore_cache = None
    print("🗑️ Cleared vectorstore and embedding cache")

if __name__ == "__main__":
    print("🧪 Testing RAG retriever...")
    try:
        ctx = retrieve_codex_context("What is a martini?", "upscale cocktail bar")
        print(f"✅ Success – {len(ctx)} characters retrieved")
    except Exception as e:
        print(f"❌ Test failed: {e}")