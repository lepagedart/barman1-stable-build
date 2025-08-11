# rag_retriever.py
import os
import re
import logging
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model - must match kb_loader.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KB_FOLDER = "knowledge_base"
VECTORSTORE_DIR = "codex_faiss_index"

# Cache
_embeddings_cache = None
_vectorstore_cache = None

# ---------------------------
# Loaders / Caches
# ---------------------------
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

# ---------------------------
# Scenario / Tag Steering
# ---------------------------
_SCENARIO_NUM_RE = re.compile(r"\bscenario\s*#?\s*(\d{1,3})\b", re.IGNORECASE)

def _detect_scenario_num(text: str) -> Optional[str]:
    if not text:
        return None
    m = _SCENARIO_NUM_RE.search(text)
    return m.group(1) if m else None

def _intent_tags_from_prompt(prompt: str) -> List[str]:
    """Lightweight keyword→tag mapping. Extend as you add scenarios."""
    p = (prompt or "").lower()
    tags = set()

    # Scenario 15 – promotions / slow season
    if any(w in p for w in ["promotion", "promotions", "slow season", "slow-month", "slow month",
                             "guest engagement", "repeat visit", "repeat-visit"]):
        tags.update(["promotions", "slow_season", "guest_engagement", "menu_strategy", "profit_margins"])

    # High-volume ops
    if any(w in p for w in ["speed rail", "ticket times", "wells", "throughput", "station layout"]):
        tags.update(["service_speed", "station_layout", "batching"])

    # Seasonal R&D – stone fruit example
    if any(w in p for w in ["stone fruit", "peach", "nectarine", "plum"]):
        tags.update(["seasonal", "stone_fruit", "syrup_base", "photography"])

    # Costing / audit
    if any(w in p for w in ["pour cost", "costing", "profit", "margin", "audit"]):
        tags.update(["costing", "profit_margins", "menu_engineering"])

    return list(tags)

def _score_doc(meta: dict, scenario_num: Optional[str], wanted_tags: List[str]) -> float:
    """
    Scoring:
      +2.0 filename/tags match scenario_num
      +1.0 per wanted_tag found in metadata 'tags'
      +0.5 if path name suggests scenario file
    """
    score = 0.0
    tags = (meta or {}).get("tags", "") or ""
    path = (meta or {}).get("source", "") or ""  # many loaders store file path as 'source'

    if scenario_num:
        # Normalize tag spaces
        tag_norm = tags.replace(" ", "").lower()
        if f"scenario_num={scenario_num}" in tag_norm:
            score += 2.0
        if re.search(rf"scenario[_-]{re.escape(scenario_num)}\b", path.lower()):
            score += 0.5

    if wanted_tags:
        t_norm = tags.lower()
        for t in wanted_tags:
            if t and t.lower() in t_norm:
                score += 1.0

    return score

def _rerank(candidates, scenario_num: Optional[str], wanted_tags: List[str], top_k: int):
    scored = []
    for d in candidates:
        meta = getattr(d, "metadata", {}) or {}
        s = _score_doc(meta, scenario_num, wanted_tags)
        scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k]]

# ---------------------------
# Public API
# ---------------------------
def retrieve_codex_context(user_prompt, venue_context, max_results=6, use_live_search=False):
    """
    Retrieve contextual documents, boosting exact-scenario docs and tag matches.
    """
    try:
        print(f"🔍 RAG query: '{user_prompt[:50]}...' with venue '{venue_context}'")
        vectordb = load_vectorstore()
        query = f"Venue: {venue_context}\nQuestion: {user_prompt}"

        # Step 1: get a candidate pool larger than k
        k_pool = max(max_results * 3, 12)
        candidates = vectordb.similarity_search(query, k=k_pool)

        if not candidates:
            print("⚠️ No relevant documents found")
            return "No relevant context found in knowledge base."

        # Step 2: detect scenario + intent tags
        scenario_num = _detect_scenario_num(user_prompt) or _detect_scenario_num(venue_context)
        wanted_tags = _intent_tags_from_prompt(user_prompt)

        # Step 3: rerank & take top-k
        docs = _rerank(candidates, scenario_num, wanted_tags, top_k=max_results)

        # Step 4: build context + print sources for your terminal
        context_parts = []
        top_sources_lines = []
        for i, doc in enumerate(docs):
            meta = doc.metadata or {}
            category = meta.get("category", "unspecified")
            tags = meta.get("tags", "")
            source = meta.get("source", "")
            print(f"\n📄 Context {i+1} (Category: {category})\n{doc.page_content}\n")
            context_parts.append(f"[Context {i+1} – {category}]\n{doc.page_content}")
            top_sources_lines.append(f"{i+1}. source={source} | tags={tags}")

        logging.getLogger(__name__).info(
            "RAG rerank → scenario=%s tags=%s\nTop sources:\n%s",
            scenario_num, wanted_tags, "\n".join(top_sources_lines)
        )

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

def check_and_update_vectorstore(kb_folder=None):
    """
    Kept for compatibility with your app import.
    Here we just report health; if you later add hot-reload, wire it here.
    """
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
        ctx = retrieve_codex_context("Promotions for slow season without discounts", "75-seat American bistro")
        print(f"✅ Success – {len(ctx)} characters retrieved")
    except Exception as e:
        print(f"❌ Test failed: {e}")