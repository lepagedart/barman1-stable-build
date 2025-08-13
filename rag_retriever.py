# rag_retriever.py
import os
import re
import json
import string
from typing import List, Optional, Tuple, Dict

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# Embedding model - must match kb_loader.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTORSTORE_DIR = "codex_faiss_index"

# Caches
_embeddings_cache = None
_vectorstore_cache = None
_manifest_cache = None
_tag_vocab_cache = None

# ---------- load FAISS / embeddings ----------
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
        print(f"🔄 Loading FAISS vectorstore from {VECTORSTORE_DIR}")
        if not os.path.exists(f"{VECTORSTORE_DIR}/index.faiss"):
            raise FileNotFoundError(f"Vectorstore missing at {VECTORSTORE_DIR}/index.faiss")
        embeddings = get_embeddings()
        _vectorstore_cache = FAISS.load_local(
            VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )
        print("✅ Vectorstore loaded")
    return _vectorstore_cache

# ---------- regex helpers ----------
_SCENARIO_NUM_RE     = re.compile(r"\bscenario[_\s-]*#?\s*(\d{1,3})\b", re.IGNORECASE)
_TAGS_LINE_RE        = re.compile(r"^Tags:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_SCENARIO_IN_PATH_RE = re.compile(r"scenario[_-](\d{1,3})", re.IGNORECASE)

def _detect_scenario_num(text: str) -> Optional[str]:
    if not text: return None
    m = _SCENARIO_NUM_RE.search(text)
    return m.group(1) if m else None

def _extract_tags(meta: dict, content: str) -> List[str]:
    # Prefer metadata (from kb_loader); fall back to parsing line 2.
    raw = (meta or {}).get("tags", "")
    if not raw:
        m = _TAGS_LINE_RE.search(content or "")
        raw = m.group(1) if m else ""
    if not raw: return []
    raw = raw.replace(",", ";")
    return [t.strip().lower() for t in raw.split(";") if t.strip()]

def _scenario_from_path(meta: dict) -> Optional[str]:
    path = (meta or {}).get("source", "") or ""
    m = _SCENARIO_IN_PATH_RE.search(path)
    return m.group(1) if m else None

# ---------- dynamic intent (generalized) ----------
WOWY_TAGS = {"wow_factor","presentation","menu_standout","creative_builds","signature_drink"}

def _load_tag_vocab() -> set:
    """Read tag vocabulary built by kb_loader; fallback to empty set."""
    global _manifest_cache, _tag_vocab_cache
    if _tag_vocab_cache is not None:
        return _tag_vocab_cache
    try:
        path = os.path.join(VECTORSTORE_DIR, "vectorstore_manifest.json")
        with open(path, "r", encoding="utf-8") as f:
            _manifest_cache = json.load(f)
        _tag_vocab_cache = set(_manifest_cache.get("tag_vocab", []))
        print(f"🗂  Loaded {len(_tag_vocab_cache)} tags from vectorstore_manifest.json")
    except Exception:
        _tag_vocab_cache = set()
        print("⚠️ No manifest/tag vocab found; continuing without dynamic intent tags")
    return _tag_vocab_cache

def _normalize_token(w: str) -> str:
    w = w.lower().strip()
    w = w.translate(str.maketrans("", "", string.punctuation))
    return w.replace(" ", "_")

def _intent_tags_from_text(text: str) -> List[str]:
    """
    Map words/phrases in the prompt/venue to *existing KB tags*.
    Auto-scales as you add scenarios/tags—no per-scenario code.
    """
    vocab = _load_tag_vocab()
    if not vocab:
        return []
    raw = (text or "").lower()
    candidates = set()

    words = [_normalize_token(w) for w in raw.split()]
    # unigrams
    for w in words:
        if w in vocab:
            candidates.add(w)
    # bigram sweep (cheap heuristic)
    for i in range(len(words)-1):
        bg = f"{words[i]}_{words[i+1]}"
        if bg in vocab:
            candidates.add(bg)
    # substring catch-all for multiword tags
    joined = raw.replace(" ", "_")
    for tag in vocab:
        if tag in joined:
            candidates.add(tag)

    return list(candidates)

def _disallowed_tags_for_prompt(prompt: str) -> List[str]:
    allow_wow = any(w in (prompt or "").lower() for w in ["wow","standout","instagram","visual","presentation"])
    return [] if allow_wow else ["wow_factor"]

# ---------- scoring, filtering, diversity ----------
def _overlap(a: List[str], b: List[str]) -> int:
    sa, sb = set(a), set(b)
    return len(sa & sb)

def _tag_penalty(tags: List[str], disallowed: List[str]) -> float:
    score = 0.0
    if any(t in tags for t in disallowed):
        score -= 1.0
    # soft downweight for flashy tags by default
    if set(tags) & WOWY_TAGS:
        score -= 1.0
    return score

def _score_doc(meta: dict, content: str, wanted: List[str], disallowed: List[str], scenario_hint: Optional[str]) -> float:
    tags = _extract_tags(meta, content)
    path_scn = _scenario_from_path(meta)

    score = 0.0
    ov = _overlap(tags, wanted)
    score += 1.6 * ov  # reward multi-tag matches

    # light fallback if doc has some tags but no overlap
    if ov == 0 and tags:
        score += 0.2

    # explicit penalties (wow_factor unless asked, etc.)
    score += _tag_penalty(tags, disallowed)

    # tiny nudge if scenario # matches hint (only if explicitly mentioned by user)
    if scenario_hint and path_scn == scenario_hint:
        score += 0.75

    return score

def _filter_by_required_tags(candidates, wanted: List[str]) -> List:
    """If intent tags exist, keep docs sharing at least one. Else return all."""
    if not wanted: return candidates
    kept = []
    for d in candidates:
        meta = getattr(d, "metadata", {}) or {}
        content = getattr(d, "page_content", "") or ""
        tags = _extract_tags(meta, content)
        if _overlap(tags, wanted) >= 1:
            kept.append(d)
    return kept or candidates

def _enforce_scenario_diversity(docs: List, max_per_scenario: int = 2) -> List:
    """Cap how many chunks any single scenario can contribute."""
    bucket: Dict[str, int] = {}
    out = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        scn = _scenario_from_path(meta) or "none"
        if bucket.get(scn, 0) >= max_per_scenario:
            continue
        out.append(d)
        bucket[scn] = bucket.get(scn, 0) + 1
    return out

# ---------- public API ----------
def retrieve_codex_context(user_prompt, venue_context, max_results=6, use_live_search=False):
    """
    Retrieve contextual docs with:
      - wide vector recall
      - dynamic tag intent derived from KB vocabulary
      - disallowed/wow downweight (unless requested)
      - scenario diversity cap
    """
    try:
        print(f"🔍 RAG query: '{user_prompt[:60]}...' with venue '{venue_context}'")
        vectordb = load_vectorstore()
        query = f"Venue: {venue_context}\nQuestion: {user_prompt}"

        # Step 1: wide candidate pool for better re-ranking
        k_pool = max(max_results * 5, 20)
        candidates = vectordb.similarity_search(query, k=k_pool)
        if not candidates:
            print("⚠️ No relevant documents found")
            return "No relevant context found in knowledge base."

        # Step 2: infer intent from BOTH prompt and venue (dynamic to vocab)
        scenario_hint = _detect_scenario_num(user_prompt) or _detect_scenario_num(venue_context)
        wanted = set(_intent_tags_from_text(user_prompt)) | set(_intent_tags_from_text(venue_context))
        wanted = list(wanted)
        disallowed = _disallowed_tags_for_prompt(user_prompt)

        # Step 3: filter by required tags (fail-open)
        candidates = _filter_by_required_tags(candidates, wanted)

        # Step 4: score & sort
        scored = []
        for d in candidates:
            meta = getattr(d, "metadata", {}) or {}
            content = getattr(d, "page_content", "") or ""
            s = _score_doc(meta, content, wanted, disallowed, scenario_hint)
            scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Step 5: scenario diversity
        ranked = [d for s, d in scored]
        ranked = _enforce_scenario_diversity(ranked, max_per_scenario=2)

        # Step 6: take top-k and build context
        docs = ranked[:max_results]
        print("🔝 Top sources (after rerank):")
        parts = []
        for i, doc in enumerate(docs):
            meta = getattr(doc, "metadata", {}) or {}
            category = meta.get("category", "unspecified")
            source = meta.get("source", "")
            content = getattr(doc, "page_content", "") or ""
            tag_str = "; ".join(_extract_tags(meta, content))
            print(f"  {i+1}. source={source} | category={category} | tags={tag_str}")
            parts.append(f"[Context {i+1} – {category}]\n{content}")

        ctx = "\n\n".join(parts)
        print(f"✅ {len(docs)} docs, {len(ctx)} characters total")
        return ctx

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
    ok, msg = check_vectorstore_health()
    print(f"📊 Vectorstore health check: {msg}")
    if not ok:
        print("⚠️ Issues detected — consider /reindex")

def clear_cache():
    global _embeddings_cache, _vectorstore_cache, _manifest_cache, _tag_vocab_cache
    _embeddings_cache = None
    _vectorstore_cache = None
    _manifest_cache = None
    _tag_vocab_cache = None
    print("🗑️ Cleared vectorstore and embedding/tag vocab caches")

if __name__ == "__main__":
    print("🧪 Testing RAG retriever (generalized intent)...")
    try:
        ctx = retrieve_codex_context(
            "First 30 days plan for a new bar manager—stabilize operations and early wins",
            "A newly promoted bar manager in a craft-focused venue"
        )
        print(f"✅ Success – {len(ctx)} characters retrieved")
    except Exception as e:
        print(f"❌ Test failed: {e}")