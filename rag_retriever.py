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

# -------- Retrieval tuning (adaptive K + diversity + budget) --------
DEFAULT_TOP_K = 8          # reasonable middle ground when we can't infer better
MMR_LAMBDA = 0.4           # 0 => pure diversity, 1 => pure relevance; 0.3–0.5 is typical
POOL_MULTIPLIER = 6        # we fetch a wider pool (k * multiplier) before re-ranking
POOL_MIN = 30              # ensure a minimum candidate pool width
SCENARIO_MAX_CHUNKS = 2    # cap chunks per scenario to avoid one-scenario domination
CHAR_BUDGET = 3000         # total characters allowed in the final concatenated context

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
    # Prefer metadata (from kb_loader); fall back to parsing the "Tags:" line.
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
    allow_wow = any(w in (prompt or "").lower() for w in ["wow","standout","instagram","visual","presentation","signature"])
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
    """If intent tags exist, keep docs sharing at least one. Else return all (fail-open)."""
    if not wanted: return candidates
    kept = []
    for d in candidates:
        meta = getattr(d, "metadata", {}) or {}
        content = getattr(d, "page_content", "") or ""
        tags = _extract_tags(meta, content)
        if _overlap(tags, wanted) >= 1:
            kept.append(d)
    return kept or candidates

def _enforce_scenario_diversity(docs: List, max_per_scenario: int = SCENARIO_MAX_CHUNKS) -> List:
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

# ---------- adaptive K helpers ----------
def _adaptive_top_k(user_prompt: str, venue_context: str, wanted_tags: List[str]) -> int:
    """
    Scale K by:
      - richness of intent tags
      - rough prompt length
    """
    tag_count = len(wanted_tags)
    length = len((user_prompt or "")) + len((venue_context or ""))

    # base by tag richness
    if tag_count >= 3:
        base = 10
    elif tag_count == 2:
        base = 8
    elif tag_count == 1:
        base = 7
    else:
        base = DEFAULT_TOP_K  # 8

    # nudge for long prompts (likely multi-faceted)
    if length > 800:
        base += 1
    if length > 1400:
        base += 1

    # clamp
    return max(6, min(base, 12))

def _trim_to_budget(docs: List, budget_chars: int = CHAR_BUDGET) -> List:
    total = 0
    out = []
    for d in docs:
        content = getattr(d, "page_content", "") or ""
        if total + len(content) > budget_chars:
            break
        out.append(d)
        total += len(content)
    return out

# ---------- public API ----------
def retrieve_codex_context(user_prompt, venue_context, max_results=None, use_live_search=False):
    """
    Retrieve contextual docs with:
      - MMR-diversified candidate pool
      - dynamic tag intent derived from KB vocabulary
      - disallowed/wow downweight (unless requested)
      - scenario diversity cap
      - adaptive K with a final character budget
    """
    try:
        print(f"🔍 RAG query: '{(user_prompt or '')[:60]}...' with venue '{(venue_context or '')[:60]}'")
        vectordb = load_vectorstore()
        query = f"Venue: {venue_context}\nQuestion: {user_prompt}"

        # Step 0: infer intent from BOTH prompt and venue (dynamic to vocab)
        scenario_hint = _detect_scenario_num(user_prompt) or _detect_scenario_num(venue_context)
        wanted = list(set(_intent_tags_from_text(user_prompt)) | set(_intent_tags_from_text(venue_context)))
        disallowed = _disallowed_tags_for_prompt(user_prompt)

        # Decide K adaptively unless caller hard-sets max_results
        top_k = _adaptive_top_k(user_prompt, venue_context, wanted) if not max_results else max_results
        pool_k = max(top_k * POOL_MULTIPLIER, POOL_MIN)

        # Step 1: wide pool with MMR for diversity (uses FAISS internal embeddings)
        # fetch_k should be >= pool_k; increase for more variety, then MMR picks k items.
        candidates = vectordb.max_marginal_relevance_search(
            query,
            k=int(pool_k),
            fetch_k=int(pool_k * 2),
            lambda_mult=MMR_LAMBDA,
        )
        if not candidates:
            print("⚠️ No relevant documents found")
            return "No relevant context found in knowledge base."

        # Step 2: filter by required tags (fail-open)
        candidates = _filter_by_required_tags(candidates, wanted)

        # Step 3: score & sort with tag-aware re-rank
        scored = []
        for d in candidates:
            meta = getattr(d, "metadata", {}) or {}
            content = getattr(d, "page_content", "") or ""
            s = _score_doc(meta, content, wanted, disallowed, scenario_hint)
            scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Step 4: scenario diversity cap
        ranked = _enforce_scenario_diversity([d for s, d in scored], max_per_scenario=SCENARIO_MAX_CHUNKS)

        # Step 5: take top-k, then trim to a total character budget
        docs = ranked[:top_k]
        docs = _trim_to_budget(docs, budget_chars=CHAR_BUDGET)

        # Step 6: build context
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
        print(f"✅ {len(docs)} docs, {len(ctx)} characters total (K={top_k}, pool≈{int(pool_k)})")
        return ctx if ctx else "No relevant context found in knowledge base."

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
    print("🧪 Testing RAG retriever (generalized intent + adaptive K + MMR)…")
    try:
        ctx = retrieve_codex_context(
            "First 30 days plan for a new bar manager—stabilize operations and early wins",
            "A newly promoted bar manager in a craft-focused venue with high-volume weekends"
        )
        print(f"✅ Success – {len(ctx)} characters retrieved")
    except Exception as e:
        print(f"❌ Test failed: {e}")