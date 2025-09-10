import os
import json
import pickle
import hashlib
import threading
import time
import re
from pathlib import Path

from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from dotenv import load_dotenv

from rag_retriever import retrieve_codex_context
from google_search import search_google
from kb_loader import load_knowledge_documents  # retained for parity, not used

# ---------------------------
# Environment & App Setup
# ---------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

CONVERSATION_CACHE_DIR = "conversation_cache"
KB_FOLDER = "knowledge_base"
VECTORSTORE_DIR = "codex_faiss_index"
os.makedirs(CONVERSATION_CACHE_DIR, exist_ok=True)

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# ---- Auto-reindex config ----
AUTO_REINDEX = os.environ.get("AUTO_REINDEX", "1") != "0"
_KB_CHECK_MIN_INTERVAL_SEC = 3.0
_last_kb_check_ts = 0.0
_last_kb_fingerprint = None
_kb_lock = threading.Lock()
KB_ROOT = Path(KB_FOLDER)

_client = None

# =============================================================================
#                 LENS / PREAMBLE LOADERS & REGEX TRIGGER HELPERS
# =============================================================================
LENS_DIR = Path("system_prompt_mods") / "lenses"
TRIGGERS_DIR = Path("system_prompt_mods") / "triggers"
PREAMBLES_DIR = Path("system_prompt_mods") / "preambles"

def _load_text(path: Path, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return fallback.strip()

def _load_triggers(path: Path) -> re.Pattern:
    """
    Load line-separated regex OR literal tokens (joined by '|').
    Lines beginning with '#' are comments. Empty lines ignored.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        pats = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        return re.compile("|".join(pats), re.IGNORECASE) if pats else re.compile("$^")
    except Exception as e:
        app.logger.warning("⚠️ Could not load triggers from %s: %s", path, e)
        return re.compile("$^")

# --------- Load lens blocks (system addenda) ----------
PORTFOLIO_LENS_BLOCK   = _load_text(LENS_DIR / "portfolio_lens.txt",   "--- Portfolio Lens (fallback) ---")
LAYOUT_LENS_BLOCK      = _load_text(LENS_DIR / "layout_lens.txt",      "--- Layout Lens (fallback) ---")
EVENT_LENS_BLOCK       = _load_text(LENS_DIR / "event_setup_lens.txt", "--- Event Lens (fallback) ---")

GUEST_EXP_LENS_BLOCK   = _load_text(LENS_DIR / "guest_experience_lens.txt",   "--- Guest Experience Lens (fallback) ---")
FINANCIALS_LENS_BLOCK  = _load_text(LENS_DIR / "financials_lens.txt",         "--- Financials Lens (fallback) ---")
WOW_FACTOR_LENS_BLOCK  = _load_text(LENS_DIR / "wow_factor_lens.txt",         "--- Wow Factor Lens (fallback) ---")
COMPLIANCE_LENS_BLOCK  = _load_text(LENS_DIR / "compliance_lens.txt",         "--- Compliance Lens (fallback) ---")
TECH_LENS_BLOCK        = _load_text(LENS_DIR / "technology_lens.txt",         "--- Technology Lens (fallback) ---")
CREATIVITY_FEASIBILITY_LENS_BLOCK = _load_text(LENS_DIR / "creativity_vs_feasibility_lens.txt","--- Creativity vs Feasibility Lens (fallback) ---")


# --------- Load preambles (appended to user prompt) ----------
LAYOUT_PREAMBLE   = _load_text(PREAMBLES_DIR / "layout_preamble.txt",   "Provide ASCII map, reach zones, rail loadout, build path, KPIs.")
EVENT_PREAMBLE    = _load_text(PREAMBLES_DIR / "event_preamble.txt",    "Provide throughput targets, batching math, ice/mixer totals, 5-step checklist.")
GUEST_PREAMBLE    = _load_text(PREAMBLES_DIR / "guest_experience_preamble.txt", "Add recovery gestures, consistency checks, staff phrasing.")
FIN_PREAMBLE      = _load_text(PREAMBLES_DIR / "financials_preamble.txt",       "Add pour-cost table, margin math, waste controls, price bands.")
WOW_PREAMBLE      = _load_text(PREAMBLES_DIR / "wow_factor_preamble.txt",       "Design a single memorable moment with fast garnishes + script.")
COMP_PREAMBLE     = _load_text(PREAMBLES_DIR / "compliance_preamble.txt",       "Add ID workflow, refusal protocol, allergy labeling, incident log.")
TECH_PREAMBLE     = _load_text(PREAMBLES_DIR / "technology_preamble.txt",       "Map POS/inventory integrations, dashboards, and forecasting cadence.")
CREATIVITY_FEASIBILITY_PREAMBLE = _load_text(
    PREAMBLES_DIR / "creativity_vs_feasibility_preamble.txt",
    "--- Creativity vs Feasibility Preamble (fallback) ---"
)
VENUE_SCOPE_PREAMBLE = _load_text(
    PREAMBLES_DIR / "venue_scope_preamble.txt",
    "Ground recommendations in venue size/type. Ask fact-finding questions if details are missing."
)
# --------- Global system addendum (always appended to system prompt) ----------
GLOBAL_GUARDRAILS = _load_text(Path("system_prompt_mods") / "global_guardrails.txt",
    "Ground advice in venue type/size/staff; ask ≤3 clarifying questions if missing; scale scope; call out trade-offs.")

# NEW: consultative questions preamble (always appended)
CONSULT_PREAMBLE  = _load_text(
    PREAMBLES_DIR / "consultative_preamble.txt",
    (
        "Before recommending, list the 5–10 most critical clarifying questions. "
        "If answers are unavailable, proceed anyway: state explicit assumptions "
        "up front and continue with a complete, actionable plan that notes where "
        "assumptions were made."
    ),
)

# --------- Load regex triggers ----------
PORTFOLIO_RE   = _load_triggers(TRIGGERS_DIR / "portfolio_triggers.txt")
LAYOUT_RE      = _load_triggers(TRIGGERS_DIR / "layout_triggers.txt")
EVENT_RE       = _load_triggers(TRIGGERS_DIR / "event_triggers.txt")
GUEST_EXP_RE   = _load_triggers(TRIGGERS_DIR / "guest_experience_triggers.txt")
FINANCIALS_RE  = _load_triggers(TRIGGERS_DIR / "financials_triggers.txt")
WOW_FACTOR_RE  = _load_triggers(TRIGGERS_DIR / "wow_factor_triggers.txt")
COMPLIANCE_RE  = _load_triggers(TRIGGERS_DIR / "compliance_triggers.txt")
TECH_RE        = _load_triggers(TRIGGERS_DIR / "technology_triggers.txt")
CREATIVITY_FEASIBILITY_RE = _load_triggers(TRIGGERS_DIR / "creativity_vs_feasibility_triggers.txt")

def _any_match(regex: re.Pattern, user_prompt: str, venue: str) -> bool:
    text = f"{user_prompt or ''} {venue or ''}"
    return bool(regex.search(text))

def detect_scenario_prompt_mod(user_prompt: str, venue: str) -> str | None:
    """Return concatenated lens blocks when multiple lenses apply."""
    blocks = []
    if _any_match(PORTFOLIO_RE, user_prompt, venue):  blocks.append(PORTFOLIO_LENS_BLOCK)
    if _any_match(LAYOUT_RE, user_prompt, venue):     blocks.append(LAYOUT_LENS_BLOCK)
    if _any_match(EVENT_RE, user_prompt, venue):      blocks.append(EVENT_LENS_BLOCK)
    if _any_match(GUEST_EXP_RE, user_prompt, venue):  blocks.append(GUEST_EXP_LENS_BLOCK)
    if _any_match(FINANCIALS_RE, user_prompt, venue): blocks.append(FINANCIALS_LENS_BLOCK)
    if _any_match(WOW_FACTOR_RE, user_prompt, venue): blocks.append(WOW_FACTOR_LENS_BLOCK)
    if _any_match(COMPLIANCE_RE, user_prompt, venue): blocks.append(COMPLIANCE_LENS_BLOCK)
    if _any_match(TECH_RE, user_prompt, venue):       blocks.append(TECH_LENS_BLOCK)
    if _any_match(CREATIVITY_FEASIBILITY_RE, user_prompt, venue):blocks.append(CREATIVITY_FEASIBILITY_LENS_BLOCK)
    return "\n\n".join(blocks) if blocks else None

def detect_scenario_prompt_mod_name(user_prompt: str, venue: str) -> str:
    names = []
    if _any_match(PORTFOLIO_RE, user_prompt, venue):  names.append("portfolio_lens")
    if _any_match(LAYOUT_RE, user_prompt, venue):     names.append("layout_lens")
    if _any_match(EVENT_RE, user_prompt, venue):      names.append("event_lens")
    if _any_match(GUEST_EXP_RE, user_prompt, venue):  names.append("guest_experience_lens")
    if _any_match(FINANCIALS_RE, user_prompt, venue): names.append("financials_lens")
    if _any_match(WOW_FACTOR_RE, user_prompt, venue): names.append("wow_factor_lens")
    if _any_match(COMPLIANCE_RE, user_prompt, venue): names.append("compliance_lens")
    if _any_match(TECH_RE, user_prompt, venue):       names.append("technology_lens")
    if _any_match(CREATIVITY_FEASIBILITY_RE, user_prompt, venue): names.append("creativity_vs_feasibility_lens")
    return "+".join(names) if names else "none"


def _preambles_for(names: str) -> list[tuple[str, str]]:
    mapping = {
        "layout_lens": ("layout_preamble", LAYOUT_PREAMBLE),
        "event_lens": ("event_preamble", EVENT_PREAMBLE),
        "guest_experience_lens": ("guest_experience_preamble", GUEST_PREAMBLE),
        "financials_lens": ("financials_preamble", FIN_PREAMBLE),
        "wow_factor_lens": ("wow_factor_preamble", WOW_PREAMBLE),
        "compliance_lens": ("compliance_preamble", COMP_PREAMBLE),
        "technology_lens": ("technology_preamble", TECH_PREAMBLE),
        # 🔥 Always-on venue scope anchor
        "venue_scope": ("venue_scope_preamble", VENUE_SCOPE_PREAMBLE),
        "creativity_vs_feasibility_lens": ("creativity_vs_feasibility_preamble", CREATIVITY_FEASIBILITY_PREAMBLE),
    }
    out = []
    if names and names != "none":
        for key, val in mapping.items():
            if key in names:
                out.append(val)
    # 🔥 Force-append venue_scope_preamble regardless of lens
    out.append(("venue_scope_preamble", VENUE_SCOPE_PREAMBLE))
    return out   
    """Return [(name, text), ...] for preambles implied by lens names."""
    mapping = {
        "layout_lens": ( "layout_preamble", LAYOUT_PREAMBLE ),
        "event_lens": ( "event_preamble", EVENT_PREAMBLE ),
        "guest_experience_lens": ( "guest_experience_preamble", GUEST_PREAMBLE ),
        "financials_lens": ( "financials_preamble", FIN_PREAMBLE ),
        "wow_factor_lens": ( "wow_factor_preamble", WOW_PREAMBLE ),
        "compliance_lens": ( "compliance_preamble", COMP_PREAMBLE ),
        "technology_lens": ( "technology_preamble", TECH_PREAMBLE ),
    }
    out = []
    if names and names != "none":
        for key, val in mapping.items():
            if key in names:
                out.append(val)
    return out

# =============================================================================
#                               OpenAI Client
# =============================================================================
def get_openai_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        _client = OpenAI(api_key=api_key)
    return _client

# =============================================================================
#                         Conversation Cache Helpers
# =============================================================================
def get_conversation_id():
    if "_id" not in session:
        session["_id"] = os.urandom(16).hex()
    if "conversation_id" not in session:
        seed = f"{session['_id']}{os.urandom(16).hex()}".encode()
        session["conversation_id"] = hashlib.md5(seed).hexdigest()[:16]
    return session["conversation_id"]

def _cache_path(conversation_id: str) -> str:
    return os.path.join(CONVERSATION_CACHE_DIR, f"{conversation_id}.pkl")

def save_conversation(conversation_id, conversation):
    try:
        with open(_cache_path(conversation_id), "wb") as f:
            pickle.dump(conversation, f)
    except Exception as e:
        app.logger.warning("Could not save conversation: %s", e)

def load_conversation(conversation_id):
    try:
        path = _cache_path(conversation_id)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        app.logger.warning("Could not load conversation: %s", e)
    return []

def load_system_prompt():
    try:
        return Path("system_prompt.txt").read_text(encoding="utf-8")
    except Exception as e:
        app.logger.warning("Could not load system prompt: %s", e)
        return "You are Lloyd, an AI-powered Bar Director providing expert bar program advice."

# Small system rule to enforce “assume & proceed” when answers aren’t available
ASSUMPTIONS_RULE = (
    "If clarifying details are missing, list targeted questions first. "
    "When answers are unavailable, state explicit assumptions and proceed with a complete, "
    "actionable plan that flags assumption-sensitive steps."
)

# =============================================================================
#                            Auto-reindex Helpers
# =============================================================================
def _kb_fingerprint() -> str:
    import hashlib as _hashlib
    h = _hashlib.md5()
    root = KB_ROOT
    if not root.exists(): return ""
    for p in sorted(root.rglob("*.txt")):
        try:
            stat = p.stat()
            h.update(str(p.relative_to(root)).encode("utf-8"))
            h.update(str(int(stat.st_mtime)).encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
        except Exception:
            continue
    return h.hexdigest()

def _reindex_now() -> int:
    from kb_loader import rebuild_vectorstore
    from rag_retriever import clear_cache
    app.logger.info("🔁 Reindex: rebuilding vectorstore…")
    n = rebuild_vectorstore()
    clear_cache()
    app.logger.info("✅ Reindex complete. Chunks indexed: %s", n)
    return n

def _maybe_reindex_on_change():
    global _last_kb_check_ts, _last_kb_fingerprint
    if not AUTO_REINDEX:
        return
    now = time.time()
    if (now - _last_kb_check_ts) < _KB_CHECK_MIN_INTERVAL_SEC:
        return
    with _kb_lock:
        _last_kb_check_ts = now
        fp = _kb_fingerprint()
        if fp and fp != _last_kb_fingerprint:
            app.logger.info("🧭 KB change detected, triggering reindex…")
            _reindex_now()
            _last_kb_fingerprint = fp

@app.before_request
def _auto_reindex_hook():
    _maybe_reindex_on_change()

# =============================================================================
#                                      Routes
# =============================================================================
@app.route("/", methods=["GET", "POST"])
def index():
    app.logger.info("🔄 Processing %s request...", request.method)
    conversation_id = get_conversation_id()
    conversation = load_conversation(conversation_id)

    venue = ""
    user_prompt = ""
    use_live_search = False
    assistant_response = ""

    if request.method == "POST":
        payload = {}
        if request.is_json:
            try:
                payload = request.get_json(silent=True) or {}
            except Exception:
                payload = {}
        form = request.form or {}

        venue = (payload.get("venue_concept") or form.get("venue_concept") or "").strip()
        user_prompt = (payload.get("user_prompt") or form.get("user_prompt") or "").strip()
        raw_use_live = payload.get("use_live_search", form.get("use_live_search"))
        use_live_search = ((str(raw_use_live).lower() in {"true", "1", "on", "yes"}) if raw_use_live is not None else False)

        app.logger.info("🧾 User prompt received — venue='%s', live_search=%s", venue, use_live_search)
        if user_prompt:
            conversation.append({"role": "user", "content": user_prompt})

            # ---- Lens detection ----
            scenario_mod = detect_scenario_prompt_mod(user_prompt, venue) or ""
            selected_mod_name = detect_scenario_prompt_mod_name(user_prompt, venue) or "none"
            app.logger.info("🧩 Prompt mod selected: %s", selected_mod_name)

            # ---- Auto-append preambles to the user prompt based on active lenses ----
            preambles = _preambles_for(selected_mod_name)
            for pre_name, pre_text in preambles:
                pass  # No operation, just iterating (can be removed if unnecessary)
            # Always-on venue scope preamble (prevents overreach & forces scaling)
            if VENUE_SCOPE_PREAMBLE:
                preambles.append(("venue_scope_preamble", VENUE_SCOPE_PREAMBLE))
               
            for pre_name, pre_text in preambles:
                user_prompt = f"{user_prompt}\n\n{pre_text}"
                app.logger.info("📝 Preamble injected: %s", pre_name)
            # ---- ALWAYS append consultative preamble ----
            if CONSULT_PREAMBLE:
                user_prompt = f"{user_prompt}\n\n{CONSULT_PREAMBLE}"
                app.logger.info("🗣️  Preamble injected: consultative_preamble")

            # ---------- RAG context ----------
            try:
                rag_context = retrieve_codex_context(user_prompt, venue, use_live_search=use_live_search)
                app.logger.info("📚 RAG context length: %d chars", len(rag_context or ""))
            except Exception as e:
                app.logger.warning("RAG context error: %s", e)
                rag_context = "RAG context temporarily unavailable."

            # ---------- Optional live search ----------
            search_snippets = ""
            if use_live_search:
                try:
                    app.logger.info("🔎 Running Google search…")
                    search_snippets = search_google(user_prompt) or ""
                except Exception as e:
                    app.logger.warning("Live search error: %s", e)
                    search_snippets = ""

            structured_context = (
                f"[Knowledge Base Insights]\n{rag_context or ''}\n\n"
                f"[Live Internet Results]\n{search_snippets or ''}"
            )

            # ---------- Compose messages ----------
            base_prompt = load_system_prompt()
            # include the assumptions rule before any lens blocks
            combined_prompt = base_prompt
            if ASSUMPTIONS_RULE:
                combined_prompt = f"{combined_prompt}\n\n{ASSUMPTIONS_RULE}"
            if scenario_mod:
                combined_prompt = f"{combined_prompt}\n\n{scenario_mod}"
            if GLOBAL_GUARDRAILS:
                combined_prompt += "\n\n" + GLOBAL_GUARDRAILS
            combined_prompt = combined_prompt.strip()

            messages = [
                {"role": "system", "content": combined_prompt},
                {"role": "system", "content": structured_context},
            ]
            messages.extend(conversation[-10:])

            def to_message_param(msg):
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    return ChatCompletionSystemMessageParam(role=role, content=content)
                elif role == "user":
                    return ChatCompletionUserMessageParam(role=role, content=content)
                elif role == "assistant":
                    return ChatCompletionAssistantMessageParam(role=role, content=content)
                else:
                    raise ValueError(f"Unknown role: {role}")

            messages_param = [to_message_param(msg) for msg in messages]

            # ---------- OpenAI call ----------
            response_text = ""
            try:
                client = get_openai_client()
                completion = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages_param,
                    max_tokens=1200,
                    temperature=0.7,
                )
                content = completion.choices[0].message.content
                assistant_response = (content or "").strip()

                try:
                    parsed = json.loads(assistant_response)
                    response_text = json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    response_text = assistant_response

                app.logger.info("✅ Assistant response length: %d chars", len(response_text))
            except Exception as e:
                app.logger.error("❌ OpenAI error: %s: %s", type(e).__name__, e)
                response_text = f"Error from AI: {str(e)}"

            # ---------- Save + return ----------
            conversation.append({"role": "assistant", "content": response_text})
            save_conversation(conversation_id, conversation)

            payload = {
                "ok": True,
                "assistant_response": str(response_text or ""),
                "response": str(response_text or ""),
                "conversation_id": conversation_id,
                "sources": [],
            }
            app.logger.info("↩️ Returning assistant_response length: %d", len(payload["assistant_response"]))
            return jsonify(payload), 200

    # GET
    return render_template(
        "index.html",
        conversation=conversation,
        venue=venue,
        user_prompt=user_prompt,
        assistant_response=assistant_response,
        use_live_search=use_live_search,
    )

# ------------------ Debug: which lenses fired ------------------
@app.route("/which-lenses", methods=["POST"])
def which_lenses():
    payload = request.get_json(silent=True) or {}
    user_p = (payload.get("user_prompt") or "").strip()
    venue_p = (payload.get("venue_prompt") or "").strip()

    names = detect_scenario_prompt_mod_name(user_p, venue_p)
    mod_block = detect_scenario_prompt_mod(user_p, venue_p) or ""
    preambles = _preambles_for(names)
    return jsonify({
        "mods": names.split("+") if names and names != "none" else [],
        "mod_text": mod_block,
        "preambles": [n for (n, _t) in preambles],
        "preamble_texts": [t for (_n, t) in preambles],
    }), 200

@app.route("/init", methods=["POST"])
def init():
    cid = get_conversation_id()
    if not load_conversation(cid):
        save_conversation(cid, [])
    return jsonify({"ok": True, "conversation_id": cid}), 200

@app.route("/reset", methods=["POST"])
def reset():
    cid = get_conversation_id()
    try:
        path = _cache_path(cid)
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        app.logger.warning("Could not clear cache: %s", e)
    session.clear()
    return jsonify({"ok": True, "assistant_response": "Conversation reset.", "conversation_id": None}), 200

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "rag_available": os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")),
        "auto_reindex": AUTO_REINDEX,
        "model": OPENAI_MODEL,
    }), 200

@app.route("/_routes")
def _routes():
    rules = [str(r) for r in app.url_map.iter_rules()]
    app.logger.info("📜 Routes: %s", rules)
    return jsonify({"routes": rules})

@app.route("/reindex", methods=["POST"])
def reindex():
    try:
        with _kb_lock:
            n = _reindex_now()
            global _last_kb_fingerprint
            _last_kb_fingerprint = _kb_fingerprint()
        return jsonify({"status": "ok", "chunks_indexed": n})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

def _prewarm_backend():
    try:
        from rag_retriever import check_vectorstore_health
        ok, msg = check_vectorstore_health()
        app.logger.info("🔥 Prewarm vectorstore: %s", msg)
        _ = get_openai_client()
        app.logger.info("🔥 Prewarm OpenAI client: ready")
    except Exception as e:
        app.logger.warning("Prewarm failed (will recover on first request): %s", e)

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Barman-1 (Lloyd) …")
    print(f"📁 Cache dir: {CONVERSATION_CACHE_DIR}")
    print(f"🔑 API key set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    print(f"📚 RAG index present: {os.path.exists(os.path.join(VECTORSTORE_DIR, 'index.faiss'))}")
    threading.Thread(target=_prewarm_backend, daemon=True).start()
    try:
        _last_kb_fingerprint = _kb_fingerprint()
        print(f"🧾 KB fingerprint: {_last_kb_fingerprint[:8]}…")
    except Exception as e:
        print(f"⚠️ Could not compute KB fingerprint: {e}")
    app.run(debug=True, use_reloader=False, port=5000)