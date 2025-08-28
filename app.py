import os
import json
import pickle
import hashlib
import threading
import time
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

# 👇 NEW: unified helpers for scenario mods + lenses
from utils import (
    detect_scenario_prompt_mod,
    detect_scenario_prompt_mod_name,
    detect_lens_mods,             # returns (lens_names, lens_blocks_text)
    lens_preamble_for,            # builds preamble text for the matched lenses
)

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

# =============================================================================
#                            Auto-reindex Helpers
# =============================================================================
def _kb_fingerprint() -> str:
    import hashlib as _hashlib
    h = _hashlib.md5()
    root = KB_ROOT
    if not root.exists():
        return ""
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

            # ---- Scenario mod (from KB/system_prompt_mods) ----
            scenario_mod = detect_scenario_prompt_mod(user_prompt, venue) or ""
            selected_scenario_mod_name = detect_scenario_prompt_mod_name(user_prompt, venue) or "none"
            if selected_scenario_mod_name != "none":
                app.logger.info("🧩 Scenario mod selected: %s", selected_scenario_mod_name)

            # ---- Lenses (file-backed triggers; multiple can fire) ----
            lens_names, lens_blocks = detect_lens_mods(user_prompt, venue)
            if lens_names:
                app.logger.info("🔎 Lenses matched: %s", ", ".join(lens_names))
                # Append lens preambles to the *user* prompt
                preambles = lens_preamble_for(lens_names)
                if preambles:
                    user_prompt = f"{user_prompt}\n\n{preambles}"
                    app.logger.info("🧱 Lens preamble(s) injected into user prompt")

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
            blocks = []
            if scenario_mod:
                blocks.append(scenario_mod)
            if lens_blocks:
                blocks.append(lens_blocks)
            combined_prompt = f"{base_prompt}\n\n{'\n\n'.join(blocks)}".strip() if blocks else base_prompt

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

@app.route("/which-mod", methods=["POST"])
def which_mod():
    payload = request.get_json(silent=True) or {}
    user_p = payload.get("user_prompt", "") or ""
    venue_p = payload.get("venue_prompt", "") or ""
    name = detect_scenario_prompt_mod_name(user_p, venue_p) or "none"
    app.logger.info("🧩 /which-mod matched: %s", name)
    return jsonify({"mod": name}), 200

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

# ---------------------------
# Debug: which lenses/preambles would fire?
# ---------------------------
import re

PORTFOLIO_RE = re.compile(r"\bportfolio\b", re.IGNORECASE)
LAYOUT_RE = re.compile(r"\blayout\b", re.IGNORECASE)
EVENT_RE = re.compile(r"\bevent\b", re.IGNORECASE)

@app.route("/which-lenses", methods=["POST"])
def which_lenses():
    """Quick QA endpoint: given a prompt, show which lenses & preambles will trigger."""
    payload = request.get_json(silent=True) or {}
    user_p = (payload.get("user_prompt") or "").strip()
    venue_p = (payload.get("venue_prompt") or "").strip()
    text = f"{user_p} {venue_p}"

    mods = []
    if PORTFOLIO_RE.search(text):
        mods.append("portfolio_lens")
    if LAYOUT_RE.search(text):
        mods.append("layout_lens")
    if EVENT_RE.search(text):
        mods.append("event_lens")

    preambles = []
    if "layout_lens" in mods:
        preambles.append("layout_preamble")
    if "event_lens" in mods:
        preambles.append("event_preamble")

    combined_block = detect_scenario_prompt_mod(user_p, venue_p) or ""

    app.logger.info("🧪 /which-lenses fired → mods=%s, preambles=%s", mods, preambles)
    return jsonify({
        "mods": mods,
        "preambles": preambles,
        "combined_block": combined_block
    }), 200

def _prewarm_backend():
    try:
        from rag_retriever import check_vectorstore_health
        ok, msg = check_vectorstore_health()
        app.logger.info("🔥 Prewarm vectorstore: %s", msg)
        _ = get_openai_client()
        app.logger.info("🔥 Prewarm OpenAI client: ready")
    except Exception as e:
        app.logger.warning("Prewarm failed (will recover on first request): %s", e)
@app.route("/_routes")
def _routes():
    rules = [str(r) for r in app.url_map.iter_rules()]
    app.logger.info("📜 Routes: %s", rules)
    return jsonify({"routes": rules})

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