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
from utils import detect_scenario_prompt_mod
from kb_loader import load_knowledge_documents  # (kept import for parity, not used here)
import threading

from utils import (
    detect_scenario_prompt_mod,
    detect_scenario_prompt_mod_name,
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

# Model can be overridden via env
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# ---- Auto-reindex config ----
AUTO_REINDEX = os.environ.get("AUTO_REINDEX", "1") != "0"
_KB_CHECK_MIN_INTERVAL_SEC = 3.0
_last_kb_check_ts = 0.0
_last_kb_fingerprint = None
_kb_lock = threading.Lock()
KB_ROOT = Path(KB_FOLDER)

_client = None


def get_openai_client():
    """Lazily instantiate OpenAI client with an explicit API key check."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        _client = OpenAI(api_key=api_key)
    return _client


def get_conversation_id():
    """Create a stable per-session conversation id."""
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
        with open("system_prompt.txt", "r") as f:
            return f.read()
    except Exception as e:
        app.logger.warning("Could not load system prompt: %s", e)
        return (
            "You are Lloyd, an AI-powered Bar Director providing expert bar program advice."
        )

# ---------------------------
# Auto-reindex helpers
# ---------------------------
def _kb_fingerprint() -> str:
    """
    Create a quick fingerprint of all *.txt under knowledge_base/.
    Uses file paths + mtimes + sizes; fast and reliable enough.
    """
    import hashlib as _hashlib
    h = _hashlib.md5()
    if not KB_ROOT.exists():
        return ""
    for p in sorted(KB_ROOT.rglob("*.txt")):
        try:
            stat = p.stat()
            h.update(str(p.relative_to(KB_ROOT)).encode("utf-8"))
            h.update(str(int(stat.st_mtime)).encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
        except Exception:
            continue
    return h.hexdigest()


def _reindex_now() -> int:
    """Rebuild FAISS and clear retriever caches."""
    from kb_loader import rebuild_vectorstore
    from rag_retriever import clear_cache
    app.logger.info("🔁 Reindex: rebuilding vectorstore…")
    n = rebuild_vectorstore()
    clear_cache()
    app.logger.info("✅ Reindex complete. Chunks indexed: %s", n)
    return n


def _maybe_reindex_on_change():
    """
    If AUTO_REINDEX is enabled, compare KB fingerprint and rebuild when changed.
    Throttled to avoid multiple rebuilds within a few seconds.
    """
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
    # Rebuild if KB changed since last request (cheap check + throttle)
    _maybe_reindex_on_change()


# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    app.logger.info("🔄 Processing %s request...", request.method)
    conversation_id = get_conversation_id()
    conversation = load_conversation(conversation_id)

    # Defaults so Jinja never gets None
    venue = ""
    user_prompt = ""
    use_live_search = False
    assistant_response = ""

    if request.method == "POST":
        # Accept either form-encoded or JSON requests defensively
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
        use_live_search = (
            (str(raw_use_live).lower() in {"true", "1", "on", "yes"})
            if raw_use_live is not None else False
        )

        app.logger.info("🧾 User prompt received — venue='%s', live_search=%s", venue, use_live_search)
        if user_prompt:
            conversation.append({"role": "user", "content": user_prompt})

            # ---- Scenario system-prompt mod detection (auto) ----
            scenario_mod = detect_scenario_prompt_mod(user_prompt, venue) or ""
            selected_mod_name = detect_scenario_prompt_mod_name(user_prompt, venue) or "none"
            app.logger.info("🧩 Prompt mod selected: %s", selected_mod_name)

# Compose the full system message
            base_prompt = load_system_prompt()
            if scenario_mod:
                combined_prompt = f"{base_prompt}\n\n{scenario_mod}".strip()
            else:
                combined_prompt = base_prompt

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
        scenario_mod = detect_scenario_prompt_mod(user_prompt) or ""
        combined_prompt = f"{base_prompt}\n\n{scenario_mod}".strip()

        messages = [
            {"role": "system", "content": combined_prompt},
            {"role": "system", "content": structured_context},
        ]
        # Append a small window of recent chat history
        messages.extend(conversation[-10:])

        # Convert messages to OpenAI message params
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

            # If model returned JSON, pretty print; otherwise keep as-is
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

        # ALWAYS return the same shape so the front-end never guesses
        payload = {
            "ok": True,
            "assistant_response": str(response_text or ""),
            "response": str(response_text or ""),  # backward-compat if frontend used data.response
            "conversation_id": conversation_id,
            "sources": [],  # populate later if you want to surface citations
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
    """Tiny pre-warm route to create a session/conversation id on page load."""
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
    user_p = payload.get("user_prompt", "")
    venue_p = payload.get("venue_prompt", "")
    name = detect_scenario_prompt_mod_name(user_p, venue_p) or "none"
    # in app.py inside which_mod()
    print(f"🧩 /which-mod matched:", os.name)
    return jsonify({"mod": name}), 200

@app.route("/reindex", methods=["POST"])
def reindex():
    try:
        with _kb_lock:
            n = _reindex_now()
            # refresh stored fingerprint so we don't rebuild again on the next request
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
        app.logger.warning("Prewarm failed (will recover on first request): %s", e)# ---------------------------
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