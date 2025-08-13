import os
import json
import pickle
import hashlib
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv

from rag_retriever import retrieve_codex_context
from google_search import search_google
from utils import detect_scenario_prompt_mod
from kb_loader import load_knowledge_documents  # (kept import for parity, not used here)

# ---------------------------
# Environment & App Setup
# ---------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

CONVERSATION_CACHE_DIR = "conversation_cache"
KB_FOLDER = "knowledge_base"
os.makedirs(CONVERSATION_CACHE_DIR, exist_ok=True)

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

        # Convert messages to ChatCompletionMessageParam objects as required by OpenAI SDK
        from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

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
                model="gpt-3.5-turbo",
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
        "rag_available": os.path.exists("codex_faiss_index/index.faiss"),
    }), 200

# app.py (add near other routes)
@app.route("/reindex", methods=["POST"])
def reindex():
    try:
        from kb_loader import rebuild_vectorstore  # add function below
        n = rebuild_vectorstore()
        # clear caches so the new index is used immediately
        from rag_retriever import clear_cache
        clear_cache()
        return jsonify({"status": "ok", "docs_indexed": n})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Barman-1 (Lloyd) …")
    print(f"📁 Cache dir: {CONVERSATION_CACHE_DIR}")
    print(f"🔑 API key set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    print(f"📚 RAG index present: {os.path.exists('codex_faiss_index/index.faiss')}")
    app.run(debug=True, use_reloader=False, port=5000)  # ← add use_reloader=False