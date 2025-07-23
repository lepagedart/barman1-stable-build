import os
import json
import pickle
import hashlib
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv
from rag_retriever import retrieve_codex_context, check_and_update_vectorstore
from google_search import search_google
from utils import detect_scenario_prompt_mod
from kb_loader import load_knowledge_documents

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

client = None
KB_FOLDER = "knowledge_base"
CONVERSATION_CACHE_DIR = "conversation_cache"

os.makedirs(CONVERSATION_CACHE_DIR, exist_ok=True)

def get_openai_client():
    global client
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        client = OpenAI(api_key=api_key)
    return client

def get_conversation_id():
    if "conversation_id" not in session:
        session["conversation_id"] = hashlib.md5(
            f"{session.get('_id', '')}{os.urandom(16).hex()}".encode()
        ).hexdigest()[:16]
    return session["conversation_id"]

def save_conversation(conversation_id, conversation):
    try:
        cache_file = os.path.join(CONVERSATION_CACHE_DIR, f"{conversation_id}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(conversation, f)
    except Exception as e:
        print(f"⚠️ Could not save conversation: {e}")

def load_conversation(conversation_id):
    try:
        cache_file = os.path.join(CONVERSATION_CACHE_DIR, f"{conversation_id}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Could not load conversation: {e}")
    return []

def load_system_prompt():
    try:
        with open("system_prompt.txt", "r") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Could not load system prompt: {e}")
        return "You are Barman-1, an AI-powered Bar Director providing expert bar program advice."

@app.route("/", methods=["GET", "POST"])
def index():
    # 🧠 Reload knowledge docs at runtime (optional — currently for diagnostics)
    docs = load_knowledge_documents("knowledge_base")
    print(f"📚 Loaded {len(docs)} KB docs")

    print(f"🔄 Processing {request.method} request...")
    conversation_id = get_conversation_id()
    conversation = load_conversation(conversation_id)
    use_live_search = False
    venue = ""
    user_prompt = ""
    assistant_response = ""

    if request.method == "POST":
        venue = request.form.get("venue_concept", "")
        user_prompt = request.form.get("user_prompt", "")
        use_live_search = request.form.get("use_live_search") == "on"

        print(f"🔄 User prompt received — venue='{venue}', live_search={use_live_search}")
        conversation.append({"role": "user", "content": user_prompt})

        # 🔍 Retrieve RAG context
        try:
            rag_context = retrieve_codex_context(user_prompt, venue, use_live_search=use_live_search)
            print(f"✅ RAG context loaded: {len(rag_context)} chars")
        except Exception as e:
            print(f"⚠️ RAG context error: {e}")
            rag_context = "RAG context temporarily unavailable."

        # 🌐 Run live search if enabled
        search_snippets = ""
        if use_live_search:
            print("🔎 Running Google search...")
            search_snippets = search_google(user_prompt)

        structured_context = (
            f"[Knowledge Base Insights]\n{rag_context}\n\n"
            f"[Live Internet Results]\n{search_snippets}"
        )

        # 🧠 Add scenario-specific mod
        base_prompt = load_system_prompt()
        scenario_mod = detect_scenario_prompt_mod(user_prompt)
        combined_prompt = base_prompt + "\n\n" + scenario_mod

        # 🗨️ Assemble full message list
        messages = [
            {"role": "system", "content": combined_prompt},
            {"role": "system", "content": structured_context}
        ]
        messages.extend(conversation[-10:])  # recent conversation only

        # 🚀 Send to OpenAI
        openai_client = get_openai_client()
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1200,
                temperature=0.7
            )
            assistant_response = completion.choices[0].message.content.strip()
            print(f"✅ Assistant response: {len(assistant_response)} chars")

            try:
                parsed = json.loads(assistant_response)
                response_text = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                response_text = assistant_response

        except Exception as e:
            print(f"❌ OpenAI error: {type(e).__name__}: {e}")
            response_text = f"Error from AI: {str(e)}"

        # 💾 Save response and return
        conversation.append({"role": "assistant", "content": response_text})
        save_conversation(conversation_id, conversation)
        return jsonify({"response": response_text})

    return render_template(
        "index.html",
        conversation=conversation,
        venue=venue,
        user_prompt=user_prompt,
        assistant_response=assistant_response,
        use_live_search=use_live_search
    )

@app.route("/reset", methods=["POST"])
def reset():
    conversation_id = get_conversation_id()
    try:
        cache_file = os.path.join(CONVERSATION_CACHE_DIR, f"{conversation_id}.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
    except Exception as e:
        print(f"⚠️ Could not clear cache: {e}")
    session.clear()
    return jsonify({"response": "Conversation reset."})

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "rag_available": os.path.exists("codex_faiss_index/index.faiss")
    })

if __name__ == "__main__":
    print("🚀 Starting Barman-1 AI Bar Director...")
    print(f"📁 Cache dir: {CONVERSATION_CACHE_DIR}")
    print(f"🔑 API key set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    print(f"📚 RAG index present: {os.path.exists('codex_faiss_index/index.faiss')}")
    app.run(debug=True, port=5000)