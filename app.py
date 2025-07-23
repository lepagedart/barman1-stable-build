import os
import json
import pickle
import hashlib
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv
from rag_retriever import retrieve_codex_context, check_and_update_vectorstore
from google_search import search_google
from utils import detect_scenario_prompt_mod  # ✅ NEW IMPORT

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
        print(f"Warning: Could not save conversation: {e}")

def load_conversation(conversation_id):
    try:
        cache_file = os.path.join(CONVERSATION_CACHE_DIR, f"{conversation_id}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load conversation: {e}")
    return []

def load_system_prompt():
    try:
        with open("system_prompt.txt", "r") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load system prompt: {e}")
        return "You are Barman-1, an AI-powered Bar Director providing expert bar program advice."

@app.route("/", methods=["GET", "POST"])
def index():
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

        print(f"🔄 Request: venue='{venue}', prompt='{user_prompt}', live_search={use_live_search}")
        conversation.append({"role": "user", "content": user_prompt})

        # 🔍 RAG Context
        print("🔄 Retrieving RAG context...")
        try:
            rag_context = retrieve_codex_context(user_prompt, venue, use_live_search=use_live_search)
            print(f"✅ RAG context: {len(rag_context)} chars")
        except Exception as e:
            print(f"⚠️  RAG context failed: {e}")
            rag_context = "RAG context temporarily unavailable."

        # 🌐 Live Search Context
        search_snippets = ""
        if use_live_search:
            print("🔎 Performing live Google search...")
            search_snippets = search_google(user_prompt)

        structured_context = (
            f"[Knowledge Base Insights]\n{rag_context}\n\n"
            f"[Live Internet Results]\n{search_snippets}"
        )

        # 🧠 Build system prompt with scenario mod if available
        base_prompt = load_system_prompt()
        scenario_mod = detect_scenario_prompt_mod(user_prompt)
        combined_prompt = base_prompt + "\n\n" + scenario_mod

        # 🗨️ Assemble messages
        messages = [
            {"role": "system", "content": combined_prompt},
            {"role": "system", "content": structured_context}
        ]

        recent_conversation = conversation[-10:]
        messages.extend(recent_conversation)

        print("🔄 Calling OpenAI...")
        openai_client = get_openai_client()

        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1200,
                temperature=0.7
            )

            assistant_response = completion.choices[0].message.content.strip()
            print(f"✅ Response: {len(assistant_response)} chars")

            try:
                parsed = json.loads(assistant_response)
                response_text = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                response_text = assistant_response

        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
            response_text = f"Error from AI: {str(e)}"

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
        print(f"Warning: Could not remove cache file: {e}")
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