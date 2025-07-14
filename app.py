import os
import json
import pickle
import hashlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv
from rag_retriever import retrieve_codex_context, check_and_update_vectorstore

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

# OpenAI client setup
client = None
KB_FOLDER = "knowledge_base"
CONVERSATION_CACHE_DIR = "conversation_cache"

# Create cache directory if it doesn't exist
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
    """Save conversation with a meaningful filename (based on first user prompt)"""
    try:
        first_prompt = next((msg["content"] for msg in conversation if msg["role"] == "user"), "untitled")
        safe_name = first_prompt.strip().splitlines()[0][:50].replace(" ", "_").replace(":", "").replace("/", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        cache_file = os.path.join(CONVERSATION_CACHE_DIR, f"{timestamp}_{safe_name}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(conversation, f)
        print(f"💾 Conversation saved to: {cache_file}")
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

    if request.method == "POST":
        venue = request.form.get("venue_concept", "")
        user_prompt = request.form.get("user_prompt", "")
        
        print(f"🔄 Processing request: venue='{venue}', prompt='{user_prompt}'")
        conversation.append({"role": "user", "content": user_prompt})

        try:
            print(f"🔄 Retrieving RAG context...")
            try:
                rag_context = retrieve_codex_context(user_prompt, venue)
                print(f"✅ RAG context:\n{rag_context}")
            except Exception as e:
                print(f"⚠️  RAG context failed, using fallback: {e}")
                rag_context = "RAG context temporarily unavailable."

            system_prompt = load_system_prompt()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Venue Type: {venue}\nRelevant knowledge base context:\n{rag_context}"}
            ]
            recent_conversation = conversation[-10:]
            messages.extend(recent_conversation)

            print(f"🔄 Getting OpenAI client...")
            openai_client = get_openai_client()
            print(f"✅ OpenAI client ready")
            print(f"🔄 Making API call...")
            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )

            assistant_response = completion.choices[0].message.content.strip()
            print(f"✅ OpenAI response received: {len(assistant_response)} characters")

            try:
                parsed = json.loads(assistant_response)
                response_text = json.dumps(parsed, indent=2)
                print("📋 Response parsed as JSON")
            except json.JSONDecodeError:
                response_text = assistant_response
                print("📄 Response kept as plain text")

        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
            response_text = f"Error from AI: {str(e)}"

        conversation.append({"role": "assistant", "content": response_text})
        save_conversation(conversation_id, conversation)
        return jsonify({"response": response_text})

    return render_template("index.html", conversation=conversation)

@app.route("/reset", methods=["POST"])
def reset():
    """Reset the session and clear server-side state"""
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
    print(f"📁 Conversation cache directory: {CONVERSATION_CACHE_DIR}")
    print(f"🔑 OpenAI API key configured: {bool(os.environ.get('OPENAI_API_KEY'))}")
    print(f"📚 RAG index available: {os.path.exists('codex_faiss_index/index.faiss')}")
    app.run(debug=True, port=5000)