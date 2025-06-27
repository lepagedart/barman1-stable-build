import os
import json
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

def get_openai_client():
    global client
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        client = OpenAI(api_key=api_key)
    return client

@app.route("/", methods=["GET", "POST"])
def index():
    check_and_update_vectorstore(KB_FOLDER)

    if "conversation" not in session:
        session["conversation"] = []

    response_text = ""

    if request.method == "POST":
        venue = request.form.get("venue_concept", "")
        user_prompt = request.form.get("user_prompt", "")

        # Store user prompt
        conversation = session["conversation"]
        conversation.append({"role": "user", "content": user_prompt})

        # RAG context
        rag_context = retrieve_codex_context(user_prompt, venue)

        # Compose message payload
        messages = [
            {"role": "system", "content": open("system_prompt.txt").read()},
            {"role": "system", "content": f"Relevant knowledge base context:\n{rag_context}"}
        ] + conversation

        try:
            openai_client = get_openai_client()
            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            assistant_response = completion.choices[0].message.content.strip()

            # Attempt to parse JSON if response is in JSON format
            try:
                parsed = json.loads(assistant_response)
                response_text = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                response_text = assistant_response

        except Exception as e:
            response_text = f"Error from AI: {str(e)}"

        # Store assistant reply
        conversation.append({"role": "assistant", "content": response_text})
        session["conversation"] = conversation

    return render_template("index.html", conversation=session["conversation"])

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return jsonify({"response": "Conversation reset."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
