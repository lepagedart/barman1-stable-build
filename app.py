import os
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
from rag_retriever import retrieve_codex_context, check_and_update_vectorstore

# Flask setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# OpenAI client setup
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Knowledge base folder
KB_FOLDER = "knowledge_base"

@app.route("/", methods=["GET", "POST"])
def index():
    check_and_update_vectorstore(KB_FOLDER)

    if "conversation" not in session:
        session["conversation"] = []

    if request.method == "POST":
        venue = request.form.get("venue_concept", "")
        user_prompt = request.form.get("user_prompt", "")

        # Add user message to conversation history
        conversation = session["conversation"]
        conversation.append({"role": "user", "content": user_prompt})

        # Retrieve RAG context
        rag_context = retrieve_codex_context(user_prompt, venue)

        # Build full messages list for OpenAI
        messages = [
            {"role": "system", "content": open("system_prompt.txt").read()},
            {"role": "system", "content": f"Relevant knowledge base context:\n{rag_context}"}
        ]
        messages.extend(conversation)

        # Get response from OpenAI
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        assistant_response = completion.choices[0].message.content

        # Add assistant response to conversation history
        conversation.append({"role": "assistant", "content": assistant_response})
        session["conversation"] = conversation

        return jsonify({"response": assistant_response})

    return render_template("index.html", conversation=session["conversation"])

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return jsonify({"response": "Conversation reset."})

if __name__ == "__main__":
    app.run(debug=True)