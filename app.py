import os
from flask import Flask, render_template, request, session, redirect, url_for
from openai import OpenAI
from rag_retriever import retrieve_codex_context, check_and_update_vectorstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your-default-secret-key")

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Knowledge base folder
KB_FOLDER = "knowledge_base"

# Load system prompt from external file
with open("system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

@app.route("/", methods=["GET", "POST"])
def index():
    if "conversation" not in session:
        session["conversation"] = []

    conversation = session["conversation"]
    ai_response = None

    if request.method == "POST":
        venue_concept = request.form.get("venue_concept", "")
        user_prompt = request.form.get("user_prompt", "")

        # Add user message to conversation history (for UI display)
        conversation.append({"role": "user", "content": user_prompt})
        session["conversation"] = conversation

        # Check and update vectorstore if needed
        check_and_update_vectorstore(KB_FOLDER)

        # Retrieve RAG context using both venue concept + user prompt
        rag_context = retrieve_codex_context(user_prompt, venue_concept)

        # Build full prompt
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Venue Concept: {venue_concept}\n"
            f"Relevant Context:\n{rag_context}\n\n"
            f"User Question: {user_prompt}"
        )

        # Call OpenAI API (system + single user turn)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7
        )

        ai_response = completion.choices[0].message.content

        # Store full assistant reply into session for UI display
        conversation.append({"role": "assistant", "content": ai_response})
        session["conversation"] = conversation

    return render_template("index.html", venue_concept="", conversation=conversation, ai_response=ai_response)


@app.route("/reset", methods=["POST"])
def reset():
    session["conversation"] = []
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)