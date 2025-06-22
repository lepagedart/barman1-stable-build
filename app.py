import os
import sys
import uuid
from flask import Flask, render_template, request, redirect, session
from flask_session import Session
from dotenv import load_dotenv
from openai import OpenAI
from rag_retriever import retrieve_codex_context, check_and_update_vectorstore
from utils import generate_pdf, send_email

load_dotenv()

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    conversation = session.get("conversation", [])

    # 🧠 Run vectorstore sync check before handling user request
    check_and_update_vectorstore()

    if request.method == "POST":
        venue = request.form.get("concept", "")
        user_prompt = request.form.get("user_prompt", "")
        email = request.form.get("email", "")

        conversation.append({"role": "user", "content": f"Venue concept: {venue}\n{user_prompt}"})

        rag_context = retrieve_codex_context(user_prompt)
        conversation.insert(-1, {"role": "system", "content": f"Helpful reference:\n{rag_context}"})

        with open("system_prompt.txt", "r") as file:
            system_prompt = file.read()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                *session.get("chat_history", [])
            ]
        )
        result = response.choices[0].message.content.strip()

        conversation.append({"role": "assistant", "content": result})
        session["conversation"] = conversation

        if email:
            generate_pdf(result)
            send_email(email, "Raise the Bar - Cocktail Response", result)

    return render_template("index.html", result=result, conversation=conversation)

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)