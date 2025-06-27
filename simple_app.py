import os
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

# OpenAI client setup
client = None

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
    if "conversation" not in session:
        session["conversation"] = []

    if request.method == "POST":
        venue = request.form.get("venue_concept", "")
        user_prompt = request.form.get("user_prompt", "")
        
        print(f"🔄 Processing request: venue='{venue}', prompt='{user_prompt}'")
        
        try:
            print(f"🔄 Getting OpenAI client...")
            openai_client = get_openai_client()
            print(f"✅ OpenAI client ready")
            
            print(f"🔄 Making API call...")
            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a helpful bartender for a {venue}. Keep responses concise."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200
            )
            
            response_text = completion.choices[0].message.content.strip()
            print(f"✅ OpenAI response received: {len(response_text)} characters")
            
        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
            response_text = f"Error: {str(e)}"
        
        conversation = session["conversation"]
        conversation.append({"role": "user", "content": user_prompt})
        conversation.append({"role": "assistant", "content": response_text})
        session["conversation"] = conversation
        
        # Return JSON response for AJAX requests
        return jsonify({"response": response_text})

    return render_template("index.html", conversation=session["conversation"])

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return jsonify({"response": "Conversation reset."})

if __name__ == "__main__":
    print("Starting simple app...")
    app.run(debug=True, port=5000)
