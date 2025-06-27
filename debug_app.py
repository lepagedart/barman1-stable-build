import os
import json
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv

print("🔄 Starting app...")

# Load environment variables
load_dotenv()
print("✅ Environment loaded")

# Flask setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
print("✅ Flask app created")

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

@app.route("/")
def index():
    print("📄 Index route called")
    try:
        return render_template("index.html", conversation=[])
    except Exception as e:
        print(f"❌ Template error: {e}")
        return f"<h1>Template Error</h1><p>{str(e)}</p>"

@app.route("/test")
def test():
    return {"status": "working", "message": "App is running"}

@app.route("/api_test")
def api_test():
    try:
        client = get_openai_client()
        return {"openai": "client_created", "status": "ok"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5001)
