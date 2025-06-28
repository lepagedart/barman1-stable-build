#!/bin/bash

echo "🔁 Resetting Barman-1 environment..."

# Remove Python cache and previous vectorstore
echo "🧹 Cleaning up __pycache__ and vectorstore..."
rm -rf __pycache__ vectorstore codex/__pycache__ codex_faiss_index/

# Rebuild virtual environment (optional)
if [ -d ".venv" ]; then
  echo "📦 Removing existing virtual environment..."
  rm -rf .venv
fi

echo "🐍 Creating new virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "📦 Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Rebuild vectorstore
echo "🧠 Rebuilding FAISS vectorstore..."
python rag_loader.py

echo "✅ Environment reset complete. You’re ready to launch Barman-1."