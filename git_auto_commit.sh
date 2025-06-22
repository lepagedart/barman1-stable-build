#!/bin/bash

# Usage: ./git_auto_commit.sh "Your commit message"

if [ -z "$1" ]; then
  echo "❌ Please provide a commit message."
  exit 1
fi

echo "📦 Staging all changes..."
git add .

echo "✍️ Committing with message: $1"
git commit -m "$1"

echo "🚀 Pushing to GitHub..."
git push origin main