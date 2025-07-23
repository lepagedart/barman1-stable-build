
from pathlib import Path

def load_knowledge_documents(base_dir="knowledge_base"):
    """
    Walk the knowledge_base/ folder and load .txt files with metadata.
    Returns a list of (content, metadata) tuples.
    """
    documents = []
    base_path = Path(base_dir)

    for file_path in base_path.rglob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                metadata = {
                    "source": str(file_path),
                    "category": file_path.parent.name,
                    "filename": file_path.stem
                }
                documents.append((content, metadata))
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")

    print(f"✅ Loaded {len(documents)} documents from {base_dir}")
    return documents
