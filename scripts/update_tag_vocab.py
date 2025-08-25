import os
import json
import re
from pathlib import Path

# === Config ===
KB_ROOT = Path("knowledge_base")
OUT_DIR = Path("codex_faiss_index")
OUT_FILE = OUT_DIR / "vectorstore_manifest.json"

# Any .txt under KB is eligible
VALID_EXT = {".txt"}

# Normalization: lowercase, spaces/hyphens->underscores, strip quotes, keep a-z0-9_
def normalize_tag(tag: str) -> str:
    t = tag.strip().lower()
    # Replace curly quotes / weird punctuation with plain
    t = t.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    # Replace spaces and hyphens with underscores
    t = re.sub(r"[\s\-]+", "_", t)
    # Drop anything not a-z0-9_
    t = re.sub(r"[^a-z0-9_]", "", t)
    # Collapse multiple underscores
    t = re.sub(r"_+", "_", t).strip("_")
    return t

def extract_tags_from_file(p: Path):
    """Extract tags from a single .txt file by looking for a line that starts with 'Tags:'."""
    tags = []
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return tags

    # Find the first line starting with 'Tags:' (case-insensitive, allow leading spaces)
    for line in text.splitlines():
        if line.strip().lower().startswith("tags:"):
            raw = line.split(":", 1)[1]
            # Split by comma or semicolon
            parts = re.split(r"[;,]", raw)
            for part in parts:
                norm = normalize_tag(part)
                if norm:
                    tags.append(norm)
            break
    return tags

def collect_all_tags():
    all_tags = set()
    for p in KB_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            tags = extract_tags_from_file(p)
            all_tags.update(tags)
    return sorted(all_tags)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag_vocab = collect_all_tags()

    # If an existing manifest exists, merge (so we don't lose any curated values)
    manifest = {}
    if OUT_FILE.exists():
        try:
            manifest = json.loads(OUT_FILE.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    manifest["tag_vocab"] = tag_vocab

    OUT_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"✅ Wrote {len(tag_vocab)} tags to {OUT_FILE}")

if __name__ == "__main__":
    main()