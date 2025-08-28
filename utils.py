from __future__ import annotations
import os
import re
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple, Optional

from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load environment variables from .env
load_dotenv()

# =========================
# Email + PDF utilities
# =========================
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL")

def generate_pdf(text, output_path="static/cocktail_response.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    text_object = c.beginText(40, height - 50)
    text_object.setFont("Helvetica", 12)
    for line in text.split("\n"):
        text_object.textLine(line.strip())
    c.drawText(text_object)
    c.save()
    return output_path

def send_email(recipient, subject, body, attachment_path="static/cocktail_response.pdf"):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM_EMAIL
    msg["To"] = recipient
    msg.set_content(body)

    try:
        with open(attachment_path, "rb") as f:
            file_data = f.read()
            file_name = Path(attachment_path).name
        msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)
    except FileNotFoundError:
        print(f"Attachment not found: {attachment_path}")

    if not SMTP_SERVER:
        raise ValueError("SMTP_SERVER environment variable is not set.")
    if SMTP_USERNAME is None or SMTP_PASSWORD is None:
        raise ValueError("SMTP_USERNAME and SMTP_PASSWORD environment variables must be set.")
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
        smtp.send_message(msg)

# =========================
# Scenario Prompt Mods (KB)
# =========================
MODS_DIR = "system_prompt_mods"

_SCEN_NUM_RE      = re.compile(r"\bscenario[_\s-]*#?\s*(\d{1,3})\b", re.IGNORECASE)
_HEADER_TITLE_RE  = re.compile(r"^\s*title\s*:\s*(.+)$", re.IGNORECASE)
_HEADER_KEYS_RE   = re.compile(r"^\s*keywords\s*:\s*(.+)$", re.IGNORECASE)
_HEADER_TAGS_RE   = re.compile(r"^\s*tags\s*:\s*(.+)$", re.IGNORECASE)

def _parse_mod_file(path: str) -> tuple[str, list[str], str]:
    """
    Returns (title, keywords, content). Accepts 'Keywords:' or 'Tags:' headers.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        title, keys = "", []
        for raw in lines[:5]:
            line = raw.strip()
            if not title:
                m = _HEADER_TITLE_RE.match(line)
                if m: title = m.group(1).strip()
            m = _HEADER_KEYS_RE.match(line)
            if m:
                keys = [k.strip().lower() for k in m.group(1).split(",") if k.strip()]
            if not keys:
                m = _HEADER_TAGS_RE.match(line)
                if m:
                    keys = [k.strip().lower() for k in m.group(1).replace(",", " ").split() if k.strip()]
        content = "".join(lines)
        return title, keys, content
    except Exception:
        return "", [], ""

def _score_keywords(keys: list[str], haystack: str) -> int:
    if not keys: return 0
    h = haystack.lower()
    score = 0
    for k in keys:
        variants = {k, k.replace("_", " "), k.replace("_", "-")}
        if any(v in h for v in variants):
            score += 1
    return score

def detect_scenario_prompt_mod(user_prompt: str, venue_prompt: str = "") -> str:
    """
    Auto-select a scenario system prompt mod.
    1) If user/venue mentions 'scenario N', load scenario_N file if present.
    2) Else, score all files by Keywords/Tags and pick the best (≥1 hit).
    Returns the file content or '' if none matched.
    """
    text = f"{user_prompt or ''} {venue_prompt or ''}"

    # 1) Explicit "scenario N" override
    m = _SCEN_NUM_RE.search(text)
    if m:
        scen = m.group(1)
        for fn in (f"scenario_{scen}_prompt_mod.txt", f"system_prompt_mod_scenario_{scen}.txt"):
            path = os.path.join(MODS_DIR, fn)
            if os.path.exists(path):
                _, _, content = _parse_mod_file(path)
                if content:
                    return content

    # 2) Keyword-based selection
    if not os.path.isdir(MODS_DIR):
        return ""
    best = ("", -1, "")
    for fn in os.listdir(MODS_DIR):
        if not fn.lower().endswith(".txt"):
            continue
        if fn.startswith("layout_") or fn.startswith("event_") or fn.endswith("_lens.txt"):
            # lens/preamble files live here too; skip for scenario selection
            continue
        path = os.path.join(MODS_DIR, fn)
        title, keys, content = _parse_mod_file(path)
        score = _score_keywords(keys, text)
        if score > best[1]:
            best = (path, score, content)
    return best[2] if best[1] > 0 else ""

def detect_scenario_prompt_mod_name(user_prompt: str, venue_prompt: str = "") -> str:
    text = f"{user_prompt or ''} {venue_prompt or ''}"
    m = _SCEN_NUM_RE.search(text)
    if m:
        scen = m.group(1)
        for fn in (f"scenario_{scen}_prompt_mod.txt", f"system_prompt_mod_scenario_{scen}.txt"):
            if os.path.exists(os.path.join(MODS_DIR, fn)):
                return fn
    if not os.path.isdir(MODS_DIR):
        return ""
    best = ("", -1)
    for fn in os.listdir(MODS_DIR):
        if not fn.lower().endswith(".txt"):
            continue
        if fn.startswith("layout_") or fn.startswith("event_") or fn.endswith("_lens.txt"):
            continue
        path = os.path.join(MODS_DIR, fn)
        _, keys, _ = _parse_mod_file(path)
        score = _score_keywords(keys, text)
        if score > best[1]:
            best = (fn, score)
    return best[0] if best[1] > 0 else ""

# =========================
# Lens triggers + preambles
# =========================
TRIGGERS_DIR = Path(MODS_DIR) / "triggers"

@lru_cache(maxsize=1)
def _load_trigger_patterns() -> dict[str, re.Pattern]:
    """
    Load trigger files from system_prompt_mods/triggers/*.txt.
    Returns dict like {"portfolio": compiled_regex, "layout": compiled_regex, ...}
    """
    patterns: dict[str, re.Pattern] = {}
    if not TRIGGERS_DIR.exists():
        return patterns
    for p in TRIGGERS_DIR.glob("*.txt"):
        key = p.stem.lower().strip()  # e.g. "portfolio", "layout", "event_setup"
        terms = []
        for raw in p.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            # Treat each line as a literal term unless it looks like a regex
            if s.startswith("(?") or s.endswith("$") or ("|" in s):
                terms.append(s)
            else:
                terms.append(re.escape(s))
        if not terms:
            continue
        combined = r"(?i)\b(?:{})\b".format("|".join(terms))
        try:
            patterns[key] = re.compile(combined)
        except re.error:
            patterns[key] = re.compile("|".join(terms), re.IGNORECASE)
    return patterns

def _read_mod_text(name: str) -> Optional[str]:
    path = Path(MODS_DIR) / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None

def detect_lens_mods(user_prompt: str, venue_prompt: str = "") -> Tuple[List[str], str]:
    """
    Returns (lens_names, concatenated_lens_blocks_text).
    Lens names map to files:
      portfolio -> portfolio_lens.txt
      layout    -> layout_lens.txt
      event_setup -> event_setup_lens.txt
    Multiple lenses may fire; we join blocks with blank lines.
    """
    text = f"{user_prompt or ''}\n{venue_prompt or ''}".lower()
    if not text:
        return ([], "")

    patterns = _load_trigger_patterns()
    key_to_block = {
        "portfolio": "portfolio_lens",
        "layout": "layout_lens",
        "event_setup": "event_setup_lens",
    }

    lens_names: List[str] = []
    blocks: List[str] = []
    for key, rx in patterns.items():
        if key in key_to_block and rx.search(text):
            lens_names.append(key)
            block = _read_mod_text(key_to_block[key])
            if block:
                blocks.append(block)

    return (lens_names, "\n\n".join(blocks).strip() if blocks else "")

def lens_preamble_for(lens_names: List[str]) -> str:
    """
    Given matched lens keys (e.g., ["layout","event_setup"]), return the concatenated
    preamble text to append to the *user* prompt.
    """
    key_to_preamble = {
        "layout": "layout_preamble",
        "event_setup": "event_preamble",
        # portfolio typically does not need a preamble, but you can add one:
        # "portfolio": "portfolio_preamble",
    }
    pieces: List[str] = []
    for key in lens_names:
        name = key_to_preamble.get(key)
        if not name:
            continue
        txt = _read_mod_text(name)
        if txt:
            pieces.append(txt)
    return "\n\n".join(pieces).strip()