from pathlib import Path
from dotenv import load_dotenv
import os
import smtplib
from email.message import EmailMessage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load environment variables from .env
load_dotenv()

# Define constants from environment
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL")

# === PDF GENERATION ===
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

# === EMAIL SENDING ===
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

# === SCENARIO MOD LOADER ===

# Folder where scenario-specific prompt mods are stored
MODS_FOLDER = os.path.join("knowledge_base", "training_modules", "system_prompt_mods")

# Keyword-to-file mapping (add more as you create mods)
scenario_mods = {
    "staff inconsistency": "system_prompt_mod_scenario_10.txt",
    "menu costing": "system_prompt_mod_scenario_08.txt",
    "opening a new bar": "system_prompt_mod_scenario_13.txt",
    "new restaurant": "system_prompt_mod_scenario_13.txt",
    "signature cocktail": "system_prompt_mod_scenario_12.txt",
    "cocktail consistency": "system_prompt_mod_scenario_10.txt",
    "bar program from scratch": "system_prompt_mod_scenario_13.txt",
    # Add more keyword:filename mappings as needed
}

import os, re

MODS_DIR = "system_prompt_mods"

_SCEN_NUM_RE      = re.compile(r"\bscenario[_\s-]*#?\s*(\d{1,3})\b", re.IGNORECASE)
_HEADER_TITLE_RE  = re.compile(r"^\s*title\s*:\s*(.+)$", re.IGNORECASE)
_HEADER_KEYS_RE   = re.compile(r"^\s*keywords\s*:\s*(.+)$", re.IGNORECASE)
_HEADER_TAGS_RE   = re.compile(r"^\s*tags\s*:\s*(.+)$", re.IGNORECASE)

def _parse_mod_file(path):
    """
    Returns (title:str, keywords:list[str], content:str).
    Accepts either 'Keywords:' or 'Tags:' as keyword sources.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        title, keys = "", []
        for i, raw in enumerate(lines[:5]):  # scan first few lines only
            line = raw.strip()
            if not title:
                m = _HEADER_TITLE_RE.match(line)
                if m: title = m.group(1).strip()
            m = _HEADER_KEYS_RE.match(line)
            if m:
                keys = [k.strip().lower() for k in m.group(1).split(",") if k.strip()]
            # tolerate existing "Tags:" in mod files
            if not keys:
                m = _HEADER_TAGS_RE.match(line)
                if m:
                    keys = [k.strip().lower() for k in m.group(1).replace(",", " ").split() if k.strip()]
        content = "".join(lines)
        return title, keys, content
    except Exception:
        return "", [], ""

def _score_keywords(keys, haystack):
    if not keys: return 0
    h = haystack.lower()
    score = 0
    for k in keys:
        # permit underscore, hyphen, or space variants
        variants = {k, k.replace("_", " "), k.replace("_", "-")}
        if any(v in h for v in variants):
            score += 1
    return score

def detect_scenario_prompt_mod(user_prompt: str, venue_prompt: str = "") -> str:
    """
    Auto-select a scenario system prompt mod.
    1) If user/venue mentions 'scenario N', load scenario_N file if present.
    2) Else, score all files by Keywords (or Tags) and pick the best.
       Requires at least one keyword hit to avoid random selection.
    Returns the file content or '' if none matched.
    """
    text = f"{user_prompt or ''} {venue_prompt or ''}"

    # 1) Explicit "scenario N" override
    m = _SCEN_NUM_RE.search(text)
    if m:
        scen = m.group(1)
        # Accept either naming convention
        candidates = [
            f"scenario_{scen}_prompt_mod.txt",
            f"system_prompt_mod_scenario_{scen}.txt",
        ]
        for fn in candidates:
            path = os.path.join(MODS_DIR, fn)
            if os.path.exists(path):
                _, _, content = _parse_mod_file(path)
                if content:
                    return content

    # 2) Keyword-based selection
    if not os.path.isdir(MODS_DIR):
        return ""
    best = ("", -1, "")  # (path, score, content)
    for fn in os.listdir(MODS_DIR):
        if not fn.lower().endswith(".txt"): 
            continue
        path = os.path.join(MODS_DIR, fn)
        title, keys, content = _parse_mod_file(path)
        score = _score_keywords(keys, text)
        if score > best[1]:
            best = (path, score, content)

    # Require ≥1 hit so we don't inject the wrong mod
    return best[2] if best[1] > 0 else ""

# Optional: which file matched (for debugging/QA)
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
        path = os.path.join(MODS_DIR, fn)
        _, keys, _ = _parse_mod_file(path)
        score = _score_keywords(keys, text)
        if score > best[1]:
            best = (fn, score)
    return best[0] if best[1] > 0 else ""