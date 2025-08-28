# utils.py  (patched for subfolders: lenses/ triggers/ preambles/)
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

# ------------------------------------------------------------
# Env
# ------------------------------------------------------------
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
# New folder layout
MODS_ROOT       = Path("system_prompt_mods")
MODS_DIR_LENSES = MODS_ROOT / "lenses"      # lens blocks + scenario-specific prompt mods live here
TRIGGERS_DIR    = MODS_ROOT / "triggers"    # one text file per lens family
PREAMBLES_DIR   = MODS_ROOT / "preambles"   # per-lens preambles

# Patterns for scenario-specific files
_SCEN_FILE_PATTERNS = (
    re.compile(r"^scenario_\d{1,3}_prompt_mod\.txt$", re.IGNORECASE),
    re.compile(r"^system_prompt_mod_scenario_\d{1,3}\.txt$", re.IGNORECASE),
)

_SCEN_NUM_RE      = re.compile(r"\bscenario[_\s-]*#?\s*(\d{1,3})\b", re.IGNORECASE)
_HEADER_TITLE_RE  = re.compile(r"^\s*title\s*:\s*(.+)$", re.IGNORECASE)
_HEADER_KEYS_RE   = re.compile(r"^\s*keywords\s*:\s*(.+)$", re.IGNORECASE)
_HEADER_TAGS_RE   = re.compile(r"^\s*tags\s*:\s*(.+)$", re.IGNORECASE)

def _parse_mod_file(path: Path) -> tuple[str, list[str], str]:
    """
    Returns (title, keywords, content). Accepts 'Keywords:' or 'Tags:' headers.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return "", [], ""
    title, keys = "", []
    for raw in lines[:6]:
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
    content = "\n".join(lines)
    return title, keys, content

def _score_keywords(keys: list[str], haystack: str) -> int:
    if not keys: return 0
    h = haystack.lower()
    score = 0
    for k in keys:
        variants = {k, k.replace("_", " "), k.replace("_", "-")}
        if any(v in h for v in variants):
            score += 1
    return score

def _is_scenario_mod_filename(name: str) -> bool:
    return any(pat.match(name) for pat in _SCEN_FILE_PATTERNS)

def detect_scenario_prompt_mod(user_prompt: str, venue_prompt: str = "") -> str:
    """
    Auto-select a scenario system prompt mod from /lenses.
    1) If user/venue mentions 'scenario N', load scenario_N file if present.
    2) Else, score scenario-* files by Keywords/Tags and pick the best (≥1 hit).
    Returns the file content or '' if none matched.
    """
    text = f"{user_prompt or ''} {venue_prompt or ''}"

    # 1) Explicit "scenario N" override
    m = _SCEN_NUM_RE.search(text)
    if m:
        scen = m.group(1)
        for fn in (f"scenario_{scen}_prompt_mod.txt", f"system_prompt_mod_scenario_{scen}.txt"):
            path = MODS_DIR_LENSES / fn
            if path.exists():
                _, _, content = _parse_mod_file(path)
                if content:
                    return content

    # 2) Keyword-based selection across scenario-* files in /lenses
    if not MODS_DIR_LENSES.is_dir():
        return ""
    best_path, best_score, best_content = "", -1, ""
    for fn in os.listdir(MODS_DIR_LENSES):
        if not fn.lower().endswith(".txt"):
            continue
        if not _is_scenario_mod_filename(fn):
            # ignore general lens blocks here
            continue
        path = MODS_DIR_LENSES / fn
        title, keys, content = _parse_mod_file(path)
        score = _score_keywords(keys, text)
        if score > best_score:
            best_path, best_score, best_content = str(path), score, content

    return best_content if best_score > 0 else ""

def detect_scenario_prompt_mod_name(user_prompt: str, venue_prompt: str = "") -> str:
    text = f"{user_prompt or ''} {venue_prompt or ''}"
    m = _SCEN_NUM_RE.search(text)
    if m:
        scen = m.group(1)
        for fn in (f"scenario_{scen}_prompt_mod.txt", f"system_prompt_mod_scenario_{scen}.txt"):
            if (MODS_DIR_LENSES / fn).exists():
                return fn
    if not MODS_DIR_LENSES.is_dir():
        return ""
    best_fn, best_score = "", -1
    for fn in os.listdir(MODS_DIR_LENSES):
        if not fn.lower().endswith(".txt"):
            continue
        if not _is_scenario_mod_filename(fn):
            continue
        path = MODS_DIR_LENSES / fn
        _, keys, _ = _parse_mod_file(path)
        score = _score_keywords(keys, text)
        if score > best_score:
            best_fn, best_score = fn, score
    return best_fn if best_score > 0 else ""

# =========================
# Lens triggers + preambles
# =========================

@lru_cache(maxsize=1)
def _load_trigger_patterns() -> dict[str, re.Pattern]:
    """
    Load trigger files from system_prompt_mods/triggers/*.txt.
    Returns dict like {"portfolio": compiled_regex, "layout": compiled_regex, ...}
    (The key is based on filename stem.)
    """
    patterns: dict[str, re.Pattern] = {}
    if not TRIGGERS_DIR.exists():
        return patterns
    for p in TRIGGERS_DIR.glob("*.txt"):
        key = p.stem.lower().strip()  # e.g. "portfolio", "layout", "event", "guest_experience"
        terms = []
        for raw in p.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            # If it looks like a regex, keep it; else treat as literal
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

def _read_lens_text(name: str) -> Optional[str]:
    """
    Read a lens block from /lenses given a base name (e.g., 'layout_lens').
    """
    p = MODS_DIR_LENSES / f"{name}.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else None

def _read_preamble_text(name: str) -> Optional[str]:
    """
    Read a preamble from /preambles given a base name (e.g., 'layout_preamble').
    """
    p = PREAMBLES_DIR / f"{name}.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else None

def detect_lens_mods(user_prompt: str, venue_prompt: str = "") -> Tuple[List[str], str]:
    """
    Returns (lens_names, concatenated_lens_blocks_text).
    Lens names map to lens files in /lenses. Multiple may fire.
    The mapping from trigger filename stem -> lens txt filename is 1:1 with '_lens' suffix.
      e.g., 'layout' trigger => 'layout_lens.txt'
    """
    text = f"{user_prompt or ''}\n{venue_prompt or ''}"
    if not text:
        return ([], "")

    patterns = _load_trigger_patterns()
    lens_names: List[str] = []
    blocks: List[str] = []

    for key, rx in patterns.items():
        if rx.search(text):
            lens_names.append(key)
            lens_file_basename = f"{key}_lens"
            block = _read_lens_text(lens_file_basename)
            if block:
                blocks.append(block)

    return (lens_names, "\n\n".join(blocks).strip() if blocks else "")

def lens_preamble_for(lens_names: List[str]) -> str:
    """
    Given matched lens keys (e.g., ["layout","event","guest_experience"]), return the concatenated
    preamble text to append to the *user* prompt. We look up '{key}_preamble.txt' in /preambles.
    """
    pieces: List[str] = []
    for key in lens_names:
        preamble_name = f"{key}_preamble"
        txt = _read_preamble_text(preamble_name)
        if txt:
            pieces.append(txt)
    return "\n\n".join(pieces).strip()