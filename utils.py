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

def detect_scenario_prompt_mod(user_prompt: str) -> str:
    """Return scenario mod string if prompt matches a known scenario keyword"""
    for keyword, filename in scenario_mods.items():
        if keyword.lower() in user_prompt.lower():
            mod_path = os.path.join(MODS_FOLDER, filename)
            try:
                with open(mod_path, "r") as f:
                    print(f"📎 Scenario mod applied: {filename}")
                    return f.read()
            except FileNotFoundError:
                print(f"⚠️ Scenario mod file not found: {mod_path}")
    return ""