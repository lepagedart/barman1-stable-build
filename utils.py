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

# Function to generate a simple PDF file
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

# Function to send an email with optional PDF attachment
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