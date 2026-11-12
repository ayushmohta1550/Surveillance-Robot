from __future__ import annotations

import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from .render import RenderedEmail
from ..core.config import RECIPIENT


def build_message(rendered: RenderedEmail, attachment: str | Path | None = None) -> EmailMessage:
    sender = os.environ.get("SURVEILLANCE_EMAIL_FROM", "")
    message = EmailMessage()
    message["Subject"] = rendered.subject
    message["From"] = sender or "Not configured"
    message["To"] = RECIPIENT
    message.set_content(rendered.text)
    message.add_alternative(rendered.html, subtype="html")
    if attachment and Path(attachment).is_file():
        path = Path(attachment)
        media_type, _ = mimetypes.guess_type(path.name)
        major, minor = (media_type or "application/octet-stream").split("/", 1)
        message.add_attachment(path.read_bytes(), maintype=major, subtype=minor, filename=path.name)
    return message


def send_message(message: EmailMessage) -> None:
    host = os.environ.get("SURVEILLANCE_SMTP_HOST")
    port = int(os.environ.get("SURVEILLANCE_SMTP_PORT", "587"))
    username = os.environ.get("SURVEILLANCE_SMTP_USERNAME")
    password = os.environ.get("SURVEILLANCE_SMTP_PASSWORD")
    sender = os.environ.get("SURVEILLANCE_EMAIL_FROM")
    missing = [name for name, value in {
        "SURVEILLANCE_SMTP_HOST": host,
        "SURVEILLANCE_SMTP_USERNAME": username,
        "SURVEILLANCE_SMTP_PASSWORD": password,
        "SURVEILLANCE_EMAIL_FROM": sender,
    }.items() if not value]
    if missing:
        raise RuntimeError("Missing email environment variables: " + ", ".join(missing))
    message.replace_header("From", sender)
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(username, password)
        smtp.send_message(message)
