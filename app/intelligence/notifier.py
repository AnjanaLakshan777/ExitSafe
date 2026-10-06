"""Email alerts for newly detected market threats (SMTP, e.g. Gmail with an app password)."""

import html
import smtplib
from email.message import EmailMessage

from app.data.schemas.event_schema import VerificationStatus

DISCLAIMER = ("Detected automatically from news headlines by keyword matching. A news or "
              "search report is a claim, not a confirmed fact. Not financial advice.")


def _status_note(event):
    if event.verification_status is VerificationStatus.CONFIRMED:
        return "official source"
    if event.verification_status is VerificationStatus.REPORTED:
        return "news report (unconfirmed)"
    return "unverified search result"


def build_alert_email(records, sender, recipients):
    """One digest email for all ``records`` (``ThreatRecord``s), most severe first."""
    worst = records[0].event.severity
    message = EmailMessage()
    message["Subject"] = (f"[ExitSafe] {len(records)} world-market threat"
                          f"{'s' if len(records) != 1 else ''} detected ({worst.value})")
    message["From"] = sender
    message["To"] = ", ".join(recipients)

    lines, rows = [], []
    for record in records:
        event = record.event
        when = event.published_time.strftime("%Y-%m-%d %H:%M UTC")
        terms = ", ".join(record.threat_terms)
        lines += [f"[{event.severity.value}] {event.title}",
                  f"  {event.source_name} ({_status_note(event)}), {when}",
                  f"  Triggered by: {terms}",
                  f"  {event.source_url or 'no link'}", ""]
        link = (f'<a href="{html.escape(event.source_url)}">{html.escape(event.title)}</a>'
                if event.source_url else html.escape(event.title))
        rows.append(f"<tr><td><b>{event.severity.value}</b></td><td>{link}<br>"
                    f"<small>{html.escape(event.source_name)} &middot; {_status_note(event)} "
                    f"&middot; {when}<br>Triggered by: {html.escape(terms)}</small></td></tr>")

    message.set_content("\n".join(lines + [DISCLAIMER]))
    message.add_alternative(
        "<html><body><h3>ExitSafe market-threat alert</h3>"
        f"<table cellpadding='6' border='1' style='border-collapse:collapse'>{''.join(rows)}</table>"
        f"<p><small>{html.escape(DISCLAIMER)}</small></p></body></html>", subtype="html")
    return message


def send_alert_email(records, settings):
    """Send the digest. Port 465 uses SSL; any other port uses STARTTLS."""
    message = build_alert_email(records, settings.smtp_user, settings.alert_email_to)
    if settings.smtp_port == 465:
        server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=30)
    else:
        server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)
        server.starttls()
    with server:
        server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(message)
