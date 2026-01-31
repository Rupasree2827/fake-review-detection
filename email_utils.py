import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

ALERT_EMAIL = "pallemsupraja03@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "suprajapallem33@gmail.com"
SMTP_PASS = "ikcuyucdhkdfzaen"  # Use App Password for Gmail

def send_fake_review_alert(review_text, score, user_id="unknown", timestamp=None):
    subject = "⚠️ Fake Review Detected"
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_body = f"""
    <html>
        <body>
            <h2 style='color:red;'>🚨 Fake Review Alert</h2>
            <p><strong>User ID:</strong> {user_id}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Review:</strong> {review_text}</p>
            <p><strong>Confidence Score:</strong> {score:.2f}</p>
        </body>
    </html>
    """
    msg = MIMEMultipart("alternative")
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = ALERT_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print("✅ HTML alert email sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
