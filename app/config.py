import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL   = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
VIOLATION_MIN  = int(os.getenv("VIOLATION_THRESHOLD_MIN",60))
PRIMARY_EMAIL  = os.getenv("PRIMARY_EMAIL")
SMTP_HOST      = os.getenv("SMTP_HOST")
SMTP_PORT      = int(os.getenv("SMTP_PORT",465))
SMTP_USER      = os.getenv("SMTP_USER")
SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD")
ALLOWED_DEVICES = {
    int(d) for d in os.getenv("ALLOWED_DEVICES", "").split(",") if d.isdigit()
}
